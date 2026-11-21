"""Inference queue balancer.

Balances inference requests across:
  - deadlines (earliest-deadline-first urgency)
  - token budgets (bounded short-job boost, so long requests are not
    permanently blocked by short ones)
  - tenant weights (weighted fair sharing, so a bursting high-priority
    tenant cannot starve low-priority tenants indefinitely)
  - VRAM capacity (greedy batch fill with skip)
  - queue wait time (aging, guaranteeing starvation-free scheduling)

Compatible with Python 3.8.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# Degradation / deferral reasons.
REASON_VRAM_EXCEEDS_CAPACITY = "vram_exceeds_capacity"
REASON_DEADLINE_MISSED = "deadline_missed"
REASON_NO_VRAM_HEADROOM = "no_vram_headroom"
REASON_BATCH_FULL = "batch_full"


@dataclass
class Request:
    """A single queued inference request.

    Attributes:
        request_id: Unique identifier.
        tenant_id: Owning tenant.
        tenant_weight: Priority weight of the tenant (> 0). Higher means
            more entitled share of the scheduler.
        token_budget: Estimated tokens the request will consume.
        vram_mb: VRAM the request needs while executing.
        deadline: Absolute timestamp (same clock as ``now``) by which the
            request should start executing.
        enqueued_at: Absolute timestamp when the request entered the queue.
    """

    request_id: str
    tenant_id: str
    tenant_weight: float
    token_budget: float
    vram_mb: float
    deadline: float
    enqueued_at: float

    def __post_init__(self) -> None:
        if self.tenant_weight <= 0:
            raise ValueError("tenant_weight must be > 0")
        if self.token_budget <= 0:
            raise ValueError("token_budget must be > 0")
        if self.vram_mb <= 0:
            raise ValueError("vram_mb must be > 0")


@dataclass
class ScoredRequest:
    """A request annotated with its scheduling score (for observability)."""

    request: Request
    score: float
    wait_time: float


@dataclass
class BatchPlan:
    """Result of one scheduling round.

    Attributes:
        ordered: Requests to execute next, in execution order.
        deferred: (request, reason) pairs not scheduled this round.
        vram_used_mb: Total VRAM reserved by ``ordered``.
    """

    ordered: List[ScoredRequest] = field(default_factory=list)
    deferred: List[Tuple[Request, str]] = field(default_factory=list)
    vram_used_mb: float = 0.0


class QueueBalancer:
    """Starvation-free weighted-fair queue balancer.

    Score model (higher runs first)::

        aging    = 1 + aging_rate * wait_time
        urgency  = deadline_urgency_weight / max(slack, slack_floor)
        short    = short_job_boost / (1 + token_budget / token_scale)
        fairness = 1 / (1 + tenant_service[tenant] / tenant_weight)

        score    = tenant_weight * aging * (1 + urgency + short) * fairness

    Properties:
      - ``aging`` grows without bound, so any queued request eventually
        outranks newly arriving ones: no permanent blocking, no starvation.
      - ``short`` is bounded in ``(0, short_job_boost]``, so short jobs get
        a finite boost only and aging always dominates in the limit.
      - ``fairness`` decays as a tenant consumes service, so a bursting
        tenant's marginal priority drops toward other tenants' levels.
    """

    def __init__(
        self,
        vram_capacity_mb: float,
        max_batch_size: int = 32,
        aging_rate: float = 0.1,
        deadline_urgency_weight: float = 2.0,
        slack_floor: float = 0.001,
        short_job_boost: float = 1.0,
        token_scale: float = 1024.0,
        service_decay: float = 0.5,
        expired_policy: str = "drop",
    ) -> None:
        if vram_capacity_mb <= 0:
            raise ValueError("vram_capacity_mb must be > 0")
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be > 0")
        if expired_policy not in ("drop", "best_effort"):
            raise ValueError("expired_policy must be 'drop' or 'best_effort'")
        self.vram_capacity_mb = float(vram_capacity_mb)
        self.max_batch_size = int(max_batch_size)
        self.aging_rate = float(aging_rate)
        self.deadline_urgency_weight = float(deadline_urgency_weight)
        self.slack_floor = float(slack_floor)
        self.short_job_boost = float(short_job_boost)
        self.token_scale = float(token_scale)
        self.service_decay = float(service_decay)
        self.expired_policy = expired_policy
        # Exponentially decayed tokens served per tenant (fairness memory).
        self._tenant_service: Dict[str, float] = {}

    def score(self, request: Request, now: float) -> float:
        wait_time = max(0.0, now - request.enqueued_at)
        aging = 1.0 + self.aging_rate * wait_time
        slack = request.deadline - now
        urgency = self.deadline_urgency_weight / max(slack, self.slack_floor)
        short = self.short_job_boost / (1.0 + request.token_budget / self.token_scale)
        service = self._tenant_service.get(request.tenant_id, 0.0)
        fairness = 1.0 / (1.0 + service / request.tenant_weight)
        return request.tenant_weight * aging * (1.0 + urgency + short) * fairness

    def plan_batch(self, requests: List[Request], now: float) -> BatchPlan:
        """Pick the next batch from ``requests`` at time ``now``.

        Applies degradation rules, scores the survivors, then greedily
        fills the batch by descending score subject to VRAM and batch-size
        limits. Updates tenant service counters for scheduled requests.
        """
        plan = BatchPlan()
        candidates: List[Request] = []

        for req in requests:
            if req.vram_mb > self.vram_capacity_mb:
                plan.deferred.append((req, REASON_VRAM_EXCEEDS_CAPACITY))
            elif now > req.deadline and self.expired_policy == "drop":
                plan.deferred.append((req, REASON_DEADLINE_MISSED))
            else:
                candidates.append(req)

        scored = [
            ScoredRequest(
                request=req,
                score=self.score(req, now),
                wait_time=max(0.0, now - req.enqueued_at),
            )
            for req in candidates
        ]
        # Deterministic tie-break: higher score, then earlier enqueue, then id.
        scored.sort(
            key=lambda s: (-s.score, s.request.enqueued_at, s.request.request_id)
        )

        vram_left = self.vram_capacity_mb
        slots_left = self.max_batch_size
        for item in scored:
            req = item.request
            if slots_left <= 0:
                plan.deferred.append((req, REASON_BATCH_FULL))
                continue
            if req.vram_mb > vram_left:
                plan.deferred.append((req, REASON_NO_VRAM_HEADROOM))
                continue
            plan.ordered.append(item)
            plan.vram_used_mb += req.vram_mb
            vram_left -= req.vram_mb
            slots_left -= 1

        self._record_service(plan)
        return plan

    def _record_service(self, plan: BatchPlan) -> None:
        for tenant in list(self._tenant_service):
            self._tenant_service[tenant] *= self.service_decay
        for item in plan.ordered:
            req = item.request
            self._tenant_service[req.tenant_id] = (
                self._tenant_service.get(req.tenant_id, 0.0) + req.token_budget
            )

    def tenant_service(self, tenant_id: str) -> float:
        return self._tenant_service.get(tenant_id, 0.0)


def jain_fairness_index(throughputs: List[float]) -> float:
    """Jain's fairness index over per-tenant served tokens.

    Returns 1.0 for perfectly equal shares; 1/n for maximally unfair.
    Empty input is defined as perfectly fair.
    """
    values = [t for t in throughputs if t >= 0]
    if not values:
        return 1.0
    total = sum(values)
    if total == 0:
        return 1.0
    sum_sq = sum(t * t for t in values)
    return (total * total) / (len(values) * sum_sq)


def weight_normalized_fairness(
    served: Dict[str, float], weights: Dict[str, float]
) -> float:
    """Jain's index over served/weight ratios (weighted fairness)."""
    ratios = [
        served[t] / weights[t]
        for t in weights
        if weights[t] > 0 and served.get(t, 0.0) > 0
    ]
    return jain_fairness_index(ratios)
