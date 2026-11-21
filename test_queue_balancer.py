"""Unit tests for queue_balancer (Python 3.8.2 compatible)."""

import unittest

from queue_balancer import (
    BatchPlan,
    QueueBalancer,
    REASON_BATCH_FULL,
    REASON_DEADLINE_MISSED,
    REASON_NO_VRAM_HEADROOM,
    REASON_VRAM_EXCEEDS_CAPACITY,
    Request,
    jain_fairness_index,
    weight_normalized_fairness,
)


def make_request(
    request_id,
    tenant_id="t1",
    tenant_weight=1.0,
    token_budget=100.0,
    vram_mb=10.0,
    deadline=1000.0,
    enqueued_at=0.0,
):
    return Request(
        request_id=request_id,
        tenant_id=tenant_id,
        tenant_weight=tenant_weight,
        token_budget=token_budget,
        vram_mb=vram_mb,
        deadline=deadline,
        enqueued_at=enqueued_at,
    )


def scheduled_ids(plan):
    return [item.request.request_id for item in plan.ordered]


def deferred_reasons(plan, request_id):
    return [reason for req, reason in plan.deferred if req.request_id == request_id]


class TestVramConstraints(unittest.TestCase):
    def test_request_larger_than_capacity_is_rejected(self):
        balancer = QueueBalancer(vram_capacity_mb=80.0)
        too_big = make_request("huge", vram_mb=160.0)
        plan = balancer.plan_batch([too_big], now=0.0)
        self.assertEqual(plan.ordered, [])
        self.assertEqual(
            deferred_reasons(plan, "huge"), [REASON_VRAM_EXCEEDS_CAPACITY]
        )

    def test_batch_never_exceeds_vram_capacity(self):
        balancer = QueueBalancer(vram_capacity_mb=100.0)
        reqs = [
            make_request("a", vram_mb=60.0),
            make_request("b", vram_mb=60.0),
            make_request("c", vram_mb=40.0),
        ]
        plan = balancer.plan_batch(reqs, now=0.0)
        self.assertLessEqual(plan.vram_used_mb, 100.0)
        # 60 + 40 fits; the second 60 MB request must wait.
        self.assertEqual(sorted(scheduled_ids(plan)), ["a", "c"])
        self.assertEqual(deferred_reasons(plan, "b"), [REASON_NO_VRAM_HEADROOM])

    def test_batch_size_limit_defers_overflow(self):
        balancer = QueueBalancer(vram_capacity_mb=1000.0, max_batch_size=2)
        reqs = [make_request(str(i)) for i in range(5)]
        plan = balancer.plan_batch(reqs, now=0.0)
        self.assertEqual(len(plan.ordered), 2)
        self.assertEqual(len(plan.deferred), 3)
        for _, reason in plan.deferred:
            self.assertEqual(reason, REASON_BATCH_FULL)


class TestTenantBurst(unittest.TestCase):
    def test_bursting_high_priority_tenant_cannot_starve_low_tenant(self):
        balancer = QueueBalancer(
            vram_capacity_mb=10000.0, max_batch_size=2, service_decay=0.5
        )
        burst = [
            make_request(
                "hp-%d" % i, tenant_id="hp", tenant_weight=10.0, token_budget=100.0
            )
            for i in range(30)
        ]
        low = [make_request("lp-0", tenant_id="lp", tenant_weight=1.0)]
        queue = burst + low

        scheduled_round = None
        for round_no in range(10):
            plan = balancer.plan_batch(queue, now=float(round_no))
            done = set(scheduled_ids(plan))
            if "lp-0" in done:
                scheduled_round = round_no
                break
            queue = [req for req in queue if req.request_id not in done]

        self.assertIsNotNone(
            scheduled_round, "low-priority tenant was never scheduled"
        )
        self.assertLessEqual(scheduled_round, 5)

    def test_weighted_fairness_improves_after_burst(self):
        balancer = QueueBalancer(
            vram_capacity_mb=10000.0, max_batch_size=2, service_decay=0.5
        )
        weights = {"hp": 10.0, "lp": 1.0}
        queue = [
            make_request("hp-%d" % i, tenant_id="hp", tenant_weight=10.0)
            for i in range(10)
        ] + [make_request("lp-%d" % i, tenant_id="lp", tenant_weight=1.0)
             for i in range(2)]

        served = {"hp": 0.0, "lp": 0.0}
        for round_no in range(6):
            plan = balancer.plan_batch(queue, now=float(round_no))
            done = set()
            for item in plan.ordered:
                served[item.request.tenant_id] += item.request.token_budget
                done.add(item.request.request_id)
            queue = [req for req in queue if req.request_id not in done]

        # The low-weight tenant must have received some service despite the burst.
        self.assertGreater(served["lp"], 0.0)
        fairness = weight_normalized_fairness(served, weights)
        self.assertGreater(fairness, 0.5)


class TestLongRequestStarvation(unittest.TestCase):
    def test_aging_guarantees_long_request_eventually_runs(self):
        balancer = QueueBalancer(
            vram_capacity_mb=10000.0,
            max_batch_size=1,
            aging_rate=1.0,
            service_decay=0.5,
        )
        long_req = make_request(
            "long",
            tenant_id="bulk",
            tenant_weight=1.0,
            token_budget=100000.0,
            deadline=10 ** 9,
        )
        scheduled_at = None
        for round_no in range(200):
            # A fresh short, high-priority request arrives every round.
            short = make_request(
                "short-%d" % round_no,
                tenant_id="interactive",
                tenant_weight=5.0,
                token_budget=10.0,
                enqueued_at=float(round_no),
                deadline=10 ** 9,
            )
            plan = balancer.plan_batch([long_req, short], now=float(round_no))
            if "long" in scheduled_ids(plan):
                scheduled_at = round_no
                break
        self.assertIsNotNone(
            scheduled_at, "long request starved under continuous short load"
        )

    def test_short_request_gets_bounded_boost_over_long(self):
        balancer = QueueBalancer(vram_capacity_mb=10000.0, max_batch_size=1)
        long_req = make_request("long", token_budget=100000.0, deadline=10 ** 9)
        short_req = make_request("short", token_budget=10.0, deadline=10 ** 9)
        plan = balancer.plan_batch([long_req, short_req], now=0.0)
        self.assertEqual(scheduled_ids(plan), ["short"])


class TestDeadlineConflicts(unittest.TestCase):
    def test_earliest_deadline_wins_when_only_one_fits(self):
        balancer = QueueBalancer(vram_capacity_mb=100.0)
        tight = make_request("tight", vram_mb=80.0, deadline=10.0)
        loose = make_request("loose", vram_mb=80.0, deadline=500.0)
        plan = balancer.plan_batch([loose, tight], now=0.0)
        self.assertEqual(scheduled_ids(plan), ["tight"])
        self.assertEqual(deferred_reasons(plan, "loose"), [REASON_NO_VRAM_HEADROOM])

    def test_expired_request_dropped_by_default(self):
        balancer = QueueBalancer(vram_capacity_mb=100.0)
        expired = make_request("expired", deadline=5.0)
        plan = balancer.plan_batch([expired], now=10.0)
        self.assertEqual(plan.ordered, [])
        self.assertEqual(
            deferred_reasons(plan, "expired"), [REASON_DEADLINE_MISSED]
        )

    def test_expired_request_best_effort_policy(self):
        balancer = QueueBalancer(vram_capacity_mb=100.0, expired_policy="best_effort")
        expired = make_request("expired", deadline=5.0)
        plan = balancer.plan_batch([expired], now=10.0)
        self.assertEqual(scheduled_ids(plan), ["expired"])

    def test_tighter_deadline_outranks_higher_weight_when_slack_small(self):
        balancer = QueueBalancer(
            vram_capacity_mb=10000.0, max_batch_size=1,
            deadline_urgency_weight=50.0,
        )
        vip = make_request("vip", tenant_weight=3.0, deadline=900.0)
        urgent = make_request("urgent", tenant_weight=1.0, deadline=0.5)
        plan = balancer.plan_batch([vip, urgent], now=0.0)
        self.assertEqual(scheduled_ids(plan), ["urgent"])


class TestFairnessMetrics(unittest.TestCase):
    def test_jain_index_bounds(self):
        self.assertAlmostEqual(jain_fairness_index([1.0, 1.0, 1.0]), 1.0)
        self.assertAlmostEqual(jain_fairness_index([1.0, 0.0, 0.0]), 1.0 / 3.0)
        self.assertEqual(jain_fairness_index([]), 1.0)

    def test_weight_normalized_fairness(self):
        served = {"a": 200.0, "b": 100.0}
        weights = {"a": 2.0, "b": 1.0}
        self.assertAlmostEqual(weight_normalized_fairness(served, weights), 1.0)


class TestValidation(unittest.TestCase):
    def test_invalid_request_rejected(self):
        with self.assertRaises(ValueError):
            make_request("bad", tenant_weight=0.0)
        with self.assertRaises(ValueError):
            make_request("bad", vram_mb=-1.0)

    def test_invalid_balancer_config_rejected(self):
        with self.assertRaises(ValueError):
            QueueBalancer(vram_capacity_mb=0.0)
        with self.assertRaises(ValueError):
            QueueBalancer(vram_capacity_mb=1.0, expired_policy="yolo")

    def test_empty_queue_yields_empty_plan(self):
        balancer = QueueBalancer(vram_capacity_mb=100.0)
        plan = balancer.plan_batch([], now=0.0)
        self.assertIsInstance(plan, BatchPlan)
        self.assertEqual(plan.ordered, [])
        self.assertEqual(plan.deferred, [])


if __name__ == "__main__":
    unittest.main()
