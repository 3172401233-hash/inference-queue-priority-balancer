# inference-queue-priority-balancer

在截止时间、租户公平和显存约束下平衡推理队列。兼容 Python 3.8.2，仅依赖标准库。

## 使用

```python
from queue_balancer import QueueBalancer, Request

balancer = QueueBalancer(vram_capacity_mb=80_000, max_batch_size=8)

requests = [
    Request(
        request_id="req-1",
        tenant_id="tenant-a",
        tenant_weight=2.0,      # 租户优先级权重
        token_budget=512.0,     # 预估 token 消耗
        vram_mb=4096.0,         # 执行期显存占用 (MB)
        deadline=now + 0.005,   # 毫秒级短请求 deadline
        enqueued_at=now,
    ),
    # ...
]

plan = balancer.plan_batch(requests, now=now)
for item in plan.ordered:      # 下一批执行顺序
    run(item.request)
for req, reason in plan.deferred:
    log_deferral(req, reason)
```

`QueueBalancer` 在多次 `plan_batch` 调用之间保留租户服务量记忆（指数衰减），
因此应作为长生命周期对象持有，而不是每轮新建。

## 打分模型

每个请求的得分（高者优先）：

```
aging    = 1 + aging_rate * wait_time
urgency  = deadline_urgency_weight / max(deadline - now, slack_floor)
short    = short_job_boost / (1 + token_budget / token_scale)
fairness = 1 / (1 + tenant_service[tenant] / tenant_weight)

score    = tenant_weight * aging * (1 + urgency + short) * fairness
```

- `aging` 随排队时间无界增长：任何请求最终都会超过新到请求，
  **长请求不会被短请求永久堵塞，低权重租户也不会被无限抢占**。
- `short` 有界（上限 `short_job_boost`）：短请求只获得有限加权，
  长请求靠 aging 必然反超。
- `fairness` 随租户近期消耗的 token 量衰减：高优先级租户突发提交时，
  其边际优先级会降到其他租户水平，服务量按权重归一后趋于均衡。

## 公平性指标

- **Jain 公平指数** `jain_fairness_index(throughputs)`：各租户吞吐量
  的 Jain 指数，1.0 为完全公平，1/n 为最不平等。
- **权重归一公平指数** `weight_normalized_fairness(served, weights)`：
  对 `served/weight` 比值求 Jain 指数，衡量是否按权重比例分配服务。
- **饥饿自由性**：aging 无界增长保证任意请求的最坏等待时间有上界
  （约 `score_gap / aging_rate`），测试
  `test_aging_guarantees_long_request_eventually_runs` 验证了该性质。
- **建议线上监控**：各租户 P50/P99 排队时长、deadline 错过率、
  上述两个 Jain 指数随时间的变化。

## 降级规则

按以下顺序生效，被降级的请求进入 `BatchPlan.deferred` 并附带原因：

| 原因 | 触发条件 | 行为 |
| --- | --- | --- |
| `vram_exceeds_capacity` | 单请求显存 > 总容量 | 永久拒绝（重试无意义，应拆批或换实例） |
| `deadline_missed` | `now > deadline` 且 `expired_policy="drop"`（默认） | 丢弃；改为 `"best_effort"` 则降级为普通请求继续排队 |
| `no_vram_headroom` | 本批剩余显存放不下 | 本轮跳过（不阻塞后续更小的请求），下轮重试 |
| `batch_full` | 达到 `max_batch_size` | 本轮跳过，下轮重试 |

补充说明：

- 批内按得分降序贪心填充，显存放不下大请求时允许跳过它继续装小请求，
  但 aging 保证大请求不会被无限跳过。
- deadline 冲突（显存只够一个）时，slack 更小的请求因 `urgency` 项胜出。
- 排序对同分请求按入队时间、请求 ID 决胜，结果确定性可复现。

## 测试

```bash
python -m unittest
```

覆盖：显存不足（超容量拒绝 / 批内容量上限 / 批大小上限）、
租户突发（高权重租户刷量时低权重租户的有界等待与权重公平）、
长请求饥饿（持续短请求流下 aging 兜底）、
deadline 冲突（最早截止优先、过期丢弃 / best-effort、紧急度压过权重）。
