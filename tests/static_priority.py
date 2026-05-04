#!/usr/bin/env python3
"""
Unit-style tests for orchestrator.scheduler.StaticPriorityPolicy.

Run from project root:
    python -m tests.static_priority
"""

from protocol import cluster_pb2
from orchestrator.scheduler import StaticPriorityPolicy, WorkerState


HOLD        = cluster_pb2.InstructionResponse.HOLD
START_FLUSH = cluster_pb2.InstructionResponse.START_FLUSH


def make_workers(*specs):
    """specs: iterable of (worker_id, checkpoint_size)."""
    return {
        wid: WorkerState(worker_id=wid, checkpoint_size=size, is_migrating=False)
        for wid, size in specs
    }


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(b))


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" --{detail}" if detail else ""))
    if not cond:
        check.failures += 1
check.failures = 0


# ---------------------------------------------------------------------------
# 1. Inactive worker -> HOLD
# ---------------------------------------------------------------------------
def test_inactive_worker_holds():
    print("test_inactive_worker_holds")
    policy = StaticPriorityPolicy(pfs_bandwidth_bps=1e8, priorities={"a": 1.0})
    workers = make_workers(("a", 0))

    instr = policy.decide("a", workers)
    check("action == HOLD",     instr.action == HOLD)
    check("rate_limit_bps == 0", instr.rate_limit_bps == 0.0)


# ---------------------------------------------------------------------------
# 2. Unknown worker (not in snapshot) -> HOLD
# ---------------------------------------------------------------------------
def test_unknown_worker_holds():
    print("test_unknown_worker_holds")
    policy = StaticPriorityPolicy(pfs_bandwidth_bps=1e8)
    workers = make_workers(("a", 1024))

    instr = policy.decide("ghost", workers)
    check("unknown id -> HOLD", instr.action == HOLD)


# ---------------------------------------------------------------------------
# 3. Single active worker takes the whole bandwidth
# ---------------------------------------------------------------------------
def test_single_active_worker_full_bw():
    print("test_single_active_worker_full_bw")
    pfs_bw = 1e8
    policy = StaticPriorityPolicy(
        pfs_bandwidth_bps=pfs_bw,
        priorities={"a": 5.0, "b": 99.0},  # b is idle, weight irrelevant
    )
    workers = make_workers(("a", 1024), ("b", 0))

    instr_a = policy.decide("a", workers)
    instr_b = policy.decide("b", workers)

    check("a -> START_FLUSH",   instr_a.action == START_FLUSH)
    check("a gets full pfs_bw", approx(instr_a.rate_limit_bps, pfs_bw),
          f"got {instr_a.rate_limit_bps}")
    check("b -> HOLD",           instr_b.action == HOLD)


# ---------------------------------------------------------------------------
# 4. Three active workers split bw in proportion 3:2:1
# ---------------------------------------------------------------------------
def test_three_workers_proportional_split():
    print("test_three_workers_proportional_split")
    pfs_bw = 1e8  # 100 MB/s
    policy = StaticPriorityPolicy(
        pfs_bandwidth_bps=pfs_bw,
        priorities={"node01": 3.0, "node02": 2.0, "node03": 1.0},
    )
    workers = make_workers(("node01", 4096), ("node02", 4096), ("node03", 4096))

    rate1 = policy.decide("node01", workers).rate_limit_bps
    rate2 = policy.decide("node02", workers).rate_limit_bps
    rate3 = policy.decide("node03", workers).rate_limit_bps

    total = rate1 + rate2 + rate3
    expected1 = pfs_bw * 3 / 6
    expected2 = pfs_bw * 2 / 6
    expected3 = pfs_bw * 1 / 6

    check("rates sum to pfs_bw", approx(total, pfs_bw), f"sum={total}")
    check("node01 = 50 MB/s",    approx(rate1, expected1), f"got {rate1}")
    check("node02 = 33.3 MB/s",  approx(rate2, expected2), f"got {rate2}")
    check("node03 = 16.7 MB/s",  approx(rate3, expected3), f"got {rate3}")


# ---------------------------------------------------------------------------
# 5. Idle worker is excluded from the denominator
# ---------------------------------------------------------------------------
def test_idle_worker_excluded():
    print("test_idle_worker_excluded")
    pfs_bw = 1e8
    policy = StaticPriorityPolicy(
        pfs_bandwidth_bps=pfs_bw,
        priorities={"a": 3.0, "b": 2.0, "c": 1.0},
    )
    # c is idle -> only a and b share bw, in 3:2 ratio
    workers = make_workers(("a", 1024), ("b", 1024), ("c", 0))

    rate_a = policy.decide("a", workers).rate_limit_bps
    rate_b = policy.decide("b", workers).rate_limit_bps
    instr_c = policy.decide("c", workers)

    check("a = 60 MB/s", approx(rate_a, pfs_bw * 3 / 5), f"got {rate_a}")
    check("b = 40 MB/s", approx(rate_b, pfs_bw * 2 / 5), f"got {rate_b}")
    check("c -> HOLD",   instr_c.action == HOLD)
    check("a + b = pfs_bw", approx(rate_a + rate_b, pfs_bw))


# ---------------------------------------------------------------------------
# 6. Worker not in the priority map falls back to default_priority
# ---------------------------------------------------------------------------
def test_default_priority_fallback():
    print("test_default_priority_fallback")
    pfs_bw = 1e8
    policy = StaticPriorityPolicy(
        pfs_bandwidth_bps=pfs_bw,
        priorities={"vip": 4.0},
        default_priority=1.0,
    )
    # rookie not in the map -> weight 1.0
    workers = make_workers(("vip", 1024), ("rookie", 1024))

    rate_vip    = policy.decide("vip", workers).rate_limit_bps
    rate_rookie = policy.decide("rookie", workers).rate_limit_bps

    check("vip = 80 MB/s",    approx(rate_vip,    pfs_bw * 4 / 5), f"got {rate_vip}")
    check("rookie = 20 MB/s", approx(rate_rookie, pfs_bw * 1 / 5), f"got {rate_rookie}")


# ---------------------------------------------------------------------------
# 7. Empty priority map -> behaves like ActiveFairSharePolicy (uniform on actives)
# ---------------------------------------------------------------------------
def test_empty_map_is_active_fair_share():
    print("test_empty_map_is_active_fair_share")
    pfs_bw = 1e8
    policy = StaticPriorityPolicy(pfs_bandwidth_bps=pfs_bw)  # empty map, default 1.0
    workers = make_workers(("a", 1024), ("b", 1024), ("c", 0))

    rate_a = policy.decide("a", workers).rate_limit_bps
    rate_b = policy.decide("b", workers).rate_limit_bps

    check("a = b (uniform)",  approx(rate_a, rate_b))
    check("a = pfs_bw / 2",   approx(rate_a, pfs_bw / 2), f"got {rate_a}")


# ---------------------------------------------------------------------------
# 8. Non-positive priority in the map is coerced to default_priority
# ---------------------------------------------------------------------------
def test_non_positive_weight_coerced():
    print("test_non_positive_weight_coerced")
    pfs_bw = 1e8
    policy = StaticPriorityPolicy(
        pfs_bandwidth_bps=pfs_bw,
        priorities={"a": 0.0, "b": -3.0, "c": 2.0},
        default_priority=1.0,
    )
    # effective weights: a=1, b=1, c=2  -> total=4
    workers = make_workers(("a", 1024), ("b", 1024), ("c", 1024))

    rate_a = policy.decide("a", workers).rate_limit_bps
    rate_b = policy.decide("b", workers).rate_limit_bps
    rate_c = policy.decide("c", workers).rate_limit_bps

    check("a = pfs_bw/4", approx(rate_a, pfs_bw / 4), f"got {rate_a}")
    check("b = pfs_bw/4", approx(rate_b, pfs_bw / 4), f"got {rate_b}")
    check("c = pfs_bw/2", approx(rate_c, pfs_bw / 2), f"got {rate_c}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_inactive_worker_holds,
        test_unknown_worker_holds,
        test_single_active_worker_full_bw,
        test_three_workers_proportional_split,
        test_idle_worker_excluded,
        test_default_priority_fallback,
        test_empty_map_is_active_fair_share,
        test_non_positive_weight_coerced,
    ]

    for t in tests:
        t()
        print()

    print("=" * 50)
    if check.failures == 0:
        print(f"All {len(tests)} test groups passed.")
    else:
        print(f"{check.failures} check(s) FAILED across {len(tests)} test groups.")
        raise SystemExit(1)
