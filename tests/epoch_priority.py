#!/usr/bin/env python3
"""
Unit-style tests for orchestrator.scheduler.EpochPriorityPolicy.

Run from project root:
    python -m tests.epoch_priority
"""

from protocol import cluster_pb2
from orchestrator.scheduler import (
    ClusterState,
    EpochPriorityPolicy,
    WorkerState,
)


HOLD        = cluster_pb2.InstructionResponse.HOLD
START_FLUSH = cluster_pb2.InstructionResponse.START_FLUSH


def make_workers(*specs):
    """specs: iterable of (worker_id, checkpoint_size, epoch, total_epochs)."""
    out = {}
    for wid, size, epoch, total in specs:
        out[wid] = WorkerState(
            worker_id=wid,
            checkpoint_size=size,
            is_migrating=False,
            epoch=epoch,
            total_epochs=total,
        )
    return out


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(b))


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not cond:
        check.failures += 1
check.failures = 0


# ---------------------------------------------------------------------------
# 1. Inactive worker -> HOLD
# ---------------------------------------------------------------------------
def test_inactive_worker_holds():
    print("test_inactive_worker_holds")
    policy = EpochPriorityPolicy(pfs_bandwidth_bps=1e8)
    workers = make_workers(("a", 0, 5, 10))

    instr = policy.decide("a", workers)
    check("action == HOLD",      instr.action == HOLD)
    check("rate_limit_bps == 0", instr.rate_limit_bps == 0.0)


# ---------------------------------------------------------------------------
# 2. Unknown worker -> HOLD
# ---------------------------------------------------------------------------
def test_unknown_worker_holds():
    print("test_unknown_worker_holds")
    policy = EpochPriorityPolicy(pfs_bandwidth_bps=1e8)
    workers = make_workers(("a", 1024, 1, 10))

    instr = policy.decide("ghost", workers)
    check("unknown id -> HOLD", instr.action == HOLD)


# ---------------------------------------------------------------------------
# 3. Single active worker takes the whole bandwidth
# ---------------------------------------------------------------------------
def test_single_active_worker_full_bw():
    print("test_single_active_worker_full_bw")
    pfs_bw = 1e8
    policy = EpochPriorityPolicy(pfs_bandwidth_bps=pfs_bw, floor=0.2)
    workers = make_workers(("a", 1024, 0, 10), ("b", 0, 9, 10))

    instr_a = policy.decide("a", workers)
    instr_b = policy.decide("b", workers)

    check("a -> START_FLUSH",   instr_a.action == START_FLUSH)
    check("a gets full pfs_bw", approx(instr_a.rate_limit_bps, pfs_bw),
          f"got {instr_a.rate_limit_bps}")
    check("b -> HOLD",          instr_b.action == HOLD)


# ---------------------------------------------------------------------------
# 4. Linear-with-floor formula
# ---------------------------------------------------------------------------
# At progress=0: priority = floor
# At progress=1: priority = 1.0
# floor=0.2:
#   a: epoch=0/10  -> progress=0    -> priority=0.2
#   b: epoch=10/10 -> progress=1    -> priority=1.0
# total = 1.2 -> rate_a = pfs_bw * 0.2/1.2; rate_b = pfs_bw * 1.0/1.2
def test_linear_with_floor_extremes():
    print("test_linear_with_floor_extremes")
    pfs_bw = 1.2e8  # 120 MB/s; chosen so rates come out clean (20 / 100 MB/s)
    policy = EpochPriorityPolicy(pfs_bandwidth_bps=pfs_bw, floor=0.2)
    workers = make_workers(("a", 1024, 0, 10), ("b", 1024, 10, 10))

    rate_a = policy.decide("a", workers).rate_limit_bps
    rate_b = policy.decide("b", workers).rate_limit_bps

    check("rates sum to pfs_bw",     approx(rate_a + rate_b, pfs_bw))
    check("a = pfs_bw * 0.2 / 1.2",  approx(rate_a, pfs_bw * 0.2 / 1.2),
          f"got {rate_a}")
    check("b = pfs_bw * 1.0 / 1.2",  approx(rate_b, pfs_bw * 1.0 / 1.2),
          f"got {rate_b}")
    check("ratio b/a = 5",           approx(rate_b / rate_a, 5.0),
          f"ratio={rate_b / rate_a}")


# ---------------------------------------------------------------------------
# 5. Mid-progress workers
# ---------------------------------------------------------------------------
# floor=0.2:
#   a: 5/10  -> progress=0.5  -> priority=0.2 + 0.8*0.5 = 0.6
#   b: 8/10  -> progress=0.8  -> priority=0.2 + 0.8*0.8 = 0.84
def test_mid_progress():
    print("test_mid_progress")
    pfs_bw = 1e8
    policy = EpochPriorityPolicy(pfs_bandwidth_bps=pfs_bw, floor=0.2)
    workers = make_workers(("a", 1024, 5, 10), ("b", 1024, 8, 10))

    rate_a = policy.decide("a", workers).rate_limit_bps
    rate_b = policy.decide("b", workers).rate_limit_bps

    p_a, p_b = 0.6, 0.84
    total = p_a + p_b
    check("rates sum to pfs_bw", approx(rate_a + rate_b, pfs_bw))
    check("a = pfs_bw * 0.6/1.44", approx(rate_a, pfs_bw * p_a / total),
          f"got {rate_a}")
    check("b = pfs_bw * 0.84/1.44", approx(rate_b, pfs_bw * p_b / total),
          f"got {rate_b}")


# ---------------------------------------------------------------------------
# 6. floor=0 -> jobs at progress=0 starve except for EPS-style behaviour
#    (here we just verify formula: priority(0)=0, priority(1)=1).
# ---------------------------------------------------------------------------
def test_floor_zero():
    print("test_floor_zero")
    pfs_bw = 1e8
    policy = EpochPriorityPolicy(pfs_bandwidth_bps=pfs_bw, floor=0.0)
    workers = make_workers(("a", 1024, 0, 10), ("b", 1024, 10, 10))

    rate_a = policy.decide("a", workers).rate_limit_bps
    rate_b = policy.decide("b", workers).rate_limit_bps

    check("a rate == 0",         approx(rate_a, 0.0), f"got {rate_a}")
    check("b takes everything",  approx(rate_b, pfs_bw), f"got {rate_b}")


# ---------------------------------------------------------------------------
# 7. floor=1 -> uniform split (priority always 1.0 regardless of progress)
# ---------------------------------------------------------------------------
def test_floor_one_uniform():
    print("test_floor_one_uniform")
    pfs_bw = 1e8
    policy = EpochPriorityPolicy(pfs_bandwidth_bps=pfs_bw, floor=1.0)
    workers = make_workers(("a", 1024, 0, 10), ("b", 1024, 9, 10))

    rate_a = policy.decide("a", workers).rate_limit_bps
    rate_b = policy.decide("b", workers).rate_limit_bps

    check("rates equal",         approx(rate_a, rate_b))
    check("each = pfs_bw / 2",   approx(rate_a, pfs_bw / 2))


# ---------------------------------------------------------------------------
# 8. total_epochs == 0 -> progress treated as 0 -> priority = floor
# ---------------------------------------------------------------------------
def test_total_epochs_zero():
    print("test_total_epochs_zero")
    pfs_bw = 1.2e8
    policy = EpochPriorityPolicy(pfs_bandwidth_bps=pfs_bw, floor=0.2)
    # a has no progress info, b is at the end of training
    workers = make_workers(("a", 1024, 0, 0), ("b", 1024, 10, 10))

    rate_a = policy.decide("a", workers).rate_limit_bps
    rate_b = policy.decide("b", workers).rate_limit_bps

    check("rates sum to pfs_bw",        approx(rate_a + rate_b, pfs_bw))
    check("a = pfs_bw * 0.2/1.2 (floor)", approx(rate_a, pfs_bw * 0.2 / 1.2),
          f"got {rate_a}")
    check("b = pfs_bw * 1.0/1.2",          approx(rate_b, pfs_bw * 1.0 / 1.2),
          f"got {rate_b}")


# ---------------------------------------------------------------------------
# 9. epoch > total_epochs -> progress clamped to 1.0
# ---------------------------------------------------------------------------
def test_epoch_overflow_clamped():
    print("test_epoch_overflow_clamped")
    pfs_bw = 1e8
    policy = EpochPriorityPolicy(pfs_bandwidth_bps=pfs_bw, floor=0.2)
    # a: progress clamped to 1.0; b: also at the end. Should split evenly.
    workers = make_workers(("a", 1024, 999, 10), ("b", 1024, 10, 10))

    rate_a = policy.decide("a", workers).rate_limit_bps
    rate_b = policy.decide("b", workers).rate_limit_bps

    check("rates equal",       approx(rate_a, rate_b))
    check("each = pfs_bw / 2", approx(rate_a, pfs_bw / 2))


# ---------------------------------------------------------------------------
# 10. Idle worker excluded from denominator
# ---------------------------------------------------------------------------
def test_idle_worker_excluded():
    print("test_idle_worker_excluded")
    pfs_bw = 1.2e8
    policy = EpochPriorityPolicy(pfs_bandwidth_bps=pfs_bw, floor=0.2)
    # c is idle but at end of training -- should be ignored
    workers = make_workers(
        ("a", 1024, 0,  10),
        ("b", 1024, 10, 10),
        ("c", 0,    10, 10),
    )

    rate_a  = policy.decide("a", workers).rate_limit_bps
    rate_b  = policy.decide("b", workers).rate_limit_bps
    instr_c = policy.decide("c", workers)

    check("rates a+b sum to pfs_bw", approx(rate_a + rate_b, pfs_bw))
    check("a = pfs_bw * 0.2/1.2",     approx(rate_a, pfs_bw * 0.2 / 1.2))
    check("b = pfs_bw * 1.0/1.2",     approx(rate_b, pfs_bw * 1.0 / 1.2))
    check("c -> HOLD",                 instr_c.action == HOLD)


# ---------------------------------------------------------------------------
# 11. ClusterState propagates epoch / total_epochs
# ---------------------------------------------------------------------------
def test_cluster_state_propagates_epoch():
    print("test_cluster_state_propagates_epoch")
    cluster = ClusterState()

    cluster.update("a", checkpoint_size=1024, is_migrating=False,
                   epoch=3, total_epochs=10)
    snap = cluster.snapshot()
    check("epoch propagated",        snap["a"].epoch == 3)
    check("total_epochs propagated", snap["a"].total_epochs == 10)

    # Defaults still work for older callers (positional + minimal kwargs)
    cluster.update("b", checkpoint_size=512, is_migrating=False)
    snap = cluster.snapshot()
    check("default epoch == 0",        snap["b"].epoch == 0)
    check("default total_epochs == 0", snap["b"].total_epochs == 0)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_inactive_worker_holds,
        test_unknown_worker_holds,
        test_single_active_worker_full_bw,
        test_linear_with_floor_extremes,
        test_mid_progress,
        test_floor_zero,
        test_floor_one_uniform,
        test_total_epochs_zero,
        test_epoch_overflow_clamped,
        test_idle_worker_excluded,
        test_cluster_state_propagates_epoch,
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
