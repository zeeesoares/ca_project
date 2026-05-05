#!/usr/bin/env python3
"""
Unit-style tests for orchestrator.scheduler.AgePriorityPolicy.

Run from project root:
    python -m tests.age_priority
"""

import time

from protocol import cluster_pb2
from orchestrator.scheduler import (
    AgePriorityPolicy,
    ClusterState,
    WorkerState,
)


HOLD        = cluster_pb2.InstructionResponse.HOLD
START_FLUSH = cluster_pb2.InstructionResponse.START_FLUSH


def make_workers(*specs):
    """specs: iterable of (worker_id, checkpoint_size, pending_age_seconds).

    pending_age_seconds = None means worker is idle (pending_since=None);
    otherwise pending_since is set so that (now - pending_since) == age.
    """
    now = time.time()
    out = {}
    for spec in specs:
        wid, size, age = spec
        pending_since = None if age is None else now - age
        out[wid] = WorkerState(
            worker_id=wid,
            checkpoint_size=size,
            is_migrating=False,
            pending_since=pending_since,
        )
    return out


def approx(a, b, tol=1e-3):
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
    policy = AgePriorityPolicy(pfs_bandwidth_bps=1e8)
    workers = make_workers(("a", 0, None))

    instr = policy.decide("a", workers)
    check("action == HOLD",      instr.action == HOLD)
    check("rate_limit_bps == 0", instr.rate_limit_bps == 0.0)


# ---------------------------------------------------------------------------
# 2. Unknown worker -> HOLD
# ---------------------------------------------------------------------------
def test_unknown_worker_holds():
    print("test_unknown_worker_holds")
    policy = AgePriorityPolicy(pfs_bandwidth_bps=1e8)
    workers = make_workers(("a", 1024, 1.0))

    instr = policy.decide("ghost", workers)
    check("unknown id -> HOLD", instr.action == HOLD)


# ---------------------------------------------------------------------------
# 3. Single active worker takes the whole bandwidth
# ---------------------------------------------------------------------------
def test_single_active_worker_full_bw():
    print("test_single_active_worker_full_bw")
    pfs_bw = 1e8
    policy = AgePriorityPolicy(pfs_bandwidth_bps=pfs_bw)
    workers = make_workers(("a", 1024, 0.5), ("b", 0, None))

    instr_a = policy.decide("a", workers)
    instr_b = policy.decide("b", workers)

    check("a -> START_FLUSH",   instr_a.action == START_FLUSH)
    check("a gets full pfs_bw", approx(instr_a.rate_limit_bps, pfs_bw),
          f"got {instr_a.rate_limit_bps}")
    check("b -> HOLD",          instr_b.action == HOLD)


# ---------------------------------------------------------------------------
# 4. alpha=1, beta=0 reduces to size-proportional split
# ---------------------------------------------------------------------------
def test_size_only_split():
    print("test_size_only_split")
    pfs_bw = 1e8
    policy = AgePriorityPolicy(pfs_bandwidth_bps=pfs_bw, alpha=1.0, beta=0.0)
    # Equal ages, sizes 3:1 -> rates 3:1
    workers = make_workers(("a", 3000, 1.0), ("b", 1000, 1.0))

    rate_a = policy.decide("a", workers).rate_limit_bps
    rate_b = policy.decide("b", workers).rate_limit_bps

    check("rates sum to pfs_bw", approx(rate_a + rate_b, pfs_bw),
          f"sum={rate_a + rate_b}")
    check("a:b == 3:1",          approx(rate_a / rate_b, 3.0),
          f"ratio={rate_a / rate_b}")


# ---------------------------------------------------------------------------
# 5. alpha=0, beta=1 reduces to age-proportional split
# ---------------------------------------------------------------------------
def test_age_only_split():
    print("test_age_only_split")
    pfs_bw = 1e8
    policy = AgePriorityPolicy(pfs_bandwidth_bps=pfs_bw, alpha=0.0, beta=1.0)
    # Equal sizes, ages 4:1 -> rates 4:1
    workers = make_workers(("a", 1000, 4.0), ("b", 1000, 1.0))

    rate_a = policy.decide("a", workers).rate_limit_bps
    rate_b = policy.decide("b", workers).rate_limit_bps

    check("rates sum to pfs_bw", approx(rate_a + rate_b, pfs_bw),
          f"sum={rate_a + rate_b}")
    check("a:b == 4:1",          approx(rate_a / rate_b, 4.0),
          f"ratio={rate_a / rate_b}")


# ---------------------------------------------------------------------------
# 6. Idle worker excluded from denominator
# ---------------------------------------------------------------------------
def test_idle_worker_excluded():
    print("test_idle_worker_excluded")
    pfs_bw = 1e8
    policy = AgePriorityPolicy(pfs_bandwidth_bps=pfs_bw, alpha=1.0, beta=0.0)
    # c is idle; a:b sizes 2:1
    workers = make_workers(("a", 2000, 1.0), ("b", 1000, 1.0), ("c", 0, None))

    rate_a = policy.decide("a", workers).rate_limit_bps
    rate_b = policy.decide("b", workers).rate_limit_bps
    instr_c = policy.decide("c", workers)

    check("rates sum to pfs_bw", approx(rate_a + rate_b, pfs_bw))
    check("a:b == 2:1",          approx(rate_a / rate_b, 2.0))
    check("c -> HOLD",           instr_c.action == HOLD)


# ---------------------------------------------------------------------------
# 7. Mixed signals: large but young vs small but old (alpha=beta=0.5)
# ---------------------------------------------------------------------------
def test_mixed_signals():
    print("test_mixed_signals")
    pfs_bw = 1e8
    policy = AgePriorityPolicy(pfs_bandwidth_bps=pfs_bw, alpha=0.5, beta=0.5)
    # a: max size, min age. b: min size, max age.
    # max_size = 1000, max_age = 4.0
    # priority_a = 0.5 * (1000/1000) + 0.5 * (1/4) = 0.5 + 0.125 = 0.625
    # priority_b = 0.5 * (250/1000)  + 0.5 * (4/4) = 0.125 + 0.5 = 0.625
    # Designed so that priorities tie -> rates equal.
    workers = make_workers(("a", 1000, 1.0), ("b", 250, 4.0))

    rate_a = policy.decide("a", workers).rate_limit_bps
    rate_b = policy.decide("b", workers).rate_limit_bps

    check("rates sum to pfs_bw", approx(rate_a + rate_b, pfs_bw))
    check("rates equal",         approx(rate_a, rate_b),
          f"a={rate_a}, b={rate_b}")


# ---------------------------------------------------------------------------
# 8. ClusterState tracks pending_since across updates
# ---------------------------------------------------------------------------
def test_cluster_state_pending_since():
    print("test_cluster_state_pending_since")
    cluster = ClusterState()

    # initially idle
    cluster.update("a", checkpoint_size=0, is_migrating=False)
    snap = cluster.snapshot()
    check("idle worker has pending_since=None",
          snap["a"].pending_since is None)

    # transition to active -> pending_since set
    cluster.update("a", checkpoint_size=1000, is_migrating=False)
    snap = cluster.snapshot()
    first_pending_since = snap["a"].pending_since
    check("active worker has pending_since set",
          first_pending_since is not None)

    time.sleep(0.05)

    # still active -> pending_since must be preserved
    cluster.update("a", checkpoint_size=800, is_migrating=True)
    snap = cluster.snapshot()
    check("pending_since preserved while active",
          snap["a"].pending_since == first_pending_since,
          f"first={first_pending_since}, now={snap['a'].pending_since}")

    # back to idle -> cleared
    cluster.update("a", checkpoint_size=0, is_migrating=False)
    snap = cluster.snapshot()
    check("pending_since cleared on idle",
          snap["a"].pending_since is None)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_inactive_worker_holds,
        test_unknown_worker_holds,
        test_single_active_worker_full_bw,
        test_size_only_split,
        test_age_only_split,
        test_idle_worker_excluded,
        test_mixed_signals,
        test_cluster_state_pending_since,
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
