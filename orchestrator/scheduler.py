"""
Pluggable scheduler policies for the orchestrator.

To add a new policy:
  1. Subclass SchedulerPolicy
  2. Implement decide(worker_id, cluster) -> InstructionResponse
  3. Pass an instance to OrchestratorService (or set via --policy CLI flag)
"""

from __future__ import annotations

import time
import threading

from typing import Dict
from protocol import cluster_pb2
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# Cluster state — shared, thread-safe snapshot of all connected workers

@dataclass
class WorkerState:
    worker_id:         str
    checkpoint_size:   float  # bytes
    is_migrating:      bool
    last_seen:         float        = field(default_factory=time.time)
    # Wall-clock time at which checkpoint_size transitioned from 0 to >0.
    # None when the worker is idle. Used by age-aware policies.
    pending_since:     float | None = None
    # Training progress reported by the migrater. 0/0 when unknown.
    epoch:             int          = 0
    total_epochs:      int          = 0


class ClusterState:
    """Thread-safe registry of all worker states."""

    def __init__(self):
        self._lock    = threading.Lock()
        self._workers: Dict[str, WorkerState] = {}

    def update(
        self,
        worker_id: str,
        checkpoint_size: float,
        is_migrating: bool,
        epoch: int = 0,
        total_epochs: int = 0,
    ):
        now  = time.time()
        with self._lock:
            prev = self._workers.get(worker_id)

            if checkpoint_size > 0:
                pending_since = (
                    prev.pending_since
                    if prev is not None and prev.pending_since is not None
                    else now
                )
            else:
                pending_since = None

            self._workers[worker_id] = WorkerState(
                worker_id=worker_id,
                checkpoint_size=checkpoint_size,
                is_migrating=is_migrating,
                last_seen=now,
                pending_since=pending_since,
                epoch=epoch,
                total_epochs=total_epochs,
            )

    def remove(self, worker_id: str):
        with self._lock:
            self._workers.pop(worker_id, None)

    def snapshot(self) -> Dict[str, WorkerState]:
        """Return an immutable copy so policies don't need to hold the lock."""
        with self._lock:
            return dict(self._workers)

    @property
    def worker_count(self) -> int:
        with self._lock:
            return len(self._workers)


# Scheduler policies

class SchedulerPolicy(ABC):
    """
    Strategy Pattern: encapsulates the scheduling logic and decision-making.
    Base class for all scheduling policies.
    """

    @abstractmethod
    def decide(
        self,
        worker_id: str,
        workers: Dict[str, WorkerState],
    ) -> cluster_pb2.InstructionResponse:
        ...


class NoLimitPolicy(SchedulerPolicy):
    """
    Baseline policy: always allow flush at full speed.
    Equivalent to writing directly to Lustre — used for benchmarking.
    """
    def decide(self, worker_id, workers):
        return cluster_pb2.InstructionResponse(
            action=cluster_pb2.InstructionResponse.START_FLUSH,
            rate_limit_bps= 1 * 1024 * 1024 * 1024,  # 1 GB/s 
        )

class FixedRatePolicy(SchedulerPolicy):
    """
    Each worker flushes at the same fixed rate, regardless of cluster state.
    """
    def __init__(self, rate_bps: float = 1 * 1024 * 1024 * 1024):  # 1 GB/s default
        self.rate = rate_bps

    def decide(self, worker_id, workers):
        return cluster_pb2.InstructionResponse(
            action=cluster_pb2.InstructionResponse.START_FLUSH,
            rate_limit_bps=self.rate,
        )


class UniformFairSharePolicy(SchedulerPolicy):
    """
    Divides total PFS bandwidth equally among all registered workers.
    Every worker gets the same share regardless of whether it has pending data.
    """

    def __init__(self, pfs_bandwidth_bps: float = 1e9):  # 1 GB/s default
        self.pfs_bandwidth_bps = pfs_bandwidth_bps

    def decide(self, worker_id, workers):
        rate = self.pfs_bandwidth_bps / len(workers) if workers else 0.0

        return cluster_pb2.InstructionResponse(
            action=cluster_pb2.InstructionResponse.START_FLUSH,
            rate_limit_bps=rate,
        )


class ActiveFairSharePolicy(SchedulerPolicy):
    """
    Divides total PFS bandwidth equally among workers with pending data only.
    Workers without pending data are told to HOLD, and their share is
    redistributed to the active ones.
    """

    def __init__(self, pfs_bandwidth_bps: float = 1e9):  # 1 GB/s default
        self.pfs_bandwidth_bps = pfs_bandwidth_bps

    def decide(self, worker_id, workers):
        worker = workers.get(worker_id)
        if worker is None or worker.checkpoint_size <= 0:
            return cluster_pb2.InstructionResponse(
                action=cluster_pb2.InstructionResponse.HOLD,
                rate_limit_bps=0.0,
            )

        active = sum(1 for w in workers.values() if w.checkpoint_size > 0)
        rate = self.pfs_bandwidth_bps / active

        return cluster_pb2.InstructionResponse(
            action=cluster_pb2.InstructionResponse.START_FLUSH,
            rate_limit_bps=rate,
        )


class StaticPriorityPolicy(SchedulerPolicy):
    """
    Weighted fair-share among active workers using fixed per-worker priorities.

    Each active worker (checkpoint_size > 0) receives a share of the PFS
    bandwidth proportional to its priority weight:

        rate_i = pfs_bw * priority_i / sum(priority_j for j in active)

    Workers without pending data are told to HOLD. Workers not present in
    the priority map fall back to default_priority. Non-positive weights are
    coerced to default_priority (defensive).
    """

    def __init__(
        self,
        pfs_bandwidth_bps: float = 1e9,
        priorities: Dict[str, float] | None = None,
        default_priority: float = 1.0,
    ):
        assert default_priority > 0, "default_priority must be positive"
        self.pfs_bandwidth_bps = pfs_bandwidth_bps
        self.priorities        = dict(priorities) if priorities else {}
        self.default_priority  = default_priority

    def _weight(self, worker_id: str) -> float:
        w = self.priorities.get(worker_id, self.default_priority)
        return w if w > 0 else self.default_priority

    def decide(self, worker_id, workers):
        worker = workers.get(worker_id)
        if worker is None or worker.checkpoint_size <= 0:
            return cluster_pb2.InstructionResponse(
                action=cluster_pb2.InstructionResponse.HOLD,
                rate_limit_bps=0.0,
            )

        total = sum(
            self._weight(w.worker_id)
            for w in workers.values()
            if w.checkpoint_size > 0
        )

        rate = self.pfs_bandwidth_bps * self._weight(worker_id) / total

        return cluster_pb2.InstructionResponse(
            action=cluster_pb2.InstructionResponse.START_FLUSH,
            rate_limit_bps=rate,
        )


class AgePriorityPolicy(SchedulerPolicy):
    """
    Dynamic priority based on pending checkpoint size and pending age.

    For each active worker (checkpoint_size > 0) the orchestrator computes:

        size_norm_i = checkpoint_size_i / max_size
        age_i       = now - pending_since_i
        age_norm_i  = age_i / max_age           (0 if max_age == 0)
        priority_i  = alpha * size_norm_i + beta * age_norm_i
        priority_i  = max(priority_i, EPS)      (avoid degenerate zero)

        rate_i      = pfs_bw * priority_i / sum(priority_j for j in active)

    Workers without pending data receive HOLD. Larger pending checkpoints get
    more bandwidth (drain faster); older pending checkpoints get more
    bandwidth (anti-starvation).
    """

    EPS = 1e-6

    def __init__(
        self,
        pfs_bandwidth_bps: float = 1e9,
        alpha: float = 0.5,
        beta: float = 0.5,
    ):
        assert alpha >= 0 and beta >= 0, "alpha and beta must be non-negative"
        assert alpha + beta > 0,         "alpha + beta must be positive"
        self.pfs_bandwidth_bps = pfs_bandwidth_bps
        self.alpha = alpha
        self.beta  = beta

    def _priority(self, w: WorkerState, now: float, max_size: float, max_age: float) -> float:
        size_norm = w.checkpoint_size / max_size if max_size > 0 else 0.0
        age       = (now - w.pending_since) if w.pending_since is not None else 0.0
        age_norm  = age / max_age if max_age > 0 else 0.0
        priority  = self.alpha * size_norm + self.beta * age_norm
        return max(priority, self.EPS)

    def decide(self, worker_id, workers):
        worker = workers.get(worker_id)
        if worker is None or worker.checkpoint_size <= 0:
            return cluster_pb2.InstructionResponse(
                action=cluster_pb2.InstructionResponse.HOLD,
                rate_limit_bps=0.0,
            )

        now    = time.time()
        active = [w for w in workers.values() if w.checkpoint_size > 0]

        max_size = max(w.checkpoint_size for w in active)
        max_age  = max(
            (now - w.pending_since) if w.pending_since is not None else 0.0
            for w in active
        )

        total_priority = sum(self._priority(w, now, max_size, max_age) for w in active)
        my_priority    = self._priority(worker, now, max_size, max_age)

        rate = self.pfs_bandwidth_bps * my_priority / total_priority

        return cluster_pb2.InstructionResponse(
            action=cluster_pb2.InstructionResponse.START_FLUSH,
            rate_limit_bps=rate,
        )


class EpochPriorityPolicy(SchedulerPolicy):
    """
    Dynamic priority based on training progress (epoch / total_epochs).

    Jobs closer to the end of training have invested more compute and have
    fewer remaining checkpoints to protect their state, so they receive a
    proportionally larger share of the PFS bandwidth.

    Linear-with-floor weight:

        progress_i  = clamp(epoch_i / total_epochs_i, 0.0, 1.0)
        priority_i  = floor + (1 - floor) * progress_i

        rate_i      = pfs_bw * priority_i / sum(priority_j for j in active)

    Workers with total_epochs == 0 (training side did not report progress)
    are treated as progress = 0 -> priority = floor. Workers without
    pending data receive HOLD.
    """

    def __init__(
        self,
        pfs_bandwidth_bps: float = 1e9,
        floor: float = 0.2,
    ):
        assert 0.0 <= floor <= 1.0, "floor must be in [0, 1]"
        self.pfs_bandwidth_bps = pfs_bandwidth_bps
        self.floor             = floor

    def _priority(self, w: WorkerState) -> float:
        if w.total_epochs <= 0:
            progress = 0.0
        else:
            progress = w.epoch / w.total_epochs
            progress = max(0.0, min(progress, 1.0))
        return self.floor + (1.0 - self.floor) * progress

    def decide(self, worker_id, workers):
        worker = workers.get(worker_id)
        if worker is None or worker.checkpoint_size <= 0:
            return cluster_pb2.InstructionResponse(
                action=cluster_pb2.InstructionResponse.HOLD,
                rate_limit_bps=0.0,
            )

        active = [w for w in workers.values() if w.checkpoint_size > 0]

        total_priority = sum(self._priority(w) for w in active)
        my_priority    = self._priority(worker)

        rate = self.pfs_bandwidth_bps * my_priority / total_priority

        return cluster_pb2.InstructionResponse(
            action=cluster_pb2.InstructionResponse.START_FLUSH,
            rate_limit_bps=rate,
        )


# Registry — map policy names to constructors (used by CLI / config)

POLICIES: Dict[str, type] = {
    "no-limit": NoLimitPolicy,
    "fixed-rate": FixedRatePolicy,
    "uniform-fair-share": UniformFairSharePolicy,
    "active-fair-share": ActiveFairSharePolicy,
    "static-priority": StaticPriorityPolicy,
    "age-priority": AgePriorityPolicy,
    "epoch-priority": EpochPriorityPolicy,
}

DEFAULT_POLICY = "no-limit"

ORCHESTRATOR_PORT = 50052
