"""
Pluggable scheduler policies for the orchestrator.

To add a new policy:
  1. Subclass SchedulerPolicy
  2. Implement decide(worker_id, cluster) -> InstructionResponse
  3. Pass an instance to OrchestratorService (or set via --policy CLI flag)
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict

from protocol import cluster_pb2


# Cluster state — shared, thread-safe snapshot of all connected workers

@dataclass
class WorkerState:
    worker_id:         str
    pending_data_size: float  # bytes
    is_migrating:      bool
    last_seen:         float = field(default_factory=time.time)


class ClusterState:
    """Thread-safe registry of all worker states."""

    def __init__(self):
        self._lock    = threading.Lock()
        self._workers: Dict[str, WorkerState] = {}

    def update(self, worker_id: str, pending_data_size: float, is_migrating: bool):
        with self._lock:
            self._workers[worker_id] = WorkerState(
                worker_id=worker_id,
                pending_data_size=pending_data_size,
                is_migrating=is_migrating,
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
            rate_limit_bps=0.0,  # 0 = unlimited
        )

class FixedRatePolicy(SchedulerPolicy):
    """
    Each worker flushes at the same fixed rate, regardless of cluster state.
    """
    def __init__(self, rate_bps: float = 500 * 1024 * 1024):  # 500 MB/s default
        self.rate = rate_bps

    def decide(self, worker_id, workers):
        return cluster_pb2.InstructionResponse(
            action=cluster_pb2.InstructionResponse.START_FLUSH,
            rate_limit_bps=self.rate,
        )

# Registry — map policy names to constructors (used by CLI / config)

POLICIES: Dict[str, type] = {
    "no-limit": NoLimitPolicy,
    "fixed-rate": FixedRatePolicy,
}

DEFAULT_POLICY = "no-limit"

ORCHESTRATOR_PORT = 50052
