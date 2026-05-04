import argparse
import json
import sys
import grpc
from concurrent import futures

from protocol import cluster_pb2, cluster_pb2_grpc
from orchestrator.scheduler import (
    ClusterState,
    SchedulerPolicy,
    NoLimitPolicy,
    StaticPriorityPolicy,
    POLICIES,
    DEFAULT_POLICY,
    ORCHESTRATOR_PORT,
)


def load_priority_map(path: str) -> tuple[dict[str, float], float]:
    """Load a priority map JSON file.

    Expected schema:
        { "default": 1.0, "workers": { "<worker_id>": <weight>, ... } }

    Returns (priorities, default_priority). Fails fast on missing file,
    invalid JSON, or invalid schema — a misconfigured policy should not
    silently degrade to uniform fair share.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level JSON must be an object")

    default_priority = data.get("default", 1.0)
    if not isinstance(default_priority, (int, float)) or default_priority <= 0:
        raise ValueError(
            f"{path}: 'default' must be a positive number, got {default_priority!r}"
        )

    workers = data.get("workers", {})
    if not isinstance(workers, dict):
        raise ValueError(f"{path}: 'workers' must be an object")

    priorities: dict[str, float] = {}
    for worker_id, weight in workers.items():
        if not isinstance(weight, (int, float)) or weight <= 0:
            raise ValueError(
                f"{path}: priority for {worker_id!r} must be a positive number, "
                f"got {weight!r}"
            )
        priorities[worker_id] = float(weight)

    return priorities, float(default_priority)


# Logger Configuration
from orchestrator.utils import setup_log_metrics
metrics = setup_log_metrics("qos_metrics", "logs/orchestrator_metrics.json")


class OrchestratorService(cluster_pb2_grpc.OrchestratorServiceServicer):

    def __init__(self, policy: SchedulerPolicy, cluster: ClusterState):
        self.policy  = policy
        self.cluster = cluster

    def Monitor(self, request_iterator, context):
        worker_id = None
        try:
            for heartbeat in request_iterator:
                worker_id = heartbeat.worker_id

                self.cluster.update(
                    worker_id,
                    heartbeat.checkpoint_size,
                    heartbeat.is_migrating
                )

                workers = self.cluster.snapshot()
                instruction = self.policy.decide(worker_id, workers)

                action_name = cluster_pb2.InstructionResponse.Action.Name(
                    instruction.action
                )

                progress = list(heartbeat.progress)
                latest_progress = progress[-1] if progress else None

                log_entry = {
                    "worker_id": worker_id,
                    "checkpoint_size": heartbeat.checkpoint_size,
                    "is_migrating": heartbeat.is_migrating,
                    "action": action_name,
                }

                if latest_progress is not None:
                    log_entry.update(
                        {
                            "bytes_copied": latest_progress.bytes_copied,
                            "total_bytes": latest_progress.total_bytes,
                            "remaining_bytes": latest_progress.remaining_bytes,
                            "interval_throughput_bps": latest_progress.interval_throughput_bps,
                            "average_throughput_bps": latest_progress.average_throughput_bps,
                            "configured_rate_bps": latest_progress.configured_rate_bps,
                            "chunk_size_bytes": latest_progress.chunk_size_bytes,
                            "nr_progress_messages": len(progress),
                        }
                    )

                metrics.info(log_entry)

                yield instruction
        finally:
            if worker_id:
                self.cluster.remove(worker_id)


def serve(port=ORCHESTRATOR_PORT, policy: SchedulerPolicy = None):
    if policy is None:
        policy = NoLimitPolicy()

    cluster = ClusterState()
    server  = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    cluster_pb2_grpc.add_OrchestratorServiceServicer_to_server(
        OrchestratorService(policy, cluster),
        server,
    )

    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"Orchestrator running on {port} | policy={policy.__class__.__name__}")
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",   type=int,   default=ORCHESTRATOR_PORT)
    parser.add_argument("--policy", type=str,   default=DEFAULT_POLICY, choices=POLICIES.keys())
    parser.add_argument("--pfs-bw", type=float, default=1e9, help="PFS bandwidth in bps")
    parser.add_argument(
        "--priority-map",
        type=str,
        default=None,
        help="Path to JSON priority map for static-priority policy "
             "(schema: {\"default\": float, \"workers\": {worker_id: weight}})",
    )
    args = parser.parse_args()

    policy_cls = POLICIES[args.policy]

    if policy_cls is StaticPriorityPolicy:
        priorities, default_priority = (
            load_priority_map(args.priority_map)
            if args.priority_map
            else ({}, 1.0)
        )
        policy = StaticPriorityPolicy(
            pfs_bandwidth_bps=args.pfs_bw,
            priorities=priorities,
            default_priority=default_priority,
        )
    else:
        if args.priority_map:
            print(
                f"warning: --priority-map ignored for policy {args.policy!r}",
                file=sys.stderr,
            )
        try:
            policy = policy_cls(pfs_bandwidth_bps=args.pfs_bw)
        except TypeError:
            policy = policy_cls()

    try:
        serve(port=args.port, policy=policy)
    except KeyboardInterrupt:
        print("Orchestrator shutting down")
