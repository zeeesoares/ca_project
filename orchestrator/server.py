import argparse
import grpc
from concurrent import futures

from protocol import cluster_pb2, cluster_pb2_grpc
from orchestrator.scheduler import (
    ClusterState,
    SchedulerPolicy,
    NoLimitPolicy,
    POLICIES,
    DEFAULT_POLICY,
    ORCHESTRATOR_PORT,
)

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
                self.cluster.update(worker_id, heartbeat.pending_data_size, heartbeat.is_migrating)

                workers = self.cluster.snapshot()
                instruction = self.policy.decide(worker_id, workers)

                action_name = cluster_pb2.InstructionResponse.Action.Name(instruction.action)
               
                metrics.info({
                    "worker_id": worker_id,
                    "pending_data_size": heartbeat.pending_data_size,
                    "is_migrating": heartbeat.is_migrating,
                    "action": action_name
                })

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
    args = parser.parse_args()

    policy_cls = POLICIES[args.policy]

    try:
        policy = policy_cls(pfs_bandwidth_bps=args.pfs_bw)
    except TypeError:
        policy = policy_cls()

    try:
        serve(port=args.port, policy=policy)
    except KeyboardInterrupt:
        print("Orchestrator shutting down")
