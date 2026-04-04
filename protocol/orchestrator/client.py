import grpc

from protocol import cluster_pb2
from protocol import cluster_pb2_grpc

ORCHESTRATOR_ADDR = "localhost:50052"


class OrchestratorClient:

    def __init__(self, worker_id: str, addr: str = ORCHESTRATOR_ADDR):
        self.worker_id = worker_id
        self.channel = grpc.insecure_channel(addr)
        self.stub = cluster_pb2_grpc.OrchestratorServiceStub(self.channel)

    def _heartbeat_stream(self, heartbeats):
        """Wrap an iterable of (pending_data_size, is_migrating) tuples into HeartbeatRequests."""
        for pending_data_size, is_migrating in heartbeats:
            yield cluster_pb2.HeartbeatRequest(
                worker_id=self.worker_id,
                pending_data_size=pending_data_size,
                is_migrating=is_migrating,
            )

    def monitor(self, heartbeats):
        """Stream heartbeats to the orchestrator and yield back InstructionResponses.

        Args:
            heartbeats: iterable of (pending_data_size: float, is_migrating: bool)

        Yields:
            cluster_pb2.InstructionResponse
        """
        for instruction in self.stub.Monitor(self._heartbeat_stream(heartbeats)):
            yield instruction

    def close(self):
        self.channel.close()
