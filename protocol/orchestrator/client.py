import grpc

from protocol import cluster_pb2
from protocol import cluster_pb2_grpc

ORCHESTRATOR_ADDR = "localhost:50052"


class OrchestratorClient:

    def __init__(self, worker_id: str, addr: str = ORCHESTRATOR_ADDR):
        self.worker_id = worker_id
        self.addr = addr
        self.channel = grpc.insecure_channel(addr)
        self.stub = cluster_pb2_grpc.OrchestratorServiceStub(self.channel)

    def _heartbeat_stream(self, heartbeats):
        for heartbeat in heartbeats:
            if not isinstance(heartbeat, cluster_pb2.HeartbeatRequest):
                raise TypeError(
                    "heartbeats must yield cluster_pb2.HeartbeatRequest "
                    f"instances, got {type(heartbeat).__name__}"
                )

            if not heartbeat.worker_id:
                heartbeat.worker_id = self.worker_id

            yield heartbeat

    def monitor(self, heartbeats):
        """
        Stream HeartbeatRequest messages to the orchestrator and yield
        InstructionResponse messages back.

        Args:
            heartbeats: iterable of cluster_pb2.HeartbeatRequest

        Yields:
            cluster_pb2.InstructionResponse
        """
        for instruction in self.stub.Monitor(self._heartbeat_stream(heartbeats)):
            yield instruction

    def close(self):
        self.channel.close()
