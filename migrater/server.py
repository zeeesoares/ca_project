import grpc
from concurrent import futures

from protocol import cluster_pb2
from protocol import cluster_pb2_grpc
from protocol.orchestrator.client import OrchestratorClient


class MigraterService(cluster_pb2_grpc.MigraterServiceServicer):

    def __init__(self):
        super().__init__()
        self.orchestrator_client = OrchestratorClient()
        
    def NotifyCheckpointSaved(self, request, context):

        print("Checkpoint notification received")
        print("local:",     request.checkpoint_local_path)
        print("pfs:",       request.checkpoint_pfs_path)
        print("timestamp:", request.timestamp)

        return cluster_pb2.CheckpointSavedResponse(ok=True)


def serve():

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))

    cluster_pb2_grpc.add_MigraterServiceServicer_to_server(
        MigraterService(),
        server
    )

    server.add_insecure_port("[::]:50051")
    server.start()

    print("Migrater running on 50051")

    server.wait_for_termination()


if __name__ == "__main__":
    try:
        serve()
    except KeyboardInterrupt:
        print("Migrater shutting down")
