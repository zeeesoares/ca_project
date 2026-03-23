import grpc
from concurrent import futures

from protocol.migrater import migrater_pb2
from protocol.migrater import migrater_pb2_grpc


class MigraterService(migrater_pb2_grpc.MigraterServiceServicer):

    def NotifyCheckpointSaved(self, request, context):

        print("Checkpoint notification received")
        print("local:",     request.checkpoint_local_path)
        print("pfs:",       request.checkpoint_pfs_path)
        print("timestamp:", request.timestamp)

        return migrater_pb2.CheckpointSavedResponse(ok=True)


def serve():

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))

    migrater_pb2_grpc.add_MigraterServiceServicer_to_server(
        MigraterService(),
        server
    )

    server.add_insecure_port("[::]:50051")
    server.start()

    print("Migrater running on 50051")

    server.wait_for_termination()


if __name__ == "__main__":
    serve()
