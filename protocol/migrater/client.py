import grpc
from protocol import cluster_pb2
from protocol import cluster_pb2_grpc


class MigraterClient:

    def __init__(self, addr="localhost:50051"):
        self.channel = grpc.insecure_channel(addr)
        self.stub = cluster_pb2_grpc.MigraterServiceStub(self.channel)

    def notify_checkpoint_saved(self, local, pfs, ts):

        request = cluster_pb2.CheckpointSavedRequest(
            checkpoint_local_path=local,
            checkpoint_pfs_path=pfs,
            timestamp=ts
        )

        self.stub.NotifyCheckpointSaved(request)
