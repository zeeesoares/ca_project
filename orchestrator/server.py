import grpc
from concurrent import futures

from protocol import cluster_pb2
from protocol import cluster_pb2_grpc

class OrchestratorService(cluster_pb2_grpc.OrchestratorServiceServicer):

    def Monitor(self, request_iterator, context):

        for heartbeat in request_iterator:
            print("Heartbeat received from", heartbeat.worker_id)

            # Future implementation area

            instruction = cluster_pb2.InstructionResponse(
                action=cluster_pb2.IntructionResponse.HOLD,
                rate_limit_bps=0.0
            )
            yield instruction
    

def serve():

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))

    cluster_pb2_grpc.add_OrchestratorServiceServicer_to_server(
        OrchestratorService(),
        server
    )

    server.add_insecure_port("[::]:50051")
    server.start()

    print("Orchestrator running on 50051")

    server.wait_for_termination()


if __name__ == "__main__":
    try:
        serve()
    except KeyboardInterrupt:
        print("Orchestrator shutting down")