import argparse
import os
import grpc
import time
import socket
import threading

from concurrent                   import futures
from protocol                     import cluster_pb2
from protocol                     import cluster_pb2_grpc
from migrater.token_bucket        import token_bucket_copy
from protocol.orchestrator.client import OrchestratorClient

HEARTBEAT_INTERVAL = 0.5

class MigraterService(cluster_pb2_grpc.MigraterServiceServicer):

    def __init__(self, orch_addr, orch_port):
        super().__init__()
        orch_full_addr = f"{orch_addr}:{orch_port}"
        self.orchestrator_client = OrchestratorClient(
            worker_id=socket.gethostname(), 
            addr=orch_full_addr
        )

        self.lock             = threading.Lock()
        self.pending_size     = 0      # bytes waiting to be transferred; 0 = idle
        self.transfer_active  = False  # True while token_bucket_copy is running
        self.current_rate     = 0      # bytes/s; updated by orchestrator instructions
        self.transfer_allowed = threading.Event()  # set when orchestrator permits transfer

        threading.Thread(target=self._heartbeat_loop, daemon=True).start()

    def _heartbeat_loop(self):
        def stream():
            while True:
                with self.lock:
                    pending   = self.pending_size
                    migrating = self.transfer_active
                    print(f"[migrater] migrating={migrating}, pending={pending}")
                yield (pending, migrating)
                time.sleep(HEARTBEAT_INTERVAL)

        for instruction in self.orchestrator_client.monitor(stream()):
            self._handle_instruction(instruction)

    def _handle_instruction(self, instruction):
        action = instruction.action

        if action == cluster_pb2.InstructionResponse.START_FLUSH:
            with self.lock:
                self.current_rate = int(instruction.rate_limit_bps) 
            self.transfer_allowed.set()

        elif action == cluster_pb2.InstructionResponse.CHANGE_RATE:
            with self.lock:
                self.current_rate = max(1, int(instruction.rate_limit_bps))
            self.transfer_allowed.set()

        # HOLD: do not touch transfer_allowed; once a transfer is running it continues

    def NotifyCheckpointSaved(self, request, context):
        local     = request.checkpoint_local_path
        pfs       = request.checkpoint_pfs_path
        file_size = os.path.getsize(local)

        with self.lock:
            if self.transfer_active:
                # TODO: notify orchestrator and let it decide whether to preempt
                #       the current transfer in favour of this newer checkpoint.
                print(f"[migrater] transfer active — checkpoint ignored: {local}")
                return cluster_pb2.CheckpointSavedResponse(ok=False)

            self.pending_size = file_size
            self.transfer_allowed.clear()  # require a fresh instruction for this transfer

        threading.Thread(
            target=self._do_transfer,
            args=(local, pfs),
            daemon=True,
        ).start()

        return cluster_pb2.CheckpointSavedResponse(ok=True)


    def _do_transfer(self, local, pfs):
        # Block until the orchestrator permission (START_FLUSH or CHANGE_RATE)
        self.transfer_allowed.wait()

        with self.lock:
            self.transfer_active = True

        try:
            token_bucket_copy(local, pfs, throughput=lambda: self.current_rate)
        finally:
            with self.lock:
                self.transfer_active = False
                self.pending_size    = 0
                self.current_rate    = 0


def serve(orchestrator_addr, orchestrator_port):

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))

    cluster_pb2_grpc.add_MigraterServiceServicer_to_server(
        MigraterService(orchestrator_addr, orchestrator_port),
        server,
    )

    server.add_insecure_port("[::]:50051")
    server.start()

    print("Migrater running on 50051")

    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--orchestrator-addr", type=str, default="localhost")
    parser.add_argument("--orchestrator-port", type=int, default=50052)
    
    args = parser.parse_args()

    try:
        serve(args.orchestrator_addr, args.orchestrator_port)
    except KeyboardInterrupt:
        print("Migrater shutting down")