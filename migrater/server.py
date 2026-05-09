import argparse
import os
import grpc
import time
import socket
import threading

from utils.size_parser            import format_size
from concurrent                   import futures
from protocol                     import cluster_pb2
from protocol                     import cluster_pb2_grpc
from migrater.token_bucket        import token_bucket_copy
from migrater.progress            import ProgressBuffer
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
        self.pending_size     = 0                  # bytes waiting to be transferred; 0 = idle
        self.transfer_active  = False              # True while token_bucket_copy is running
        self.current_rate     = 0                  # bytes/s; updated by orchestrator instructions
        self.transfer_allowed = threading.Event()  # set when orchestrator permits transfer
        self.progress         = ProgressBuffer(maxlen=128)
        self.last_progress    = None
        self.epoch            = 0
        self.total_epochs     = 0

        threading.Thread(target=self._heartbeat_loop, daemon=True).start()

    def _heartbeat_loop(self):
        def stream():
            while True:
                progress_items = self.progress.pop()

                with self.lock:
                    pending = self.pending_size
                    migrating = self.transfer_active

                progress_proto = [
                    self._progress_to_proto(item)
                    for item in progress_items
                ]

                if progress_items:
                    latest = progress_items[-1]

                    with self.lock:
                        self.pending_size = latest["remaining_bytes"]
                        pending = self.pending_size

                    print(
                        "[migrater] "
                        f"timestamp={time.time():.2f}s, "
                        f"migrating={migrating}, "
                        f"epoch={self.epoch}, total_epochs={self.total_epochs}, "
                        f"configured={latest['configured_rate_bps']:.0f} B/s, "
                        f"rate={latest['interval_throughput_bps']:.0f} B/s, "
                        f"pending={pending} bytes "
                        f"([{(latest['bytes_copied'] / latest['total_bytes']) * 100 :.2f}%] "
                        f"{latest['bytes_copied']}/{latest['total_bytes']} bytes copied)"
                    )
                else:
                    print(f"[migrater] timestamp={time.time()}: migrating={migrating}, pending={pending}, "
                          f"epoch={self.epoch}, total_epochs={self.total_epochs}")

                yield cluster_pb2.HeartbeatRequest(
                    worker_id=socket.gethostname(),
                    checkpoint_size=float(pending),
                    is_migrating=migrating,
                    progress=progress_proto,
                    epoch=self.epoch,
                    total_epochs=self.total_epochs
                )

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
                self.current_rate = max(0, int(instruction.rate_limit_bps))
            self.transfer_allowed.set()

        # HOLD: do not touch transfer_allowed; once a transfer is running it continues

    def NotifyCheckpointSaved(self, request, context):
        local     = request.checkpoint_local_path
        pfs       = request.checkpoint_pfs_path
        epoch     = request.epoch
        total_epochs = request.total_epochs
        file_size = os.path.getsize(local)

        with self.lock:
            if self.transfer_active:
                # TODO: notify orchestrator and let it decide whether to preempt
                #       the current transfer in favour of this newer checkpoint.
                print(f"[migrater] transfer active — checkpoint ignored: {local}")
                return cluster_pb2.CheckpointSavedResponse(ok=False)

            self.checkpoint_size = file_size
            self.transfer_allowed.clear()  # require a fresh instruction for this transfer
            self.epoch = epoch
            self.total_epochs = total_epochs

        threading.Thread(
            target=self._do_transfer,
            args=(local, pfs),
            daemon=True,
        ).start()

        return cluster_pb2.CheckpointSavedResponse(ok=True)

    def get_current_rate(self) -> int:
        with self.lock:
            return self.current_rate

    def _do_transfer(self, local, pfs):
        # Block until the orchestrator permission (START_FLUSH or CHANGE_RATE)
        self.transfer_allowed.wait()

        with self.lock:
            self.transfer_active = True

        try:
            token_bucket_copy(
                local,
                pfs,
                throughput=self.get_current_rate,
                progress_update_interval=HEARTBEAT_INTERVAL,
                progress_callback=self.progress.add
            )
        finally:
            with self.lock:
                self.transfer_active = False
                self.checkpoint_size    = 0
                self.current_rate    = 0

    def _progress_to_proto(self, item: dict):
        return cluster_pb2.TransferProgress(
            timestamp=item["timestamp"],
            bytes_copied=int(item["bytes_copied"]),
            total_bytes=int(item["total_bytes"]),
            remaining_bytes=int(item["remaining_bytes"]),
            interval_seconds=float(item["interval_seconds"]),
            interval_throughput_bps=float(item["interval_throughput_bps"]),
            average_throughput_bps=float(item["average_throughput_bps"]),
            configured_rate_bps=float(item["configured_rate_bps"]),
            chunk_size_bytes=int(item["chunk_size_bytes"]),
        )


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
