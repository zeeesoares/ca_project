import argparse
import os
import grpc
import time
import socket
import threading

from concurrent import futures

from src.protocol                     import cluster_pb2, cluster_pb2_grpc
from src.migrater.token_bucket        import token_bucket_copy
from src.migrater.progress            import ProgressBuffer
from src.protocol.orchestrator.client import OrchestratorClient

# from src.utils.size_parser        import format_size

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

        self.shutdown_requested = threading.Event()
        self.shutdown_complete  = threading.Event()

        threading.Thread(target=self._heartbeat_loop, daemon=True).start()

    def _heartbeat_loop(self):
        def stream():
            while not self.shutdown_complete.is_set():
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

                    # Use interval throughput so the per-heartbeat sample reflects
                    # the current rate (not a long-running average that lags rate
                    # changes from the orchestrator).
                    instant_rate = latest["interval_throughput_bps"] if migrating else 0.0

                    print(
                        "[migrater] "
                        f"timestamp={latest['timestamp']:.2f}, "
                        f"migrating={migrating}, "
                        f"epoch={self.epoch}, total_epochs={self.total_epochs}, "
                        f"configured={latest['configured_rate_bps']:.0f} B/s, "
                        f"rate={instant_rate:.0f} B/s, "
                        f"pending={pending} bytes "
                        f"([{(latest['bytes_copied'] / latest['total_bytes']) * 100 :.2f}%] "
                        f"{latest['bytes_copied']}/{latest['total_bytes']} bytes copied)"
                    )
                else:
                    # Emit a regex-matching line even when idle, so downstream
                    # plotting can see that this worker is consuming 0 B/s
                    # instead of carrying forward its last observed rate.
                    print(
                        "[migrater] "
                        f"timestamp={time.time():.2f}, "
                        f"migrating={migrating}, "
                        f"epoch={self.epoch}, total_epochs={self.total_epochs}, "
                        f"configured=0 B/s, "
                        f"rate=0 B/s, "
                        f"pending={pending} bytes"
                    )

                yield cluster_pb2.HeartbeatRequest(
                    worker_id=socket.gethostname(),
                    checkpoint_size=float(pending),
                    is_migrating=migrating,
                    progress=progress_proto,
                    epoch=self.epoch,
                    total_epochs=self.total_epochs
                )

                time.sleep(HEARTBEAT_INTERVAL)

        try:
            for instruction in self.orchestrator_client.monitor(stream()):
                self._handle_instruction(instruction)

                if instruction.action == cluster_pb2.InstructionResponse.EXIT:
                    break

        except grpc.RpcError as e:
            print(f"[migrater] heartbeat stream ended: {e}")

        finally:
            if self.shutdown_requested.is_set():
                with self.lock:
                    active = self.transfer_active
                    pending = self.pending_size

                if not active and pending == 0:
                    self.shutdown_complete.set()

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

        elif action == cluster_pb2.InstructionResponse.EXIT:
            print("[migrater] received EXIT instruction from orchestrator")
            self.shutdown_requested.set()

            with self.lock:
                if not self.transfer_active:
                    self.shutdown_complete.set()

                # If a transfer is pending but not yet started, allow it to start.
                # If current_rate is 0, it would otherwise wait forever.
                if self.pending_size > 0 and self.current_rate <= 0:
                    self.current_rate = int(instruction.rate_limit_bps) or 500_000_000

            self.transfer_allowed.set()

        # HOLD: do not touch transfer_allowed; once a transfer is running it continues

    def NotifyCheckpointSaved(self, request, context):
        if self.shutdown_requested.is_set():
            print("[migrater] shutdown requested — checkpoint rejected")
            return cluster_pb2.CheckpointSavedResponse(ok=False)

        local        = request.checkpoint_local_path
        pfs          = request.checkpoint_pfs_path
        epoch        = request.epoch
        total_epochs = request.total_epochs
        file_size    = os.path.getsize(local)

        with self.lock:
            if self.transfer_active:
                # TODO: notify orchestrator and let it decide whether to preempt
                #       the current transfer in favour of this newer checkpoint.
                print(f"[migrater] transfer active — checkpoint ignored: {local}")
                return cluster_pb2.CheckpointSavedResponse(ok=False)

            if self.shutdown_requested.is_set():
                print("[migrater] shutdown requested — checkpoint rejected")
                return cluster_pb2.CheckpointSavedResponse(ok=False)

            self.transfer_allowed.clear()  # require a fresh instruction for this transfer
            self.pending_size = file_size
            self.checkpoint_size = file_size
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
                self.checkpoint_size = 0
                self.current_rate    = 0

            if self.shutdown_requested.is_set():
                print("[migrater] transfer completed after EXIT")
                self.shutdown_complete.set()

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

    service = MigraterService(orchestrator_addr, orchestrator_port)

    cluster_pb2_grpc.add_MigraterServiceServicer_to_server(service, server)

    server.add_insecure_port("[::]:50051")
    server.start()

    print("Migrater running on 50051")

    try:
        while True:
            if service.shutdown_complete.wait(timeout=0.5):
                print("[migrater] graceful shutdown complete")
                server.stop(grace=2)
                break

    except KeyboardInterrupt:
        print("Migrater shutting down")
        service.shutdown_requested.set()

        with service.lock:
            active = service.transfer_active
            pending = service.pending_size

        if not active and pending == 0:
            service.shutdown_complete.set()

        service.shutdown_complete.wait()
        server.stop(grace=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--orchestrator-addr", type=str, default="localhost")
    parser.add_argument("--orchestrator-port", type=int, default=50052)

    args = parser.parse_args()

    try:
        serve(args.orchestrator_addr, args.orchestrator_port)
    except KeyboardInterrupt:
        print("Migrater shutting down")
