#!/usr/bin/env python3
"""
Orchestrated checkpoint worker — local-staging + migrater pipeline.

Simulates the proposed system: torch.save() writes to local NVMe, then
NotifyCheckpointSaved() hands off to the migrater for async flush to PFS.

For each checkpoint:
  1. Write a synthetic file to the local NVMe directory
     (simulates torch.save() to local SSD — fast, < 1 s for 256 MB).
  2. Call NotifyCheckpointSaved() on the locally-running migrater.
     Training can continue after this call returns.
  3. Record stall_s = local_write + gRPC notify (≪ baseline stall).
  4. Poll the PFS destination until the file appears at the correct size
     (migration completed by the migrater in the background).
  5. Record migration_s = time from start of step 1 to PFS completion.

Requires:
  - migrater.server running at localhost:50051 (started by the SLURM script
    before this process is launched)
  - orchestrator.server reachable at ORCH_HOST:ORCH_PORT (separate node)

Invoked by scripts/bench-worker.sbatch via SLURM.

Required env vars (set by sbatch --export):
  EXPERIMENT_TAG, WORKER_ID, N_CHECKPOINTS, CHECKPOINT_SIZE_BYTES,
  LOCAL_DIR, PFS_DIR, RESULTS_DIR, POLICY, PFS_BW_BPS, N_WORKERS
"""

import argparse
import subprocess
import json
import os
import socket
import time
from pathlib import Path
from typing import Optional
import shutil

import grpc
# from numpy.ma import size

from src.protocol import cluster_pb2, cluster_pb2_grpc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

POLL_INTERVAL_S = 0.05   # 50 ms polling for PFS file completion
DEFAULT_TIMEOUT_MULTIPLIER = 10
MIN_TIMEOUT_S = 60.0


def write_file(path: Path, size_bytes: int):
    """Write zero-filled file of size_bytes and fsync."""
    path.parent.mkdir(parents=True, exist_ok=True)
    chunk = 8 * 1024 * 1024
    with open(path, "wb", buffering=0) as f:
        remaining = size_bytes
        while remaining > 0:
            n = min(chunk, remaining)
            f.write(b"\x00" * n)
            remaining -= n
        f.flush()
        os.fsync(f.fileno())


def write_checkpoint(dst_path: Path, size_bytes: int, dummy_src: Optional[Path] = None):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    if dummy_src and dummy_src.exists():
        shutil.copy2(dummy_src, dst_path)
    else:
        chunk = 8 * 1024 * 1024
        with open(dst_path, "wb", buffering=0) as f:
            remaining = size_bytes
            while remaining > 0:
                n = min(chunk, remaining)
                f.write(b"\x00" * n)
                remaining -= n
            f.flush()
            os.fsync(f.fileno())



def poll_pfs_complete(
    pfs_path: Path,
    expected_size: int,
    timeout_s: float,
) -> Optional[float]:
    """
    Block until pfs_path has expected_size bytes (transfer complete).
    Returns the monotonic time of detection, or None on timeout.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if pfs_path.exists() and pfs_path.stat().st_size == expected_size:
                return time.monotonic()
        except OSError:
            pass
        time.sleep(POLL_INTERVAL_S)
    return None


def wait_for_migrater(addr: str = "localhost:50051", timeout_s: float = 15.0):
    """Wait until the migrater gRPC server is reachable."""
    channel = grpc.insecure_channel(addr)
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout_s)
    except grpc.FutureTimeoutError:
        raise TimeoutError(f"Migrater not reachable at {addr} after {timeout_s:.0f}s")
    finally:
        channel.close()


def format_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def timestamp_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        json.dump(row, f, sort_keys=True, default=str)
        f.write("\n")

def get_nr_of_concurrent_jobs() -> Optional[int]:
    """
    Returns the number of currently queued/running SLURM jobs visible to squeue.

    If squeue is not available, returns None.
    """
    if shutil.which("squeue") is None:
        return None

    try:
        output = subprocess.check_output(
            ["bash", "-c", "squeue --noheader --format=%i | wc -l"],
            text=True,
        )
        return int(output.strip())

    except Exception as e:
        print(f"WARNING: failed to collect squeue job count: {e}")
        return None

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Orchestrated worker: local staging + migrater async flush.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--experiment-tag",
                        default=os.environ.get("EXPERIMENT_TAG", "orchestrated"))
    parser.add_argument("--worker-id",
                        default=os.environ.get("WORKER_ID",
                                               os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    parser.add_argument("--n-checkpoints",
                        type=int,
                        default=int(os.environ.get("N_CHECKPOINTS", "3")))
    parser.add_argument("--checkpoint-size",
                        type=int,
                        default=int(os.environ.get("CHECKPOINT_SIZE_BYTES",
                                                    str(256 * 1024 * 1024))))
    parser.add_argument("--local-dir",
                        type=Path,
                        default=Path(os.environ.get("LOCAL_DIR", "/tmp/bench_local")))
    parser.add_argument("--pfs-dir",
                        type=Path,
                        default=Path(os.environ.get("PFS_DIR", "/tmp/bench_pfs")))
    parser.add_argument("--results-dir",
                        type=Path,
                        default=Path(os.environ.get("RESULTS_DIR", "/tmp/bench_results")))
    parser.add_argument("--migrater-addr",
                        default="localhost:50051",
                        help="Address of the local migrater service")
    parser.add_argument("--policy",
                        default=os.environ.get("POLICY", "uniform-fair-share"))
    parser.add_argument("--pfs-bw-bps",
                        type=int,
                        default=int(os.environ.get("PFS_BW_BPS",
                                                    str(512 * 1024 * 1024))))
    parser.add_argument("--n-workers",
                        type=int,
                        default=int(os.environ.get("N_WORKERS", "1")))
    parser.add_argument("--dummy-src",
                        type=Path,
                        default=None,
                        help="Optional path to a pre-generated dummy checkpoint file to copy instead of writing zeros")
    parser.add_argument("--compute-time",
                        type=float,
                        default=0.0,
                        help="Tempo em segundos para simular o treino entre checkpoints")
    parser.add_argument("--jitter",
                        type=float,
                        default=float(os.environ.get("JITTER", "0.0")),
                        help="Tempo em segundos para adicionar jitter aleatório antes do primeiro checkpoint")
    args = parser.parse_args()

    hostname   = socket.gethostname()
    worker_id  = f"worker_{args.worker_id}"
    slurm_job  = os.environ.get("SLURM_JOB_ID", "local")
    slurm_task = os.environ.get("SLURM_ARRAY_TASK_ID", args.worker_id)

    results_file = (
        args.results_dir
        / args.experiment_tag
        / f"orchestrated_{slurm_job}_{slurm_task}.jsonl"
    )

    args.local_dir.mkdir(parents=True, exist_ok=True)
    args.pfs_dir.mkdir(parents=True, exist_ok=True)

    expected_per_cp_s = args.checkpoint_size / (args.pfs_bw_bps / args.n_workers)
    timeout_s = max(MIN_TIMEOUT_S, expected_per_cp_s * DEFAULT_TIMEOUT_MULTIPLIER)

    print(f"[{worker_id}@{hostname}] Orchestrated worker starting")
    print(f"  Experiment  : {args.experiment_tag}")
    print(f"  Policy      : {args.policy}  PFS BW: {args.pfs_bw_bps / (1024**2):.0f} MB/s")
    print(f"  Workers     : {args.n_workers}")
    print(f"  Checkpoints : {args.n_checkpoints} × {format_size(args.checkpoint_size)}")
    print(f"  Local dir   : {args.local_dir}")
    print(f"  PFS dir     : {args.pfs_dir}")
    print(f"  Migrater    : {args.migrater_addr}")
    print(f"  Results     : {results_file}")
    print()

    print("  Waiting for migrater...", flush=True)
    wait_for_migrater(args.migrater_addr)
    print("  Migrater ready.")

    # Allow one heartbeat cycle so the migrater-orchestrator stream is
    # established and the first START_FLUSH has been acknowledged.
    time.sleep(0.7)

    channel = grpc.insecure_channel(args.migrater_addr)
    stub    = cluster_pb2_grpc.MigraterServiceStub(channel)

    # Jitter before starting, if configured (simulates variability in time to first checkpoint)
    if args.jitter > 0:
        print(f"  Applying initial jitter of {args.jitter:.2f}s before first checkpoint...", flush=True)
        time.sleep(args.jitter)

    for i in range(args.n_checkpoints):
        cp_local = args.local_dir / f"ckpt_{worker_id}_{i}.bin"
        cp_pfs   = args.pfs_dir   / f"ckpt_{worker_id}_{i}.bin"

        cp_local.unlink(missing_ok=True)
        cp_pfs.unlink(missing_ok=True)

        print(f"  [{worker_id}] checkpoint {i + 1}/{args.n_checkpoints}", flush=True)

        # ---- STALL begins (equivalent to torch.save + notify) ----
        t_stall_start = time.monotonic()

        # Write to local NVMe — fast (simulates torch.save to local SSD)
        write_checkpoint(cp_local, args.checkpoint_size, args.dummy_src)

        # Notify migrater — non-blocking from training perspective
        response = stub.NotifyCheckpointSaved(
            cluster_pb2.CheckpointSavedRequest(
                checkpoint_local_path=str(cp_local),
                checkpoint_pfs_path=str(cp_pfs),
                timestamp=time.time(),
                epoch=i + 1,
                total_epochs=args.n_checkpoints,
            )
        )
        t_stall_end = time.monotonic()
        # ---- STALL ends — training can continue from here ----

        stall_s = t_stall_end - t_stall_start
        notified_ok = bool(response.ok)

        if not notified_ok:
            print(
                f"    WARNING: migrater returned ok=False — "
                f"a transfer is already active. Waiting 1 s and retrying...",
                flush=True,
            )
            # Back off and retry once: the previous migration might still be active.
            time.sleep(1.0)
            response = stub.NotifyCheckpointSaved(
                cluster_pb2.CheckpointSavedRequest(
                    checkpoint_local_path=str(cp_local),
                    checkpoint_pfs_path=str(cp_pfs),
                    timestamp=time.time(),
                    epoch=i + 1,
                    total_epochs=args.n_checkpoints,
                )
            )
            notified_ok = bool(response.ok)

        # --- ASYNC migration: poll PFS until transfer completes ---
        t_done = poll_pfs_complete(cp_pfs, args.checkpoint_size, timeout_s)
        timed_out = t_done is None
        t_done    = t_done or time.monotonic()

        migration_s = t_done - t_stall_start
        
        transfer_time = migration_s - stall_s
        throughput = args.checkpoint_size / transfer_time if transfer_time > 0 else 0.0

        row = {
            "mode":                    "orchestrated",
            "experiment_tag":          args.experiment_tag,
            "worker_id":               worker_id,
            "hostname":                hostname,
            "slurm_job_id":            slurm_job,
            "slurm_task_id":           slurm_task,
            "checkpoint_index":        i,
            "checkpoint_size_bytes":   args.checkpoint_size,
            "n_workers":               args.n_workers,
            "policy":                  args.policy,
            "pfs_bw_bps":              args.pfs_bw_bps,
            "stall_s":                 stall_s,
            "migration_s":             migration_s,
            "effective_throughput_bps": throughput,
            "notified_ok":             notified_ok,
            "timed_out":               timed_out,
            "timestamp":               timestamp_iso(),
            "concurrent_jobs":         get_nr_of_concurrent_jobs(),
        }

        append_jsonl(results_file, row)

        cp_local.unlink(missing_ok=True)
        cp_pfs.unlink(missing_ok=True)

        if i < args.n_checkpoints - 1 and args.compute_time > 0:
            print(f"    [Simulação] Treinando época {i + 2} por {args.compute_time:.2f}s...", flush=True)
            time.sleep(args.compute_time)

        status = "TIMEOUT" if timed_out else "ok"
        print(
            f"    stall={stall_s:.3f}s  "
            f"migration={migration_s:.2f}s  "
            f"throughput={throughput / (1024**2):.1f} MB/s  "
            f"[{status}]"
        )

    channel.close()
    print(f"\n[{worker_id}] Done. Results: {results_file}")


if __name__ == "__main__":
    main()
