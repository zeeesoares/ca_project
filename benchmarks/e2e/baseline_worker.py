#!/usr/bin/env python3
"""
Baseline checkpoint worker — direct PFS write.

Simulates the training stall that occurs when torch.save() writes a
checkpoint directly to the parallel file system with no local staging.
This is the "no migrater" reference that the orchestrated solution is
compared against.

For each checkpoint:
  1. Write a synthetic file of checkpoint_size_bytes directly to the PFS
     destination (simulates torch.save → Lustre).
  2. fsync() to guarantee the write has reached the file system.
  3. Record wall-clock stall time and effective throughput.

Invoked by scripts/bench-baseline.sbatch via SLURM.

Required env vars (set by sbatch --export):
  EXPERIMENT_TAG, WORKER_ID, N_CHECKPOINTS, CHECKPOINT_SIZE_BYTES,
  PFS_DIR, RESULTS_DIR

Or pass equivalent CLI flags.
"""

import argparse
import json
import os
import socket
import sys
import subprocess
import shutil
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_file(path: Path, size_bytes: int):
    """Write size_bytes of zero bytes to path, then fsync."""
    path.parent.mkdir(parents=True, exist_ok=True)
    chunk = 8 * 1024 * 1024  # 8 MB write chunks
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
        description="Baseline: measure training stall for direct PFS writes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--experiment-tag",
                        default=os.environ.get("EXPERIMENT_TAG", "baseline"),
                        help="Unique identifier for this experiment run")
    parser.add_argument("--worker-id",
                        default=os.environ.get("WORKER_ID",
                                               os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
                        help="Worker index (from SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--n-checkpoints",
                        type=int,
                        default=int(os.environ.get("N_CHECKPOINTS", "3")),
                        help="Number of checkpoints to write")
    parser.add_argument("--checkpoint-size",
                        type=int,
                        default=int(os.environ.get("CHECKPOINT_SIZE_BYTES",
                                                    str(256 * 1024 * 1024))),
                        help="Checkpoint size in bytes")
    parser.add_argument("--pfs-dir",
                        type=Path,
                        default=Path(os.environ.get("PFS_DIR", "/tmp/bench_pfs")),
                        help="PFS directory for direct writes (Lustre on Deucalion)")
    parser.add_argument("--results-dir",
                        type=Path,
                        default=Path(os.environ.get("RESULTS_DIR", "/tmp/bench_results")),
                        help="Directory where per-worker JSONL results are written")
    parser.add_argument("--n-workers",
                        type=int,
                        default=int(os.environ.get("N_WORKERS", "1")),
                        help="Total number of concurrent workers (metadata only)")
    parser.add_argument("--dummy-src",
                        type=Path,
                        default=None,
                        help="Optional path to a dummy checkpoint file to copy instead of writing zeros")
    args = parser.parse_args()

    hostname   = socket.gethostname()
    worker_id  = f"worker_{args.worker_id}"
    slurm_job  = os.environ.get("SLURM_JOB_ID", "local")
    slurm_task = os.environ.get("SLURM_ARRAY_TASK_ID", args.worker_id)

    results_file = (
        args.results_dir
        / args.experiment_tag
        / f"baseline_{slurm_job}_{slurm_task}.jsonl"
    )

    args.pfs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{worker_id}@{hostname}] Baseline worker starting")
    print(f"  Experiment  : {args.experiment_tag}")
    print(f"  Checkpoints : {args.n_checkpoints} × {format_size(args.checkpoint_size)}")
    print(f"  PFS dir     : {args.pfs_dir}")
    print(f"  Results     : {results_file}")
    print()

    for i in range(args.n_checkpoints):
        cp_pfs = args.pfs_dir / f"ckpt_{worker_id}_{i}.bin"
        cp_pfs.unlink(missing_ok=True)

        print(f"  [{worker_id}] checkpoint {i + 1}/{args.n_checkpoints} → {cp_pfs.name}", flush=True)

        # ---- stall begins ----
        t_start = time.monotonic()
        write_checkpoint(cp_pfs, args.checkpoint_size, args.dummy_src)
        t_end = time.monotonic()
        # ---- stall ends ----

        stall_s      = t_end - t_start
        throughput   = args.checkpoint_size / stall_s if stall_s > 0 else 0.0
        migration_s  = stall_s  # in baseline, stall == migration (synchronous)

        row = {
            "mode":                    "baseline",
            "experiment_tag":          args.experiment_tag,
            "worker_id":               worker_id,
            "hostname":                hostname,
            "slurm_job_id":            slurm_job,
            "slurm_task_id":           slurm_task,
            "checkpoint_index":        i,
            "checkpoint_size_bytes":   args.checkpoint_size,
            "n_workers":               args.n_workers,
            "policy":                  None,
            "pfs_bw_bps":              None,
            "stall_s":                 stall_s,
            "migration_s":             migration_s,
            "effective_throughput_bps": throughput,
            "timed_out":               False,
            "timestamp":               timestamp_iso(),
            "concurrent_jobs":         get_nr_of_concurrent_jobs(),
        }

        append_jsonl(results_file, row)
        cp_pfs.unlink(missing_ok=True)

        print(
            f"    stall={stall_s:.3f}s  "
            f"throughput={throughput / (1024**2):.1f} MB/s"
        )

    print(f"\n[{worker_id}] Done. Results: {results_file}")


if __name__ == "__main__":
    main()
