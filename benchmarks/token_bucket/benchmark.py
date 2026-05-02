#!/usr/bin/env python3

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path


THROUGHPUTS_MIB_PER_SECOND = [
    8,
    64,
    256,
    1024,
    4096,
    10240,
    30720,
]

CHUNK_SIZES_BYTES = [
    8192,
    65536,
    262144,
    4194304,
    10485760,
    31457280,
]


def now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def get_arch() -> str:
    try:
        return subprocess.check_output(["arch"], text=True).strip()
    except Exception:
        return platform.machine()


def mib_to_bytes(value: int) -> int:
    return value * 1024 * 1024


def safe_name(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace(" ", "_")
        .replace(":", "_")
        .replace("-", "_")
    )


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        json.dump(row, f, sort_keys=True)
        f.write("\n")


def remove_if_exists(path: Path):
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def build_run_id(
    *,
    benchmark_id: str,
    arch: str,
    target: str,
    chunk_size: int,
    throughput_mib: int,
) -> dict:
    return {
        "benchmark_id": benchmark_id,
        "arch": arch,
        "target": target,
        "chunk_size_bytes": chunk_size,
        "throughput_mib_per_second": throughput_mib,
    }


def build_file_stem(
    *,
    benchmark_id: str,
    arch: str,
    target: str,
    chunk_size: int,
    throughput_mib: int,
) -> str:
    return (
        f"{benchmark_id}"
        f"__arch_{safe_name(arch)}"
        f"__target_{safe_name(target)}"
        f"__chunk_{chunk_size}"
        f"__throughput_{throughput_mib}MiBps"
    )


def run_token_bucket(
    *,
    python_executable: str,
    src: Path,
    dst: Path,
    throughput_bytes: int,
    chunk_size: int,
    json_output: Path,
    dry_run: bool,
):
    cmd = [
        python_executable,
        "-m",
        "migrater.token_bucket",
        str(src),
        str(dst),
        str(throughput_bytes),
        "--chunk-size",
        str(chunk_size),
        "--json-output",
        str(json_output),
    ]

    if dry_run:
        cmd.append("--dry-run")

    subprocess.run(cmd, check=True)


def add_overhead_metrics(
    *,
    row: dict,
    local_baseline: dict | None,
) -> dict:
    if local_baseline is None:
        row["overhead_vs_local"] = None
        return row

    current = row["token_bucket_metrics"]
    baseline = local_baseline["token_bucket_metrics"]

    current_time = current["actual_time_seconds"]
    baseline_time = baseline["actual_time_seconds"]

    current_tput = current["actual_effective_throughput_bytes_per_second"]
    baseline_tput = baseline["actual_effective_throughput_bytes_per_second"]

    row["overhead_vs_local"] = {
        "time_delta_seconds": current_time - baseline_time,
        "time_ratio_pfs_local": current_time / baseline_time,
        "throughput_delta_bytes_per_second": current_tput - baseline_tput,
        "throughput_ratio_pfs_local": current_tput / baseline_tput,
    }

    return row


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark token bucket copy across targets, rates and chunk sizes."
    )
    parser.add_argument(
        "--src",
        required=True,
        type=Path,
        help="Input file to copy from local storage",
    )
    parser.add_argument(
        "--local-dir",
        required=True,
        type=Path,
        help="Directory on local node storage for Local -> Local benchmark",
    )
    parser.add_argument(
        "--pfs-dir",
        required=True,
        type=Path,
        help="Directory on PFS for Local -> PFS benchmark",
    )
    parser.add_argument(
        "--results-file",
        required=True,
        type=Path,
        help="Aggregate JSONL output file. New rows are appended.",
    )
    parser.add_argument(
        "--metrics-dir",
        required=True,
        type=Path,
        help="Directory for per-run token bucket JSON metric files",
    )
    parser.add_argument(
        "--benchmark-id",
        default=None,
        help="Optional benchmark id. Defaults to timestamp + arch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pass --dry-run to token bucket for fast functional testing.",
    )
    parser.add_argument(
        "--keep-outputs",
        action="store_true",
        help="Keep copied destination files instead of deleting them after each run.",
    )
    parser.add_argument(
        "--only-target",
        choices=["Local", "PFS"],
        default=None,
        help="Run only one target type.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use. Defaults to current interpreter.",
    )

    args = parser.parse_args()

    arch = get_arch()
    benchmark_id = args.benchmark_id or f"{now_id()}_{arch}"

    args.local_dir.mkdir(parents=True, exist_ok=True)
    args.pfs_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    args.results_file.parent.mkdir(parents=True, exist_ok=True)

    targets = [
        ("Local", args.local_dir),
        ("PFS", args.pfs_dir),
    ]

    if args.only_target:
        targets = [item for item in targets if item[0] == args.only_target]

    print(f"Benchmark id: {benchmark_id}")
    print(f"Architecture: {arch}")
    print(f"Source:       {args.src}")
    print(f"Results:      {args.results_file}")
    print(f"Metrics dir:  {args.metrics_dir}")
    print(f"Dry run:      {args.dry_run}")
    print()

    total_runs = (
        len(targets)
        * len(THROUGHPUTS_MIB_PER_SECOND)
        * len(CHUNK_SIZES_BYTES)
    )
    run_index = 0

    # Baseline used to compute PFS overhead:
    # key = (throughput_mib, chunk_size)
    local_baselines: dict[tuple[int, int], dict] = {}

    for throughput_mib in THROUGHPUTS_MIB_PER_SECOND:
        throughput_bytes = mib_to_bytes(throughput_mib)

        for chunk_size in CHUNK_SIZES_BYTES:
            for target_name, target_dir in targets:
                run_index += 1

                run_id = build_run_id(
                    benchmark_id=benchmark_id,
                    arch=arch,
                    target=target_name,
                    chunk_size=chunk_size,
                    throughput_mib=throughput_mib,
                )

                file_stem = build_file_stem(
                    benchmark_id=benchmark_id,
                    arch=arch,
                    target=target_name,
                    chunk_size=chunk_size,
                    throughput_mib=throughput_mib,
                )

                dst = target_dir / f"{file_stem}.out"
                per_run_json = args.metrics_dir / f"{file_stem}.json"

                remove_if_exists(dst)
                remove_if_exists(per_run_json)

                print(
                    f"[{run_index}/{total_runs}] "
                    f"target={target_name} "
                    f"throughput={throughput_mib} MiB/s "
                    f"chunk={chunk_size} B"
                )

                start = time.monotonic()

                run_token_bucket(
                    python_executable=args.python,
                    src=args.src,
                    dst=dst,
                    throughput_bytes=throughput_bytes,
                    chunk_size=chunk_size,
                    json_output=per_run_json,
                    dry_run=args.dry_run,
                )

                end = time.monotonic()

                token_bucket_metrics = load_json(per_run_json)

                row = {
                    "id": run_id,
                    "benchmark_id": benchmark_id,
                    "timestamp": timestamp(),
                    "arch": arch,
                    "target": target_name,
                    "throughput_mib_per_second": throughput_mib,
                    "throughput_bytes_per_second": throughput_bytes,
                    "chunk_size_bytes": chunk_size,
                    "chunk_rate_ratio": chunk_size / throughput_bytes,
                    "benchmark_wall_time_seconds": end - start,
                    "token_bucket_metrics": token_bucket_metrics,
                    "per_run_metrics_json": str(per_run_json),
                    "dst": str(dst),
                    "dry_run": args.dry_run,
                }

                key = (throughput_mib, chunk_size)

                if target_name == "Local":
                    local_baselines[key] = row
                    row["overhead_vs_local"] = None

                elif target_name == "PFS":
                    row = add_overhead_metrics(
                        row=row,
                        local_baseline=local_baselines.get(key),
                    )

                append_jsonl(args.results_file, row)

                if not args.keep_outputs and not args.dry_run:
                    remove_if_exists(dst)

                print("  appended result")
                print()

    print("Benchmark completed.")
    print(f"Aggregate results: {args.results_file}")


if __name__ == "__main__":
    main()
