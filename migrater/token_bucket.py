#!/usr/bin/env python3

import json
import os
import platform
import time
from typing import Callable, Union

###
# Token bucket constants and implementation
###

DEFAULT_CHUNK_SIZE = 8  # KB

SLEEP_CAP = 1  # percentage of the calculated sleep time to actually sleep
assert 0 < SLEEP_CAP <= 1, "Invalid sleep cap value"

PAUSE_SLEEP = 0.05  # 50 ms pause between checks when throughput is zero
assert PAUSE_SLEEP > 0, "Pause sleep time must be positive"


def token_bucket_copy(
    src: str,
    dst: str,
    throughput: Union[int, Callable[[], int]],
    chunk_size: int = DEFAULT_CHUNK_SIZE * 1024,
):
    """
    Copy a file using token bucket rate limiting.

    Parameters:
        src (str): Source file path (local node storage)
        dst (str): Destination file path (PFS)
        throughput (int | Callable[[], int]):
            Transfer rate in bytes per second. Can be:
                - int: fixed throughput
                - Callable: dynamic throughput provider
            Asserted to be positive.
        chunk_size (int): Size of each read/write chunk (default: 8KB)

    Raises:
        IOError: If there is an error reading from the source or writing
            to the destination.

    TODO:
        - Dynamically adjust the chunk size based on the observed throughput
            and latency.

    Possible improvements:
        - Allow partial writes
    """
    # Check if source file exists
    if not os.path.isfile(src):
        raise IOError(f"Source file does not exist: {src}")

    # Ensure destination directory exists
    dst_dir = os.path.dirname(dst)
    if dst_dir:
        os.makedirs(dst_dir, exist_ok=True)

    # INFO destination file check is skipped to slighty reduce initial overhead
    # Check if destination file can be written
    # try:
    #     with open(dst, "wb"):
    #         pass
    # except Exception as e:
    #     raise IOError(f"Cannot write to destination: {dst}") from e

    def get_throughput() -> int:
        return throughput() if callable(throughput) else throughput

    # Initial throughput check
    rate = get_throughput()
    assert rate >= 0, "Throughput must be non-negative"

    # Chunk size check
    assert chunk_size > 0, "Chunk size must be positive"

    capacity = rate  # burst-limited token bucket with 1-second capacity
    tokens = 0  # start with empty bucket to avoid initial burst
    last = time.monotonic()

    with open(src, "rb") as src_f, open(dst, "wb") as dst_f:
        while True:
            now = time.monotonic()
            elapsed = now - last
            last = now

            rate = get_throughput()
            assert rate >= 0, "Throughput must be non-negative"

            if rate == 0:
                tokens = 0
                time.sleep(PAUSE_SLEEP)
                continue

            capacity = rate

            # Refill tokens
            tokens = min(capacity, tokens + elapsed * rate)

            # Clamp tokens if rate decreased
            tokens = min(tokens, capacity)

            if tokens < chunk_size:
                # Dynamic sleep time
                missing = chunk_size - tokens
                sleep_time = missing / rate

                # Cap sleep to improve responsiveness to dynamic updates
                # INFO As a improvement we don't sleep the full deficit to
                # allow for quicker adjustments to dynamic throughput changes.
                time.sleep(min(sleep_time * SLEEP_CAP, 0.01))
                continue

            data = src_f.read(chunk_size)
            if not data:
                break

            dst_f.write(data)
            tokens -= len(data)

        # Ensure data is flushed to disk
        dst_f.flush()
        os.fsync(dst_f.fileno())


###
# Auxiliary functions for metrics collection and reporting
###

def expected_copy_time(file_size: int, throughput: int) -> float:
    assert throughput > 0, "Throughput must be positive for expected time"
    return file_size / throughput


def build_metrics(
    *,
    src: str,
    dst: str,
    file_size: int,
    throughput: int,
    chunk_size: int,
    expected_time: float,
    actual_time: float,
    dry_run: bool,
) -> dict:
    expected_effective_throughput = file_size / expected_time
    actual_effective_throughput = file_size / actual_time

    return {
        "src": src,
        "dst": dst,
        "file_size_bytes": file_size,
        "throughput_bytes_per_second": throughput,
        "chunk_size_bytes": chunk_size,
        "expected_time_seconds": expected_time,
        "actual_time_seconds": actual_time,
        "expected_effective_throughput_bytes_per_second": expected_effective_throughput,
        "actual_effective_throughput_bytes_per_second": actual_effective_throughput,
        "time_ratio_actual_expected": actual_time / expected_time,
        "throughput_ratio_actual_expected": (
            actual_effective_throughput / expected_effective_throughput
        ),
        "dry_run": dry_run,
        "arch": platform.machine(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def write_json_metrics(path: str, metrics: dict):
    dst_dir = os.path.dirname(path)
    if dst_dir:
        os.makedirs(dst_dir, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")


###
# Main function for command-line execution, testing and benchmarking
###

if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(
        description="Copy a file using token bucket rate limiting."
    )
    argparser.add_argument(
        "src",
        type=str,
        help="Source file path",
    )
    argparser.add_argument(
        "dst",
        type=str,
        help="Destination file path",
    )
    argparser.add_argument(
        "throughput",
        type=int,
        help="Transfer rate in bytes per second",
    )
    argparser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE * 1024,
        help=(
            "Size of each read/write chunk "
            f"(default: {DEFAULT_CHUNK_SIZE}KB)"
        ),
    )
    argparser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Write copy metrics to this JSON file",
    )
    argparser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Do not copy or sleep. Produce fake metrics. "
            "Useful for testing benchmark orchestration."
        ),
    )

    args = argparser.parse_args()

    file_size = os.path.getsize(args.src)
    expected_time = expected_copy_time(file_size, args.throughput)

    print(
        f"Starting token bucket copy:\n"
        f"  - Source:      {args.src}\n"
        f"  - Destination: {args.dst}\n"
        f"  - Throughput:  {args.throughput} B/s\n"
        f"  - Chunk size:  {args.chunk_size} B\n"
        f"  - Sleep cap:   {SLEEP_CAP * 100:.0f}% of calculated sleep time\n"
        f"  - Dry run:     {args.dry_run}"
    )

    if args.dry_run:
        # Fake metrics:
        # - actual time is 105% of expected
        # - actual throughput is therefore roughly below expected
        actual_time = expected_time * 1.05

        # Optional: make fake throughput exactly 95%.
        # This makes the fake metrics intentionally illustrative rather than
        # physically consistent with file_size / actual_time.
        actual_effective_throughput = args.throughput * 0.95
        actual_time = file_size / actual_effective_throughput

    else:
        start = time.monotonic()

        token_bucket_copy(
            args.src,
            args.dst,
            args.throughput,
            args.chunk_size,
        )

        end = time.monotonic()
        actual_time = end - start

    metrics = build_metrics(
        src=args.src,
        dst=args.dst,
        file_size=file_size,
        throughput=args.throughput,
        chunk_size=args.chunk_size,
        expected_time=expected_time,
        actual_time=actual_time,
        dry_run=args.dry_run,
    )

    print()
    print("Token bucket copy completed.")

    print()
    print(f"File size:                     {metrics['file_size_bytes']} B")

    print()
    print(f"Expected time:                 {metrics['expected_time_seconds']:.2f} s")
    print(f"Actual time:                   {metrics['actual_time_seconds']:.2f} s")

    print()
    print(
        f"Expected effective throughput: "
        f"{metrics['expected_effective_throughput_bytes_per_second']:.0f} B/s"
    )
    print(
        f"Actual effective throughput:   "
        f"{metrics['actual_effective_throughput_bytes_per_second']:.0f} B/s"
    )

    print()
    print(
        f"{'Time ratio actual/expected:':<36}"
        f"{metrics['time_ratio_actual_expected']:>8.2%}"
    )
    print(
        f"{'Throughput ratio actual/expected:':<36}"
        f"{metrics['throughput_ratio_actual_expected']:>8.2%}"
    )

    if args.json_output:
        write_json_metrics(args.json_output, metrics)
        print()
        print(f"Wrote metrics JSON: {args.json_output}")
