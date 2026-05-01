#!/usr/bin/env python3

import threading
import time
from migrater.token_bucket import token_bucket_copy

# List of: (rate in bytes/s, time to run in seconds)
# duration == 0 means "run until the copy finishes"
THROUGHPUT_ADJUSTMENTS = [
    (5_000_000,  3),  # 5   MB/s for 3s
    (1_000_000,  5),  # 1   MB/s for 5s
    (8_000_000,  4),  # 8   MB/s for 4s
    (  500_000,  1),  # 0.5 MB/s for 1s
    (        0, 10),  # pause    for 10s
    (5_000_000,  0),  # 5   MB/s until the end
]

# Shared state
current_rate = THROUGHPUT_ADJUSTMENTS[0][0]


def get_rate():
    return current_rate


def set_rate(rate: int):
    global current_rate
    current_rate = rate


def orchestrator():
    """
    Applies the throughput schedule.

    The first rate is already active when the copy starts.
    After each duration expires, the next rate is applied.
    """
    print(f"[orchestrator] Initial throughput is {format_rate(current_rate):>12}")

    for i in range(1, len(THROUGHPUT_ADJUSTMENTS)):
        previous_rate, previous_duration = THROUGHPUT_ADJUSTMENTS[i - 1]

        if previous_duration > 0:
            time.sleep(previous_duration)

        next_rate, _ = THROUGHPUT_ADJUSTMENTS[i]
        print(f"[orchestrator] Setting throughput to {format_rate(next_rate):>12}")
        set_rate(next_rate)


def format_rate(rate: int) -> str:
    if rate == 0:
        return "0  B/s"

    if rate >= 1_000_000_000:
        return f"{rate / 1_000_000_000:.2f} GB/s"

    if rate >= 1_000_000:
        return f"{rate / 1_000_000:.2f} MB/s"

    if rate >= 1_000:
        return f"{rate / 1_000:.2f} KB/s"

    return f"{rate}  B/s"


def expected_copy_time(file_size: int, schedule: list[tuple[int, int]]) -> float:
    """
    Computes expected copy time based on a changing throughput schedule.

    Each tuple is:
        (rate_in_bytes_per_second, duration_in_seconds)

    If duration == 0, that rate is assumed to continue until the copy ends.
    """
    remaining = file_size
    total_time = 0.0

    for rate, duration in schedule:
        if remaining <= 0:
            break

        if duration == 0:
            if rate == 0:
                raise ValueError(
                    "Final throughput cannot be 0 if data still remains."
                )

            total_time += remaining / rate
            remaining = 0
            break

        if rate == 0:
            total_time += duration
            continue

        bytes_copied_in_window = rate * duration

        if remaining <= bytes_copied_in_window:
            total_time += remaining / rate
            remaining = 0
            break

        remaining -= bytes_copied_in_window
        total_time += duration

    if remaining > 0:
        raise ValueError(
            "Throughput schedule ended before the whole file could be copied. "
            "Make sure the last schedule entry has duration 0."
        )

    return total_time


# Run with:
# python3 -m tests.token_bucket <src_file> <dst_file> [--chunk-size <size_in_bytes>]
if __name__ == "__main__":
    import argparse
    import os

    DEFAULT_CHUNK_SIZE = 64  # KB

    argparser = argparse.ArgumentParser(
        description="Test token bucket copy with dynamic throughput changes."
    )
    argparser.add_argument(
        "src",
        type=str,
        help="Source file path (local node storage)"
    )
    argparser.add_argument(
        "dst",
        type=str,
        help="Destination file path (PFS)"
    )
    argparser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE * 1024,
        help=f"Size of each read/write chunk (default: {DEFAULT_CHUNK_SIZE}KB)"
    )
    args = argparser.parse_args()

    initial_rate = THROUGHPUT_ADJUSTMENTS[0][0]
    set_rate(initial_rate)

    print("Throughput schedule:")
    for rate, duration in THROUGHPUT_ADJUSTMENTS:
        if duration == 0:
            duration_str = "until completion"
        else:
            duration_str = f"{duration}s"

        print(f"  - {format_rate(rate):>12} for {duration_str}")

    print()
    print(
        f"Starting token bucket copy:\n"
        f"  - Source:             {args.src}\n"
        f"  - Destination:        {args.dst}\n"
        f"  - Initial throughput: {format_rate(initial_rate)}\n"
        f"  - Chunk size:         {args.chunk_size} B"
    )
    print()

    file_size = os.path.getsize(args.src)
    expected_time = expected_copy_time(file_size, THROUGHPUT_ADJUSTMENTS)
    expected_effective_throughput = file_size / expected_time

    # Start orchestrator in background
    t = threading.Thread(target=orchestrator, daemon=True)
    t.start()

    start = time.monotonic()

    try:
        token_bucket_copy(args.src, args.dst, get_rate, args.chunk_size)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
        exit(1)

    end = time.monotonic()

    actual_time = end - start
    actual_effective_throughput = file_size / actual_time

    print()
    print("Token bucket copy completed.")

    print()
    print(f"File size:                        {file_size} B")

    print()
    print(f"Expected time:                    {expected_time:.2f} seconds")
    print(f"Actual time:                      {actual_time:.2f} seconds")

    print()
    print(f"Expected effective throughput:    {expected_effective_throughput:.0f} B/s")
    print(f"Actual effective throughput:      {actual_effective_throughput:.0f} B/s")

    print()
    print(f"Time ratio actual/expected:       {actual_time / expected_time:>7.2%}")
    print(f"Throughput ratio actual/expected: "
          f"{actual_effective_throughput / expected_effective_throughput:>7.2%}")

    # Validate destination file matches source file
    print()
    with open(args.src, "rb") as f_src, open(args.dst, "rb") as f_dst:
        src_data = f_src.read()
        dst_data = f_dst.read()
        if src_data != dst_data:
            print("ERROR: Destination file does not match source file!")
            exit(1)
        else:
            print("Destination file matches source file. Copy is correct.")
