#!/usr/bin/env python3

import time
import os
from typing import Callable, Union

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
    tokens = rate
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


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(
        description="Copy a file using token bucket rate limiting."
    )
    argparser.add_argument("src", type=str,
                           help="Source file path (local node storage)")
    argparser.add_argument("dst", type=str,
                           help="Destination file path (PFS)")
    argparser.add_argument("throughput", type=int,
                           help="Transfer rate in bytes per second")
    argparser.add_argument("--chunk-size", type=int,
                           default=DEFAULT_CHUNK_SIZE * 1024,
                           help="Size of each read/write chunk "
                                f"(default: {DEFAULT_CHUNK_SIZE}KB)")
    args = argparser.parse_args()

    print(f"Starting token bucket copy:\n"
          f"  - Source:      {args.src}\n"
          f"  - Destination: {args.dst}\n"
          f"  - Throughput:  {args.throughput} (B/s)\n"
          f"  - Chunk size:  {args.chunk_size} (B)\n"
          f"  - Sleep cap:   {SLEEP_CAP * 100:.0f}% of calculated sleep time")

    start = time.monotonic()

    token_bucket_copy(args.src, args.dst, args.throughput, args.chunk_size)

    end = time.monotonic()

    print()
    print("Token bucket copy completed.")

    file_size = os.path.getsize(args.src)
    effective_throughput = file_size / (end - start)
    print()
    print(f"Expected throughput:  {args.throughput} (B/s)")
    print(f"Effective throughput: {effective_throughput:.0f} (B/s)")

    print()
    print(f"Expected time: {file_size / args.throughput:.2f} seconds")
    print(f"Actual time:   {end - start:.2f} seconds")

    print()
    print("Ratio (actual/expected): "
          f"{effective_throughput / args.throughput:.2%}")
