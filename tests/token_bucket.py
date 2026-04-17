#!/usr/bin/env python3

import threading
import time
from migrater import token_bucket_copy

# Shared state (simulates orchestrator control)
current_rate = 5_000_000  # 5 MB/s


def get_rate():
    return current_rate


def set_rate(rate: int):
    global current_rate
    current_rate = rate


def orchestrator():
    global current_rate

    time.sleep(3)
    print("[orchestrator] Reducing throughput to 1 MB/s")
    current_rate = 1_000_000

    time.sleep(5)
    print("[orchestrator] Increasing throughput to 8 MB/s")
    current_rate = 8_000_000

    time.sleep(4)
    print("[orchestrator] Reducing throughput to 500 KB/s")
    current_rate = 500_000


if __name__ == "__main__":
    import argparse
    import os

    DEFAULT_CHUNK_SIZE = 64  # KB

    argparser = argparse.ArgumentParser(
        description="Test token bucket copy with dynamic throughput changes."
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

    set_rate(args.throughput)

    # Start orchestrator in background
    t = threading.Thread(target=orchestrator, daemon=True)
    t.start()

    print(f"Starting token bucket copy:\n"
          f"  - Source:             {args.src}\n"
          f"  - Destination:        {args.dst}\n"
          f"  - Initial throughput: {args.throughput} (B/s)\n"
          f"  - Chunk size:         {args.chunk_size} (B)")

    start = time.monotonic()

    # Run copy with dynamic throughput
    token_bucket_copy(args.src, args.dst, get_rate, args.chunk_size)

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
