import time
import os


def token_bucket_copy(
    src: str,
    dst: str,
    throughput: int,
    chunk_size: int = 8192
):
    """
    Copy a file using token bucket rate limiting.

    Parameters:
        src (str): Source file path (local node storage)
        dst (str): Destination file path (PFS)
        throttle (int): Transfer rate in bytes per second
        chunk_size (int): Size of each read/write chunk (default: 8KB)

    Raises:
        IOError: If there is an error reading from the source or writing
            to the destination.

    TODO:
        - Handle exceptions more gracefully, possibly with retries or logging,
            and update docstring Exceptions section.
        - Dynamically adjust the chunk size based on the observed throughput
            and latency.
        - Dynamic adjustment of throughput based on orchestrator orders.
    """

    capacity = throughput  # set as chunk_size to make this a leaky bucket
    tokens = throughput
    last = time.monotonic()

    # os.makedirs(os.path.dirname(dst), exist_ok=True)

    with open(src, "rb") as src_f, open(dst, "wb") as dst_f:
        while True:
            now = time.monotonic()
            elapsed = now - last
            last = now

            tokens = min(capacity, tokens + elapsed * throughput)

            if tokens < chunk_size:
                # time.sleep(0.005)  # Constant sleep time
                # Dynamic sleep time
                missing = chunk_size - tokens
                time.sleep(missing / throughput)
                continue

            data = src_f.read(chunk_size)
            if not data:
                break

            dst_f.write(data)
            tokens -= len(data)

        # TODO At the end of copy, should we flush dst on the PFS?
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
    argparser.add_argument("--chunk-size", type=int, default=8192,
                           help="Size of each read/write chunk (default: 8KB)")
    args = argparser.parse_args()

    token_bucket_copy(args.src, args.dst, args.throughput, args.chunk_size)
