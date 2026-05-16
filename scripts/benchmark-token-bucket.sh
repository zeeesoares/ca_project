#!/bin/bash

set -eux

module load "Python/3.12.3-GCCcore-13.3.0"
source venv/bin/activate

RESULTS_DIR="benchmarks/token_bucket/results"

LOCAL_DIR="/tmp/token_bucket_local"
PFS_DIR="/projects/checkpoints/token_bucket_pfs"  # TODO update PFS path

mkdir -p "$PFS_DIR"
mkdir -p "$LOCAL_DIR"

echo "Running token bucket benchmark on $(arch) partition..."

python3 -m benchmarks.token_bucket.benchmark \
    --src checkpoints/random_small.bin \
    --local-dir "$LOCAL_DIR" \
    --pfs-dir "$PFS_DIR" \
    --results-file "$RESULTS_DIR/results.jsonl" \
    --metrics-dir "$RESULTS_DIR/per_run"

echo "Generating plots..."

python3 -m benchmarks.token_bucket.plot \
    --results-file "$RESULTS_DIR/results.jsonl" \
    --output-dir "$RESULTS_DIR/plots"
