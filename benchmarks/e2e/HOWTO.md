# Running Benchmark

## Baseline

The baseline simulates training jobs writing checkpoints directly to the PFS
without any orchestration.

```bash
PFS_DIR=$(pwd)  # /projects/example

# Example: 4 workers, 3 checkpoints each
./scripts/submit-baseline.sh \
    --n-workers 4 \
    --n-checkpoints 3 \
    --pfs-dir $PFS_DIR/checkpoints \
    --results-dir $PFS_DIR/results
```

## Orchestrated

This runs the system with the Orchestrator managing data transfers based on a
specific policy.

```bash
PFS_DIR=$(pwd)  # /projects/example

POLICIES=(uniform-fair-share active-fair-share age-priority epoch-priority)

for POLICY in "${POLICIES[@]}"; do
    ./scripts/submit-orchestrated.sh \
        --policy $POLICY \
        --pfs-bw 2GB \
        --n-workers 4 \
        --pfs-dir $PFS_DIR/checkpoints \
        --results-dir $PFS_DIR/results
done
```

This runs a group of test combining different policies and workloads.

```bash
PFS_DIR=$(pwd)  # /projects/example

POLICIES=(uniform-fair-share active-fair-share age-priority epoch-priority)

for POLICY in "${POLICIES[@]}"; do
    ./scripts/submit-orchestrated.sh \
        --policy "$POLICY" \
        --pfs-bw 2GB  \
        --n-workers 4 \
        --compute-time 0 \
        --experiment-tag "burst_${POLICY}" \
        --pfs-dir $PFS_DIR/checkpoints \
        --results-dir $PFS_DIR/results
done

for POLICY in "${POLICIES[@]}"; do
    ./scripts/submit-orchestrated.sh \
        --policy "$POLICY" \
        --pfs-bw 2GB  \
        --n-workers 4 \
        --compute-time 30 \
        --experiment-tag "steady_${POLICY}" \
        --pfs-dir $PFS_DIR/checkpoints \
        --results-dir $PFS_DIR/results
done

for POLICY in "${POLICIES[@]}"; do
    ./scripts/submit-orchestrated.sh \
        --policy "$POLICY" \
        --pfs-bw 2GB  \
        --n-workers 4 \
        --compute-time "random" \
        --use-jitter \
        --experiment-tag "dynamic_${POLICY}" \
        --pfs-dir $PFS_DIR/checkpoints \
        --results-dir $PFS_DIR/results
done
```

## Data Collection

After the jobs complete, the results are stored as JSONL files in the results
directory. Use the collect module to aggregate these into a summary CSV.

```bash
python3 -m benchmarks.e2e.collect \
    --results-dir $PFS_DIR/results \
    --output-csv $PFS_DIR/results/summary.csv
```

## Plotting Results

To generate visual comparisons (Stall Time, Throughput, and Summary Tables),
use the plot module.

```bash
python3 -m benchmarks.e2e.plot \
    --results-dir $PFS_DIR/results \
    --output-dir $PFS_DIR/results/plots
```

## Directory Structure

- e2e/: Contains the Python logic for workers, collection, and plotting.
- scripts/: Shell scripts for SLURM submission and environment preparation.
- results/: Default output folder for raw JSONL logs.
