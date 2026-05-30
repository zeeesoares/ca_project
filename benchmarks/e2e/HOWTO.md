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

This runs a group of tests combining different policies and workloads.

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

After the jobs complete, the results are stored under the results directory.

For orchestrated runs, each experiment directory contains:

- JSONL result files written by the workers.
- Migrater logs under `migrater_logs/`.
- Optional generated plots under `plots/`.

Use the collect module to aggregate JSONL results into CSV files.

```bash
PFS_DIR=$(pwd)  # /projects/example

python3 -m benchmarks.e2e.collect \
    --results-dir $PFS_DIR/results \
    --output-csv $PFS_DIR/results/summary.csv \
    --detail-csv $PFS_DIR/results/detail.csv
```

To collect one specific experiment:

```bash
PFS_DIR=$(pwd)  # /projects/example

python3 -m benchmarks.e2e.collect \
    --results-dir $PFS_DIR/results/steady_age-priority \
    --output-csv $PFS_DIR/results/steady_age-priority/summary.csv \
    --detail-csv $PFS_DIR/results/steady_age-priority/detail.csv
```

## Plotting Results

To generate visual comparisons such as Stall Time, Throughput, and Summary
Tables, use the plot module.

```bash
python3 -m benchmarks.e2e.plot \
    --results-dir $PFS_DIR/results \
    --output-dir $PFS_DIR/results/plots
```

## Accumulated Bandwidth Graphs

Accumulated bandwidth graphs are now generated through the collect module.

The workers store migrater logs automatically under each experiment directory:

```text
$PFS_DIR/results/<experiment_tag>/migrater_logs/
```

To aggregate results and generate accumulated bandwidth graphs for all
experiments:

```bash
python3 -m benchmarks.e2e.collect \
    --results-dir $PFS_DIR/results \
    --output-csv $PFS_DIR/results/summary.csv \
    --detail-csv $PFS_DIR/results/detail.csv \
    --accumulated-bw-plots \
    --plots-dir $PFS_DIR/results/plots
```

This produces one accumulated bandwidth graph per experiment:

```bash
$PFS_DIR/results/plots/<experiment_tag>/accumulated_bandwidth.png
```

For a single experiment:

```bash
python3 -m benchmarks.e2e.collect \
    --results-dir $PFS_DIR/results/steady_age-priority \
    --output-csv $PFS_DIR/results/steady_age-priority/summary.csv \
    --detail-csv $PFS_DIR/results/steady_age-priority/detail.csv \
    --accumulated-bw-plots \
    --plots-dir $PFS_DIR/results/steady_age-priority/plots
```

This produces:

```text
$PFS_DIR/results/steady_age-priority/plots/steady_age-priority/accumulated_bandwidth.png
```

Manual execution of `plot_accumulated_bw.py` is no longer needed for the normal
workflow.

## Directory Structure

- `benchmarks/e2e/`: Python logic for workers, collection, and plotting.
- `scripts/`: SLURM submission scripts and helper scripts.
- `results/`: Default output folder for benchmark outputs.
- `results/<experiment_tag>/`: Output directory for one experiment.
- `results/<experiment_tag>/*.jsonl`: Per-worker checkpoint result records.
- `results/<experiment_tag>/migrater_logs/`: Per-worker migrater logs used for accumulated bandwidth graphs.
- `results/<experiment_tag>/summary.csv`: Aggregated experiment summary.
- `results/<experiment_tag>/detail.csv`: Per-worker detail summary.
- `results/plots/<experiment_tag>/accumulated_bandwidth.png`: Accumulated bandwidth graph for one experiment.
