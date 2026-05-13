# Running Benchmark

## Baseline

The baseline simulates training jobs writing checkpoints directly to the Parallel File System without any orchestration.

```bash
# Example: 4 workers, 3 checkpoints each
./scripts/submit-baseline.sh --n-workers 4 --n-checkpoints 3
```

## Orchestrated

This runs the system with the Orchestrator managing data transfers based on a specific policy.

```bash
for POLICY in uniform-fair-share active-fair-share age-priority epoch-priority; 
do ./scripts/submit-orchestrated.sh --policy $POLICY --pfs-bw 2GB --n-workers 4; done
```

This runs a group of test combining different policies and workloads.

```bash
for POLICY in uniform-fair-share active-fair-share age-priority epoch-priority; do
    ./scripts/submit-orchestrated.sh \
        --policy "$POLICY" \
        --pfs-bw 2GB  \
        --n-workers 4 \
        --compute-time 0 \
        --experiment-tag "burst_${POLICY}"

    ./scripts/submit-orchestrated.sh \
        --policy "$POLICY" \
        --pfs-bw 2GB  \
        --n-workers 4 \
        --compute-time 30 \
        --experiment-tag "steady_${POLICY}"

    ./scripts/submit-orchestrated.sh \
        --policy "$POLICY" \
        --pfs-bw 2GB  \
        --n-workers 4 \
        --compute-time "random" \
        --use-jitter \
        --experiment-tag "dynamic_${POLICY}"
done
```

### Note

All of the results are written in a `ca_bench/` folder.

## Data Collection

After the jobs complete, the results are stored as .jsonl files in the results directory. Use the collect module to aggregate these into a summary CSV.

```bash
python3 -m e2e.collect \\
    --results-dir ca_bench \\
    --output-csv ca_bench
```

## Plotting Results

To generate visual comparisons (Stall Time, Throughput, and Summary Tables), use the plot module.


```bash
python3 -m e2e.plot \\
    --results-dir ./results \\
    --output-dir benchmarks/e2e/results
```


## Directory Structure

- e2e/: Contains the Python logic for workers, collection, and plotting.

- scripts/: Shell scripts for SLURM submission and environment preparation.

- results/: Default output folder for raw JSONL logs.