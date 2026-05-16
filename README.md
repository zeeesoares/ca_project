# Storage system for optimizing LLM checkpointing in HPC <br> *Advanced Computing Project*

## Assignment

> **Advisors:**
Ricardo Macedo ([d12010@di.uminho.pt](mailto:d12010@di.uminho.pt));
João Paulo ([jtpaulo@di.uminho.pt](mailto:jtpaulo@di.uminho.pt]))
>
> Large Language Models (LLMs) training is a complex workload that can take
several weeks to complete. The complexity and long execution time are inherent
to their size, as they can easily reach billions of parameters that need to be
refined across training. Upon a failure (e.g., one or more nodes used in the
training process become unavailable), if these parameters are not saved, the
training process needs to be restarted from scratch. To address this challenge,
the model's state (i.e., activations, weights) is periodically checkpointed to
persistent storage, such as the Parallel File System (PFS) in HPC environments.
However, storing multiple checkpoints in a shared storage system can lead to
performance degradation, especially when the storage system is being used by
multiple jobs at the same time.
>
> The goal of this work is to propose a solution that improves checkpointing
performance of LLM training (without compromising correctness), while reducing
the I/O pressure at the PFS. The project should consider different fault models
(*i.e.*, failure of multiple nodes) and leverage the resources of all nodes
participating in the LLM training.
>
> **References**
> - Accelerating Deep Learning Training Through Transparent Storage Tiering.
    Dantas, M., Leitão, D., Cui, P., Macedo, R., Liu, X., Xu, W., & Paulo, J.
    In 22nd IEEE/ACM International Symposium on Cluster, Cloud and Internet
    Computing (CCGrid). IEEE, 2022.
> - Mohan, Jayashree, Amar Phanishayee, and Vijay Chidambaram. "{CheckFreq}:
    Frequent,{Fine-Grained}{DNN} Checkpointing." In 19th USENIX Conference on
    File and Storage Technologies (FAST 21), pp. 203-216. 2021.

## Structure

**TODO**

## Requirements

All required packages are listed in [requirements.txt](requirements.txt).

It's advised to use a virtual environment to manage dependencies.

```bash
python3 -m venv venv
source venv/bin/activate
```

Install them with pip:

```bash
pip install -r requirements.txt
```

> Python 3.12+ is supported.

## Usage

**TODO**

## Benchmarking

### E2E

**TODO**

### Token Bucket

```sh
RESULTS_DIR="benchmarks/token_bucket/results"
```

```sh
python3 -m benchmarks.token_bucket.benchmark \
    --src checkpoints/random_small.bin \
    --local-dir /tmp/token_bucket_local \
    --pfs-dir checkpoints/token_bucket_pfs \
    --results-file "$RESULTS_DIR/results.jsonl" \
    --metrics-dir "$RESULTS_DIR/per_run" \
    --dry-run  # Remove for actual benchmarking
```

```sh
python3 -m benchmarks.token_bucket.plot \
    --results-file "$RESULTS_DIR/results.jsonl" \
    --output-dir "$RESULTS_DIR/plots"
```

**TODO** update `--pfs-dir` flag
