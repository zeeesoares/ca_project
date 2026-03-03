# *Advanced Computing Project*

## Storage system for optimizing LLM checkpointing in HPC

### Assignment

> **Advisors:** Ricardo Macedo ([d12010@di.uminho.pt](mailto:d12010@di.uminho.pt));
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

### Structure

```
.
├── checkpointing
│   ├── __init__.py
│   ├── interface.py                 # Checkpointing interface definition (save / load)
│   ├── decorator.py                 # Base decorator for wrapping checkpoint managers
│   ├── torch_manager.py             # Core / baseline PyTorch checkpointing
│   ├── async_wrapper.py             # Asynchronous / parallel checkpointing
│   ├── compression_wrapper.py       # FP16 compression checkpointing (TODO INT8 quantization)
│   ├── incremental_wrapper.py       # Delta / incremental checkpointing (TODO)
│   ├── sharded_wrapper.py           # Distributed / sharded checkpointing (TODO)
│   └── utils.py                     # Utility functions (list checkpoints, etc.)
├── checkpoints
│   └── checkpoint_N.pt              # Example checkpoint file (N = step number)
├── train
│   ├── __init__.py
│   ├── dataset.py                   # Data loader for WikiText-2
│   ├── model.py                     # Builds Google's BERT MLM model
│   ├── trainer.py                   # Trainer class for running experiments
│   └── train.py                     # Main experiment script
├── README.md
├── requirements.txt
├── setup.sh                         # Setup virtual environment and install dependencies
└── run-local.sh                     # Local running script (TODO HPC SLURM jobs)
```

### Dependencies

All required packages are listed in [requirements.txt](requirements.txt).

It's advised to use a virtual environment (e.g., `venv` or `conda`) to manage
dependencies.

```bash
python3 -m venv venv
source venv/bin/activate
```

Install them with pip:

```bash
pip install -r requirements.txt
```

> Python 3.12+ is supported.

### Usage

#### Python API (Composing Checkpoint Wrappers)

Checkpointing wrappers can be composed as follows:

```python
from checkpointing.torch_manager       import TorchCheckpoint
from checkpointing.async_wrapper       import AsyncCheckpointWrapper
from checkpointing.compression_wrapper import CompressionCheckpointWrapper

# Base checkpoint manager
checkpoint = TorchCheckpoint("./checkpoints")

# Apply parameter compression wrapper
checkpoint = CompressionCheckpointWrapper(checkpoint)

# Apply asynchronous wrapper (non-blocking saves)
checkpoint = AsyncCheckpointWrapper(checkpoint)
```

Wrappers are **composable in any order**, allowing flexible experimentation.

TODO add `checkpoint.save` and `checkpoint.load` examples.

#### Running Training Experiments

From the project root:

```bash
python3 -m train.train --enable-async --enable-compression
```

#### CLI Flags

Flag                      | Description
--------------------------|----------------------------------------------------
`--enable-async`          | Enable asynchronous checkpointing
`--enable-compression`    | Enable FP16 (TODO INT8) compression for checkpoints
`--checkpoint-interval N` | TODO Save a checkpoint every N training steps
`...`                     | TODO

### Benchmarking

TODO
