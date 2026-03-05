# Project Planning and Objectives

## Contextualization and Motivation

We understanded that LLM's with billions of parameters require weeks of Training on massive HPC systems, and to prevent losing those weeks of work, the "state" must be saved in the process of Checkpointing.

### Motivation
- Resilience : As cluster size increases, the Mean Time Between Failures decreases. At scale, a failure can occur in shorts amounts of time.
- Checkpoint Bottleneck : With the increase of the number of parameters, the checkpoint size increases drastically, and the storage itself is a problem, even in an HPC context.
- PFS Contention : Writing those massive files in a shared and parallel file system creates a contention effect where the nodes compete for I/O bandwith, leading to significant stalls (training time).

## Related Work

We can categorize the state-of-the-art into three main technical approaches:

- A. CheckFreq - Adaptative Frequency and Pipelined Checkpointing
    - How? - it uses an algorithmic approach to identify optimal checkpointing boundaries. It pipelines the checkpointing process with training iterations, dynamically adjusting frequency.
    - Problems? - it optimizes when to save, it still relies on the PFS for the final write, wich concern us beacause of the motivations.

- B. Check-N-Run
    - How? - focuses on "hiding" the cost of I/O by using local node resources (DRAM/SSD) as a temporary staging area, allowing the training to resume immediately while the data is moved.
    - Problems? - If a node fails before its local data is flushed to the PFS, that checkpoint is lost.

- C. DeepFreeze (Similar to Monarch)
    - How? - uses a transparent storage tiering system. It uses high-speed local tiers (NVMe/PMEM) to accelerate the write path, abstracting the complexity from the user.
    - Problems? - ...

- D. BitSnap 
    - How? – It focuses on what is being saved rather than where. It introduces Sparsification and Quantization to the checkpointing process. It identifies which weights are "critical" and compresses the rest (e.g., from 16-bit to 4-bit or less) before the data even reaches the I/O layer.

    - Problems? – While it significantly reduces the size of the files, it introduces a trade-off between compression time (CPU overhead) and accuracy. If the quantization is too aggressive, the model might not converge correctly after a resume. Also, it does not manage the storage hierarchy (PFS congestion) by itself.

E. FastPersist 

    How? – It exploits the tensor immutability during the Forward and Backward passes. While the GPU is busy computing gradients, FastPersist proactively moves the "stale" model state from GPU memory to the Host RAM and Local SSD in the background. It also uses a distributed "sharding" approach to parallelize writes across nodes.

    Problems? – It requires a carefully managed memory buffer on the Host (RAM) to avoid interfering with the training process. While it hides I/O latency, it still ends up pushing the full data volume to the PFS eventually, which can still lead to long-term congestion if the model is massive and the frequency is high.

## Proposed Solution

We propose a Python-based library that sits between the DL Framework (PyTorch/DeepSpeed) and the storage layer, implementing a Multi-Tiered Reduction Architecture.

- A. Reducing Volume
    - Delta Checkpoint: Only save the "diff" between the current state and the previous one.
    - Quantization and Compression: Applying lossy or lossless compression to the optimizer states and weights before they leave the GPU/Node boundary.

- B. Temporal/Frequency Control
    - Dynamic Checkpointing Frequency: Inspired by CheckFreq, our library will monitor PFS I/O congestion. If the PFS is busy, it will increase the local checkpoint   frequency.
    - Asynchronous Execution: Ensuring the training loop never "waits" for the disk.
    - Overlap Training/Checkpointing  - ???

- C. Hierarchical Storage Management
    - L1 - Distributed Node RAM?
    - L2 - Node-local SSD
    - L3 - PFS

## Objectives

Develop a checkpointing system for ML/AI training that minimizes IO congestion,
checkpoint size and optimizes checkpointing time within the HPC environment.

The system will be designed to be used with the PyTorch ML framework and will be
compatible with the LUSTRE HPC storage system.

Modular design to allow for future extensions and allow optimizations combinations,
according to the use case and constraints.

## Optimizations

- **Asynchronous Checkpointing**: ...
- **Compression**: ...
- **Quantization**: ...
- **Delta Checkpointing**: ...
- **Hierarchical Storage System**: Implement a hierarchical storage system to
optimize checkpointing. Node(s) RAM > Node(s) local storage > PFS.
- **Checkpoint Dynamic Frequency**: Determine optimal checkpointing frequency
based on IO constraints minimizing IO congestion. With an hierarchical storage
system the PFS usage can be reduced on congestion periods.
- **Minimize Storage**: Keeping the last N checkpoints ...

## Benchmarks

### Tools

- [PyTorch Profiler](https://docs.pytorch.org/docs/main/profiler.html)
- [Flexible I/O tester](https://github.com/axboe/fio) (Maybe)

### Metrics

- Checkpointing time, size and frequency
- Training time, loss and accuracy
- IO congestion
- CPU/GPU utilization
- Memory usage
- ...
