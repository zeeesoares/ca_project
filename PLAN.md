# Project Planning and Objectives

## Contextualization

<<<<<<< HEAD
Large Language Models (LLMs) with billions of parameters require **weeks of
training on large-scale High-Performance Computing (HPC) systems**. Given the
significant computational cost and time involved, it is essential to preserve
the progress of training. This is achieved through **checkpointing**, a
mechanism that periodically saves the training state so that computation can be
resumed after interruptions or failures.
||||||| 0e6ae7b
We understanded that LLM's with billions of parameters require weeks of Training on massive HPC systems, and to prevent losing those weeks of work, the "state" must be saved in the process of Checkpointing.
=======
We understand that LLM's with billions of parameters require weeks of Training on massive HPC systems, and to prevent losing those weeks of work, the "state" must be saved in the process of Checkpointing.
>>>>>>> 281bf46fa79aed164fbdf9e8f8ec284287f7a626

<<<<<<< HEAD
Without checkpointing, a single failure during training could result in the loss
of days or even weeks of computation. Consequently, checkpointing has become a
critical component in large-scale machine learning workflows.
||||||| 0e6ae7b
### Motivation
- Resilience : As cluster size increases, the Mean Time Between Failures decreases. At scale, a failure can occur in shorts amounts of time.
- Checkpoint Bottleneck : With the increase of the number of parameters, the checkpoint size increases drastically, and the storage itself is a problem, even in an HPC context.
- PFS Contention : Writing those massive files in a shared and parallel file system creates a contention effect where the nodes compete for I/O bandwith, leading to significant stalls (training time).
=======
### Motivation
- Resilience : As cluster size increases, the Mean Time Between Failures decreases. At scale, a failure can occur in shorts amounts of time.
- Checkpoint Bottleneck : With the increase of the number of parameters, the checkpoint size increases drastically, and the storage itself is a problem, even in an HPC context.
- PFS Contention : Writing those massive files in a shared and parallel file system creates a contention effect where the nodes compete for I/O bandwidth, leading to significant stalls (training time).
>>>>>>> 281bf46fa79aed164fbdf9e8f8ec284287f7a626

<div style="page-break-after: always;"></div>

## Motivation

<<<<<<< HEAD
- **Resilience** \
    As HPC clusters grow in size, the **Mean Time Between Failures (MTBF)**
    decreases. At large scales, hardware or software failures can occur
    frequently, making reliable checkpointing essential to ensure training
    progress is not lost.
||||||| 0e6ae7b
- A. CheckFreq - Adaptative Frequency and Pipelined Checkpointing
    - How? - it uses an algorithmic approach to identify optimal checkpointing boundaries. It pipelines the checkpointing process with training iterations, dynamically adjusting frequency.
    - Problems? - it optimizes when to save, it still relies on the PFS for the final write, wich concern us beacause of the motivations.
=======
- A. CheckFreq - Adaptive Frequency and Pipelined Checkpointing
    - How? - it uses an algorithmic approach to identify optimal checkpointing boundaries. It pipelines the checkpointing process with training iterations, dynamically adjusting frequency.
    - Problems? - it optimizes when to save, it still relies on the PFS for the final write, which concern us because of the motivations.
>>>>>>> 281bf46fa79aed164fbdf9e8f8ec284287f7a626

- **Checkpoint Bottleneck** \
    With the increase of the number of parameters, the checkpoint size increases
    drastically, and the storage itself is a problem, even in an HPC context,
    increasing cost and time.

- **PFS Contention** \
    Writing those massive files in a shared and parallel file system creates a
    contention effect where the nodes compete for I/O bandwidth, leading to
    significant stalls (training time).

<<<<<<< HEAD
- **Storage Overhead** \
    Maintaining multiple checkpoints for fault tolerance and recovery requires
    significant storage capacity. This increases both infrastructure costs and
    the complexity of managing checkpoint data over long training runs.
||||||| 0e6ae7b
- D. BitSnap 
    - How? – It focuses on what is being saved rather than where. It introduces Sparsification and Quantization to the checkpointing process. It identifies which weights are "critical" and compresses the rest (e.g., from 16-bit to 4-bit or less) before the data even reaches the I/O layer.
=======
- D. BitSnap
    - How? – It focuses on what is being saved rather than where. It introduces Sparsification and Quantization to the checkpointing process. It identifies which weights are "critical" and compresses the rest (e.g., from 16-bit to 4-bit or less) before the data even reaches the I/O layer.
>>>>>>> 281bf46fa79aed164fbdf9e8f8ec284287f7a626

<<<<<<< HEAD
<div style="page-break-after: always;"></div>
||||||| 0e6ae7b
    - Problems? – While it significantly reduces the size of the files, it introduces a trade-off between compression time (CPU overhead) and accuracy. If the quantization is too aggressive, the model might not converge correctly after a resume. Also, it does not manage the storage hierarchy (PFS congestion) by itself.

- E. FastPersist 
    - How? – It exploits the tensor immutability during the Forward and Backward passes. While the GPU is busy computing gradients, FastPersist proactively moves the "stale" model state from GPU memory to the Host RAM and Local SSD in the background. It also uses a distributed "sharding" approach to parallelize writes across nodes.

    - Problems? – It requires a carefully managed memory buffer on the Host (RAM) to avoid interfering with the training process. While it hides I/O latency, it still ends up pushing the full data volume to the PFS eventually, which can still lead to long-term congestion if the model is massive and the frequency is high.

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
=======
    - Problems? – While it significantly reduces the size of the files, it introduces a trade-off between compression time (CPU overhead) and accuracy. If the quantization is too aggressive, the model might not converge correctly after a resume. Also, it does not manage the storage hierarchy (PFS congestion) by itself.

- E. FastPersist
    - How? – It exploits the tensor immutability during the Forward and Backward passes. While the GPU is busy computing gradients, FastPersist proactively moves the "stale" model state from GPU memory to the Host RAM and Local SSD in the background. It also uses a distributed "sharding" approach to parallelize writes across nodes.

    - Problems? – It requires a carefully managed memory buffer on the Host (RAM) to avoid interfering with the training process. While it hides I/O latency, it still ends up pushing the full data volume to the PFS eventually, which can still lead to long-term congestion if the model is massive and the frequency is high.

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
>>>>>>> 281bf46fa79aed164fbdf9e8f8ec284287f7a626

## Objectives

Develop a checkpointing system for ML/AI training that minimizes IO congestion,
checkpoint size and optimizes checkpointing time within the HPC environment.

The system will be designed to be used with the PyTorch ML framework and will be
compatible with the LUSTRE HPC storage system.

Modular design to allow for future extensions and allow optimizations
combinations, according to the use case and constraints.

<div style="page-break-after: always;"></div>

## Related Work

<!-- We can categorize the state-of-the-art into ?three? main technical approaches: -->

#### A. ***CheckFreq*** - Adaptive Frequency and Pipelined Checkpointing

- **How?** \
    It uses an algorithmic approach to identify optimal checkpointing
    boundaries. It pipelines the checkpointing process with training iterations,
    dynamically adjusting frequency.

- **Problems?** \
    It optimizes when to save, it still relies on the PFS for the final write,
    which concern us because of the motivations.

#### B. ***Check-N-Run***

- **How?** \
    Focuses on "hiding" the cost of I/O by using local node resources (DRAM/SSD)
    as a temporary staging area, allowing the training to resume immediately
    while the data is moved.

- **Problems?** \
    If a node fails before its local data is flushed to the PFS, that checkpoint
    is lost.

#### C. ***DeepFreeze*** (Similar to Monarch)

- **How?** \
    Uses a transparent storage tiering system. It uses high-speed local tiers
    (NVMe/PMEM) to accelerate the write path, abstracting the complexity from
    the user.

- **Problems?** \
    ...

<div style="page-break-after: always;"></div>

#### D. ***BitSnap***

- **How?** \
    It focuses on what is being saved rather than where. It introduces
    Sparsification and Quantization to the checkpointing process. It identifies
    which weights are "critical" and compresses the rest (e.g., from 16-bit to
    4-bit or less) before the data even reaches the I/O layer.

- **Problems?** \
    While it significantly reduces the size of the files, it introduces a
    trade-off between compression time (CPU overhead) and accuracy. If the
    quantization is too aggressive, the model might not converge correctly after
    a resume. Also, it does not manage the storage hierarchy (PFS congestion)
    by itself.

#### E. ***FastPersist***

- **How?** \
    It exploits the tensor immutability during the Forward and Backward passes.
    While the GPU is busy computing gradients, *FastPersist* proactively moves
    the "stale" model state from GPU memory to the Host RAM and Local SSD in the
    background. It also uses a distributed "sharding" approach to parallelize
    writes across nodes.

- **Problems?** \
    It requires a carefully managed memory buffer on the Host (RAM) to avoid
    interfering with the training process. While it hides I/O latency, it still
    ends up pushing the full data volume to the PFS eventually, which can still
    lead to long-term congestion if the model is massive and the frequency is
    high.

<div style="page-break-after: always;"></div>

## Proposed Solution

We propose a Python-based library that sits between the DL Framework
(PyTorch/DeepSpeed) and the storage layer, implementing a Multi-Tiered Reduction
Architecture.

#### Reducing Volume

- **Delta Checkpoint**: Only save the "diff" between the current state and the
    previous one.
- **Quantization and Compression**: Applying lossy or lossless compression to
    the optimizer states and weights before they leave the GPU/Node boundary.
- **Checkpoint Retention Policies**: Limit the number of stored checkpoints by
  keeping only the most recent *N* checkpoints. Older checkpoints are
  automatically removed or overwritten using a rolling checkpoint strategy.

#### Temporal/Frequency Control

- **Dynamic Checkpointing Frequency**: Inspired by *CheckFreq*, our library will
    monitor PFS I/O congestion. If the PFS is busy, it will increase the local
    checkpoint frequency.
- **Asynchronous Execution**: Ensuring the training loop never "waits" for the
    disk.
- **Overlap Training/Checkpointing**: ???

#### Hierarchical Storage Management (3 layers)

1. Distributed Node RAM (Maybe)
2. Node local SSD
3. PFS

<div style="page-break-after: always;"></div>

## Benchmark Policy

#### Environment

- Deucalion HPC Cluster (??? partition)

#### Tools

- [PyTorch Profiler](https://docs.pytorch.org/docs/main/profiler.html)
- [Flexible I/O tester](https://github.com/axboe/fio) (Maybe)
- Still researching...

#### Metrics

- Checkpointing time, size and frequency
- Training time, loss and accuracy
- IO congestion
- CPU/GPU usage
- Memory usage
- ...
