# Project Planning and Objectives

## Contextualização e Motivo

TODO

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
