# *Advanced Computing Project* <br/> Storage system for optimizing LLM *checkpointing* in HPC

> Large Language Models (LLMs) training is a complex workload that can take
several weeks to complete. The complexity and long execution time are inherent
to their size, as they can easily reach billions of parameters that need to be
refined across training. Upon a failure (e.g., one or more nodes used in the
training process become unavailable), if these parameters are not saved, the
training process needs to be restarted from scratch. To address this challenge,
the model's state (i.e., activations, weights) is periodically *checkpointed* to
persistent storage, such as the Parallel File System (PFS) in HPC environments.
However, storing multiple checkpoints in a shared storage system can lead to
performance degradation, especially when the storage system is being used by
multiple jobs at the same time.
>
> The goal of this work is to propose a solution that improves *checkpointing*
performance of LLM training (without compromising correctness), while reducing
the I/O pressure at the PFS. The project should consider different fault models
(*i.e.*, failure of multiple nodes) and leverage the resources of all nodes
participating in the LLM training.
