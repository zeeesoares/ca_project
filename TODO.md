# TODOs

- Experimentar modelos (deucalion/local)
    - Bert (Hugging Face)
- Criar ambiente de testes
- Instrumentação Python
- Dataloader custom from pytorch to measure files created, deleted, modified, etc.
- Use strace on token bucket (`strace -r --trace=openat,read,write,close ...`)
- Check (evaluate) min time sleep
- Add logs for benchmarking
- Explore token bucket configuration for better performance
- Deadline based policy: [Deadline QoSPolicy](https://community.rti.com/static/documentation/connext-dds/current/doc/manuals/connext_dds_professional/users_manual/users_manual/DEADLINE_QosPolicy.htm) with CP Frequency, consider how async torch.save affects periodicity
- analyze latest benchmark results

- plot token bucket (from migrater metrics; Zé graph and actual/efective)
- plot accumulated graph based on pca benchmark results
- 5AM benchmark
- alternate checkpoint path
- create `src` dir to clean root dir
- discover why `torch.save` is not saving last checkpoint when running asynchronously;
  also research `async` flag
- documentation (README, etc)
