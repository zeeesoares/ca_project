# Políticas de Escalonamento

## Rate Limiting (Token Bucket) (TO-DO)

O orquestrador define um limite máximo de tokens de bandwitdh que cade job de treino pode submeter ao PFS. Evitando assim que um job com checkpoints muito frequentes sature todo o sistema.

Source: Macedo, R., et al. "PADLL: Taming Metadata-intensive HPC Jobs Through Dynamic, Application-agnostic QoS Control." CCGrid 2023. (arXiv: 2302.06418) — GitHub: github.com/dsrhaslab/padll

## Uniform Fair-Share (Implemented)

Avaliar diferentes setups de QoS, incluindo uma política "Uniform" em que a taxa máxima de operações é dividida igualmente por todos os jobs. No caso, 4 jobs a fazer checkpoing = 25% da bandwith para cada para o PFS. Muito simples, mas pode ser ineficiente, pouco escalável e em alguns casos injusto.

Source: Source: Macedo, R., et al. "PADLL: Taming Metadata-intensive HPC Jobs Through Dynamic, Application-agnostic QoS Control." CCGrid 2023. (arXiv: 2302.06418)

## Priority-Based Scheduling (TO-DO)

Jobs com maior prioridade recebem maior bandwith para o PFS. Podia ser utilizado para dar maior prioridade a jobs que estejam perto de terminar o treino ou a jobs cujo último checkpoint é o mais antigo por exemplo.

Source: Source: Macedo, R., et al. "PADLL: Taming Metadata-intensive HPC Jobs Through Dynamic, Application-agnostic QoS Control." CCGrid 2023. (arXiv: 2302.06418)

## Staggered / Coordinated Checkpoint Scheduling

Coordenação de atividades de I/O entre aplicações, sobrepondo os ciclos de I/O de uma aplicação com os ciclos de computação de outras, para que não acedam ao PFS ao mesmo tempo.
O orquestrador escalona processos ao mesmo tempo, mas em vez dos jobs fazerem todos o checkpoint ao mesmo tempo, distribui-os por janelas temporais.

Source: Dongarra, J. et al. "Checkpointing Orchestration: Toward a Scalable HPC Fault-Tolerant Environment." CCGrid 2012. (IEEE Xplore: 10.1109/CCGrid.2012.66)
Source adicional: LLNL I/O Scheduling Research — computing.llnl.gov/projects/scalable-checkpoint-restart-for-mpi/io-scheduling-research

## Two-Tier Storage (SSD → PFS)

A abordagem de checkpoints em 2 níveis. Prevê fast-tier checkpoints armazenados no SSD e durable-tier checkpoints menos frequentes armazenados em storage do PFS. O orchestrator gere esta migração: o cliente escreveria para o SSD local, e o orchestrador faz flush async para o PFS quando a contenção é baixa.

Source: AWS Blog — "Architecting scalable checkpoint storage for large-scale ML training on AWS" (2025).
Source: Gupta, Y. et al. "FastPersist: Accelerating Model Checkpointing in Deep Learning." arXiv:2406.13768 (2024)

## Convergence-Aware / Adaptive Frequency

Tenta abordar o facto de um progresso de um job não é linear. As primeiras epochs costumam ser mais valiosas(Nas primeiras épocas do treino, a loss function desce rapidamente no fim do treino o modelo já está perto da convergência), com isto intervalos uniformes de checkpoint podem desperdiçar recursos nas fases finais ou sub-proteger as iniciais. Sendo assim o orchestrator adapta a frequência do checkpoint com base no treino (tempo a que está a decorrer, variação da loss function, etc....)

Source: Li, H. et al. "Convergence-aware optimal checkpointing for exploratory deep learning training jobs." Future Generation Computer Systems (2024). (ScienceDirect)

## I/O Burst Prediction

Usar modelos ML para prever I/O bursts com uma accuracy bastante alta, estas previsões podem ser usadas então para adiar operações I/O tolerantes a atraso, como é o caso do checkpointing. O orchestrador pode monitorizar padrões I/O e adiar checkpoints quando prevê contenção elevada no PFS

Source: Saeedizade, E., Barbosa, J., & Dede, E. "I/O Burst Prediction for HPC Clusters using Darshan Logs." arXiv:2308.10311, 2023.

## Adaptive Checkpoint Scheduling com Online Profiling

O orchestrator não usa intervalos de checkpoint fixos, mas sim aprende e adapta dinamicamente quando e com que frequência cada job deve fazer o checkpoint, usando 2 fontes de informação.

    - Do lado do treino (training-aware): O orchestrator recebe métricas do cliente — a loss atual, a taxa de variação da loss (gradiente da curva de convergência), e o tempo por iteração. Com o CheckFreq (Microsoft Research), é possível determinar automaticamente a frequência de checkpointing ao nível de interações individuais usando profiling online, e ajustar dinamicamente essa frequência em runtime para manter o overhead abaixo de um limiar. O orchestrator quando a loss estiver a descer rapidamente, o orchestrator pede checkpoints mais frequentes e quando estabiliza reduz as suas frequências.

    - Do lado do storage (I/O aware): Em paralelo, o orchestrator monitoriza a carga do PFS. Modelos de ML tipo XGBoost, treinados com logs de I/O (por exemplo do Darshan), conseguem prever bursts de I/O. O orchestrator pode usar um modelo simples para estimar se o PFS está ou vai ficar congestionado, e adiar o checkpoint de um job para um momento mais calmo.

    Source: Mohan, J., Phanishayee, A., Chidambaram, V. "CheckFreq: Frequent, Fine-Grained DNN Checkpointing." FAST '21, USENIX, pp. 203-216, 2021. — GitHub: github.com/msr-fiddle/CheckFreq

    Source: Saeedizade, E., Barbosa, J., Dede, E. "I/O Burst Prediction for HPC Clusters using Darshan Logs." arXiv:2308.10311, 2023.

    Source: Li, H. et al. "Convergence-aware optimal checkpointing for exploratory deep learning training jobs." Future Generation Computer Systems, 2024. (ScienceDirect)

    Source: Macedo, R., et al. "PADLL: Taming Metadata-intensive HPC Jobs Through Dynamic, Application-agnostic QoS Control." CCGrid 2023. (arXiv: 2302.06418)