# Metodologia de Benchmark e Teste de Checkpointing (Inspirada no PADLL)

---

## Métrica

- Checkpoint Migration Throughput: Taxa de transferência real (MiB/s ou GiB/s) medida
    durante a migração do SSD para o PFS.
- Training Stall Time (Overhead no Treino): É o tempo que o processo de treino fica 
    bloqueado a guardar o checkpoint.  
- Migration Latency: Tempo total desde que o treino acaba de escrever no SSD até o 
    ficheiro estar disponível no PFS. ???

---

## Cenários de Teste
### Baseline vs. Arquitetura Proposta

- Objetivo: Validar o isolamento do treino face ao ruído de I/O no PFS.  
- Procedimento: Comparar o treino direto no PFS (Lustre) com a abordagem orquestrada.  
- Análise: Utilizar o torch.profiler para identificar a variância no tempo de paragem.
    No cenário orquestrado, o tempo de escrita deve ser independente da carga de rede.  

### Avaliação de Escalabilidade do Plano de Controlo

- Objetivo: Determinar se o Orquestrador constitui um gargalo centralizado.  
- Procedimento: Incrementar progressivamente o número de workers ativos (ex: 2, 4, 8, 16) 
    no Deucalion.  
- Análise: Medir a latência de decisão no orchestrator_metrics.json para verificar se o 
    tempo de resposta se mantém estável.  

### Stress Test e Saturação de QoS

- Objetivo: Validar o comportamento do sistema sob condições extremas de contenção.  
- Procedimento: Configurar o Orquestrador com uma largura de banda global reduzida 
    (ex: 10 MB/s) e forçar checkpoints frequentes em todos os workers.  
- Análise: Observar a emissão de instruções HOLD e a redução dinâmica das taxas 
    (CHANGE_RATE) para evitar a saturação do PFS.

---

## Metodologia de Execução

1. Arranque do Orquestrador: Iniciar o orchestrator.server num nó dedicado com a 
    política e largura de banda desejada.  
2. Submissão em Massa: Usar o teu script launch_experiments.sh para disparar N workers 
    (ex: 2, 4, 8) simultaneamente.  
3. Execução do Nó: Cada job de treino deve ativar o seu próprio migrater.server em 
    background antes de iniciar o loop de treino do PyTorch.  
    1. Verificar:
        - O Orquestrador gera o log central orchestrator_metrics.json com as decisões de QoS.  
        - O Treino gera os logs do torch.profiler para medir os tempos de paragem.  
        - O Migrater regista o throughput efetivo de cada transferência.  
4. Análise: Cruzar os timestamps de todos os logs para criar gráficos de "Throughput vs Tempo"