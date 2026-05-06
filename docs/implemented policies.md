# Políticas de Escalonamento — Implementadas

Documenta as políticas implementadas em
[orchestrator/scheduler.py](../orchestrator/scheduler.py)

---

## Modelo comum

Cada política implementa o `Strategy Pattern` definido por `SchedulerPolicy`:

```python
class SchedulerPolicy(ABC):
    def decide(self, worker_id: str,
               workers: Dict[str, WorkerState]) -> InstructionResponse: ...
```

A cada heartbeat de um worker, o orquestrador atualiza o `ClusterState`
partilhado e chama `policy.decide(worker_id, snapshot)`. A
resposta é uma de:

- `START_FLUSH` com `rate_limit_bps` — o worker deve transferir, ao
  rate indicado.
- `HOLD` com `rate=0` — o worker não deve iniciar transferência (uma
  transferência já em curso continua até ao fim).
- `CHANGE_RATE` — emitida implicitamente quando uma policy devolve um novo
  `rate_limit_bps` para um worker já em flush; o token bucket no migrater lê
  a rate em runtime e adapta-se sem reiniciar a cópia.

`WorkerState` carrega tudo o que as políticas precisam de saber sobre um
worker:

| Campo             | Origem                                            | Usado por             |
| ----------------- | ------------------------------------------------- | --------------------- |
| `worker_id`       | `socket.gethostname()` no migrater                | static-priority       |
| `checkpoint_size` | bytes pendentes de flush                          | active, static, age   |
| `is_migrating`    | flag do migrater                                  | (ainda não usado)     |
| `last_seen`       | `time.time()` na update                           |                       |
| `pending_since`   | `time.time()` quando passou de `0` para `>0`      | age-priority          |
| `epoch`           | progresso do treino reportado pelo migrater       | epoch-priority        |
| `total_epochs`    | total de épocas reportado pelo migrater           | epoch-priority        |

Um worker é considerado **ativo** se `checkpoint_size > 0`. Workers inativos
recebem `HOLD` em qualquer política que respeite atividade.

---

## NoLimit (`no-limit`)

Baseline. Devolve sempre `START_FLUSH` com `rate=1 GiB/s`, ignorando o
estado do cluster.

```bash
python -m orchestrator.server --policy no-limit
```

Sem flags próprias.

---

## FixedRate (`fixed-rate`)

Cada worker recebe a mesma rate fixa, independentemente de quantos workers
existem ou estão ativos.

```
rate_i = constant (default: 1 GiB/s)
```

```bash
python -m orchestrator.server --policy fixed-rate --pfs-bw 100MB
```

---

## UniformFairShare (`uniform-fair-share`)

Divide a banda total do PFS igualmente por **todos** os workers
registados, ativos ou não.

```
rate_i = pfs_bw / |workers|
```

```bash
python -m orchestrator.server --policy uniform-fair-share --pfs-bw 1GB
```

---

## ActiveFairShare (`active-fair-share`)

Divide a banda só pelos workers **ativos**;
inativos recebem `HOLD`.

```
rate_i = pfs_bw / |active|     (se i ∈ active)
HOLD                           (caso contrário)
```

```bash
python -m orchestrator.server --policy active-fair-share --pfs-bw 1GB
```

---

## StaticPriority (`static-priority`)

Fair-share **ponderado** com pesos fixos por worker, configurados via
ficheiro JSON externo. Workers ausentes do mapa caem num `default_priority`.
Pesos não-positivos são forçados a `default_priority` (defensivo).

```
weight(w)   = priorities.get(w.worker_id, default_priority)
total       = Σ weight(w)  para w ∈ active
rate_i      = pfs_bw * weight(i) / total

inativo     → HOLD
```

**Schema do JSON** (ver exemplo em
[orchestrator/priorities.json](../orchestrator/priorities.json)):

```json
{
  "default": 1.0,
  "workers": {
    "node01": 3.0,
    "node02": 2.0,
    "node03": 1.0
  }
}
```

**Casos particulares:**

- JSON ausente (`--priority-map` não passado) → `priorities={}`,
  `default=1.0`. Comportamento idêntico a `active-fair-share`.
- Worker ausente do mapa mas presente no cluster → recebe `default_priority`.

```bash
python -m orchestrator.server \
    --policy static-priority \
    --priority-map orchestrator/priorities.json \
    --pfs-bw 100MB
```

Testes:
[tests/static_priority.py](../tests/static_priority.py).

---

## AgePriority (`age-priority`)

Política **dinâmica**: combina dois sinais observáveis no orquestrador para
calcular a prioridade em runtime.

- **Tamanho do checkpoint pendente** — quem tem mais bytes pendentes
  acaba mais rápido se receber mais bandwith, libertando o sistema mais rápido.
- **Idade do pending** — quanto tempo o worker está com pending > 0.

A normalização é feita pelo máximo entre os ativos, garantindo que ambos
os sinais ficam em `[0, 1]` independentemente das ordens de grandeza
(bytes vs segundos).

```
size_norm_i = checkpoint_size_i / max_size                  (entre ativos)
age_i       = now - pending_since_i
age_norm_i  = age_i / max_age                               (0 se max_age == 0)
priority_i  = α · size_norm_i + β · age_norm_i
priority_i  = max(priority_i, EPS)                          (EPS = 1e-6)

rate_i      = pfs_bw · priority_i / Σ priority_j            (para j ∈ active)

inativo     → HOLD
```

**Tracking de `pending_since`:** `ClusterState.update()` regista
`time.time()` no instante da transição `0 → >0`, preserva-o em updates
estáveis (`>0 → >0`), e limpa-o no regresso a inativo (`>0 → 0`).

**Casos particulares e flags extremos:**

| `(α, β)`     | Comportamento                           |
| ------------ | --------------------------------------- |
| `(1, 0)`     | Largest-pending-first                   |
| `(0, 1)`     | Oldest-pending-first                    |
| `(0.5, 0.5)` | Default — equilíbrio entre os sinais    |

```bash
# default — equilibrado
python -m orchestrator.server --policy age-priority --pfs-bw 1GB

# favorecer só tamanho
python -m orchestrator.server --policy age-priority --alpha 1 --beta 0

# anti-starvation puro
python -m orchestrator.server --policy age-priority --alpha 0 --beta 1
```

**Restrições:** `α, β ≥ 0` e `α + β > 0`.

Testes:
[tests/age_priority.py](../tests/age_priority.py).

---

## EpochPriority (`epoch-priority`)

Política **dinâmica** baseada no progresso do treino.

A motivação é diferente das outras: jobs perto do fim do treino
investiram mais computação e têm menos checkpoints futuros;

O training side reporta `epoch` e `total_epochs` em cada heartbeat (campos
adicionados à `HeartbeatRequest` em `protocol/cluster.proto`). O migrater
guarda-os e propaga-os; o orquestrador armazena no `WorkerState` para a
policy consumir.

```
progress_i  = clamp(epoch_i / total_epochs_i, 0, 1)         (0 se total_epochs == 0)
priority_i  = floor + (1 - floor) · progress_i

rate_i      = pfs_bw · priority_i / Σ priority_j            (para j ∈ active)

inativo     → HOLD
```

A fórmula é **linear com floor**:

- `progress = 0` → `priority = floor` (jobs no início recebem uma share
  mínima; evita starvation total).
- `progress = 1` → `priority = 1.0` (jobs terminados recebem peso máximo).
- O slope `(1 - floor)` controla a agressividade.

**Default:** `floor = 0.2`. Significa que um job em `epoch=0/N` recebe 20%
do peso de um job em `epoch=N/N`.

**Casos particulares e flags extremos:**

| `floor` | Comportamento                                                     |
| ------- | ----------------------------------------------------------------- |
| `0.0`   | Job em `epoch=0` recebe `rate=0` (starvation no início)           |
| `0.2`   | Default                                                           |
| `1.0`   | Colapsa para `active-fair-share` (priority = 1 para todos)        |

**Defensivo:**

- `total_epochs == 0` (training não reportou) → tratado como `progress = 0`,
  recebe `floor`.
- `epoch > total_epochs` (improvável, mas defensivo) → clamp a `1.0`.

```bash
# default
python -m orchestrator.server --policy epoch-priority --pfs-bw 1GB

# floor mais agressivo: favorecer fortemente quem está perto do fim
python -m orchestrator.server --policy epoch-priority --epoch-floor 0.05

# colapsar para active-fair-share
python -m orchestrator.server --policy epoch-priority --epoch-floor 1.0
```

**Restrições:** `floor ∈ [0, 1]`.

Testes:
[tests/epoch_priority.py](../tests/epoch_priority.py).

---

## Tabela-resumo

| Policy               | Tipo     | Sinais usados                            | Flags próprias              |
| -------------------- | -------- | ---------------------------------------- | --------------------------- |
| `no-limit`           | baseline | —                                        | —                           |
| `fixed-rate`         | estática | —                                        | (`--pfs-bw`)                |
| `uniform-fair-share` | estática | nº workers                               | `--pfs-bw`                  |
| `active-fair-share`  | estática | nº workers ativos                        | `--pfs-bw`                  |
| `static-priority`    | estática | pesos do JSON                            | `--priority-map`            |
| `age-priority`       | dinâmica | `checkpoint_size`, `pending_since`       | `--alpha`, `--beta`         |
| `epoch-priority`     | dinâmica | `epoch / total_epochs`                   | `--epoch-floor`             |

Todas aceitam `--pfs-bw` para definir a banda total do PFS (default `1GB`).