# HOWTO — Orchestrator & Migrater Communication

## Pré-requisitos

Todos os comandos são executados a partir da raiz do projeto.
O Python do virtualenv é `venv/bin/python`.

```bash
# Verificar que as dependências estão instaladas
venv/bin/python -c "import grpc; print(grpc.__version__)"
```

---

## 1. Correr os testes automatizados

Não requer servidores externos — cada teste levanta o seu próprio servidor numa thread.

```bash
venv/bin/python -m unittest tests.test_communication -v
```

Esperado: **21 tests, 0 failures**.

---

## 2. Testar manualmente com servidores reais

Abre **3 terminais** na raiz do projeto.

### Terminal 1 — Migrater (porta 50051)

```bash
venv/bin/python -m migrater.server
```

```
Migrater running on 50051
```

### Terminal 2 — Orchestrator (porta 50052)

```bash
# Policy por omissão: UniformBandwidth com 1 Gbps
venv/bin/python -m orchestrator.server
```

```
Orchestrator running on 50052 | policy=UniformBandwidthPolicy
```

**Opções disponíveis:**

```bash
# Mudar o bandwidth total disponível no PFS (em bps)
venv/bin/python -m orchestrator.server --pfs-bw 500000000   # 500 Mbps
```

### Terminal 3 — Cliente de teste

```bash
venv/bin/python -c "
import time
from protocol.migrater.client import MigraterClient
from protocol.orchestrator.client import OrchestratorClient

# --- Migrater ---
m = MigraterClient()  # localhost:50051
m.notify_checkpoint_saved('/tmp/ckpt.pt', '/pfs/ckpt.pt', time.time())
print('Migrater: notificacao enviada')

# --- Orchestrator ---
o = OrchestratorClient(worker_id='worker-0')  # localhost:50052

MB = 1024 * 1024
heartbeats = [
    (0.0,        False),  # sem dados        -> HOLD
    (50  * MB,   False),  # abaixo threshold -> HOLD
    (200 * MB,   False),  # acima threshold  -> START_FLUSH (com rate limit)
    (200 * MB,   True),   # a migrar         -> CHANGE_RATE
]

actions = {0: 'HOLD', 1: 'START_FLUSH', 2: 'CHANGE_RATE'}
for instruction in o.monitor(heartbeats):
    name  = actions.get(instruction.action, '?')
    rate  = f' @ {instruction.rate_limit_bps/1e6:.1f} Mbps' if instruction.rate_limit_bps else ''
    print(f'Orchestrator: {name}{rate}')

o.close()
"
```
