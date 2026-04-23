# HOWTO — Orchestrator & Migrater Communication

## Pré-requisitos

Todos os comandos são executados a partir da raiz do projeto.
O Python do virtualenv é `venv/bin/python`.

```bash
# Verificar que as dependências estão instaladas
venv/bin/python -c "import grpc; print(grpc.__version__)"
```

---

## Testar com servidores

### Gerar um ficheiro de teste

```bash
dd if=/dev/urandom of=/tmp/ckpt.pt bs=1M count=50 # 50MiB
```

Abre **3 terminais** na raiz do projeto.

### Terminal 1 — Orchestrator (porta 50052)

```bash
venv/bin/python -m orchestrator.server --policy uniform-fair-share --pfs-bw 1000000
```


### Terminal 2 — Migrater (porta 50051)

```bash
venv/bin/python -m migrater.server
```

```
Migrater running on 50051
```

### Terminal 3 — Cliente de teste

```bash
venv/bin/python -c "
import time
from protocol.migrater.client import MigraterClient
m = MigraterClient()
m.notify_checkpoint_saved('/tmp/ckpt.pt', '/tmp/ckpt_pfs.pt', time.time())
print('notification sent')
"
```


### Verificar Resultados

```bash
md5sum /tmp/ckpt.pt /tmp/ckpt_pfs.pt
```
