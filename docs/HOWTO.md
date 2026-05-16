# HOWTO -- Orchestrator & Migrater Communication

## Prerequisites

All commands should be executed from the project root.

Activate the virtual environment first:

```bash
source venv/bin/activate
```

After activation, use `python3` for all Python commands.

```bash
# Check that the dependencies are installed
python3 -c "import grpc; print(grpc.__version__)"
```

---

## Test with servers

Open **3 terminals** in the project root.

In each terminal, activate the virtual environment first:

```bash
source venv/bin/activate
```

### Terminal 1 -- Orchestrator, port 50052

```bash
python3 -m src.orchestrator.server \
 --port 50052 \
 --policy uniform-fair-share \
 --pfs-bw $((500 * 1024 ** 2))  # 500 MiB/s
```

### Terminal 2 -- Migrater, port 50051

```bash
python3 -m src.migrater.server \
 --orchestrator-addr localhost \
 --orchestrator-port 50052
```

### Terminal 3 -- Training client

```bash
python3 -m src.train.train \
 --checkpoint-pfs-dir /tmp/pfs \
 --checkpoint-local-dir /tmp/local \
 --total-steps 50 \
 --checkpoint-interval 10
```

Or send a manual test notification, after creating a dummy checkpoint file:

```bash
# Create a dummy checkpoint file of 50MiB
dd if=/dev/urandom of=/tmp/checkpoint.pt bs=1M count=50  # 50MiB
```

```bash
venv/bin/python -c "
import time
from src.protocol.migrater.client import MigraterClient
m = MigraterClient()
m.notify_checkpoint_saved('/tmp/checkpoint.pt', '/tmp/checkpoint_pfs.pt', time.time(), 1, 5)
print('notification sent')
"
```

One liner:

```bash
venv/bin/python -c "import time; from src.protocol.migrater.client import MigraterClient; m = MigraterClient(); m.notify_checkpoint_saved('/tmp/checkpoint.pt', '/tmp/checkpoint_pfs.pt', time.time(), 1, 5); print('notification sent')"
```

Verify checkpoint integrity with:

```bash
md5sum /tmp/checkpoint.pt /tmp/checkpoint_pfs.pt
```
