#!/bin/bash
# =============================================================================
# bench-worker.sbatch — Orchestrated checkpoint worker (local staging)
#
# Each array task:
#   1. Reads orchestrator address from the PFS coordination file.
#   2. Polls until the orchestrator gRPC server is reachable.
#   3. Starts migrater.server in background (connects to orchestrator).
#   4. Runs orchestrated_worker.py: write to local NVMe, notify migrater,
#      measure async migration time to PFS.
#   5. Stops migrater.
#
# Submit via scripts/submit-orchestrated.sh — do NOT submit directly.
# Required --export vars: EXPERIMENT_TAG, N_CHECKPOINTS, CHECKPOINT_SIZE_BYTES,
#                         PFS_DIR, RESULTS_DIR, POLICY, PFS_BW_BPS, N_WORKERS,
#                         COORD_FILE, ORCH_PORT
# =============================================================================
#SBATCH --job-name=bench_worker
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=normal-arm
#SBATCH --account=f202500010hpcvlabuminhoa
#SBATCH --time=01:30:00
#SBATCH --output=out/bench_worker_%A_%a.out
#SBATCH --error=out/bench_worker_%A_%a.err

set -euo pipefail

echo "=========================================="
echo "  bench-worker (orchestrated)"
echo "  Job    : ${SLURM_JOB_ID}"
echo "  Task   : ${SLURM_ARRAY_TASK_ID}"
echo "  Host   : $(hostname)"
echo "  Date   : $(date -Iseconds)"
echo "=========================================="

module load "Python/3.12.3-GCCcore-13.3.0"
source venv/bin/activate

: "${EXPERIMENT_TAG:?        EXPERIMENT_TAG must be set}"
: "${N_CHECKPOINTS:?         N_CHECKPOINTS must be set}"
: "${CHECKPOINT_SIZE_BYTES:? CHECKPOINT_SIZE_BYTES must be set}"
: "${PFS_DIR:?               PFS_DIR must be set}"
: "${RESULTS_DIR:?           RESULTS_DIR must be set}"
: "${POLICY:?                POLICY must be set}"
: "${PFS_BW_BPS:?            PFS_BW_BPS must be set}"
: "${N_WORKERS:?             N_WORKERS must be set}"
: "${COORD_FILE:?            COORD_FILE must be set}"
: "${ORCH_PORT:?             ORCH_PORT must be set}"
: "${DUMMY_SRC:?             DUMMY_SRC must be set}"
: "${COMPUTE_TIME:?          COMPUTE_TIME must be set}"
: "${USE_JITTER:?            USE_JITTER can be set to 'true' to apply random jitter to compute time}"

LOCAL_DIR="/tmp/ca_bench_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
MIGRATER_LOG="logs/migrater_$(hostname)_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

mkdir -p "${LOCAL_DIR}" out logs

echo "  Experiment tag : ${EXPERIMENT_TAG}"
echo "  Worker id      : ${SLURM_ARRAY_TASK_ID}"
echo "  Policy         : ${POLICY}"
echo "  PFS BW         : ${PFS_BW_BPS} B/s"
echo "  N workers      : ${N_WORKERS}"
echo "  Local dir      : ${LOCAL_DIR}"
echo "  PFS dir        : ${PFS_DIR}"
echo "  Coord file     : ${COORD_FILE}"
echo "  Dummy src      : ${DUMMY_SRC}"
echo "  Compute time   : ${COMPUTE_TIME}"
echo "  Jitter         : ${USE_JITTER:-false}"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Wait for orchestrator coordination file and gRPC readiness
# ---------------------------------------------------------------------------
echo "  Waiting for orchestrator coordination file..."

WAIT_LIMIT=120    # seconds
WAITED=0
while [ ! -f "${COORD_FILE}" ]; do
    sleep 2
    WAITED=$((WAITED + 2))
    if [ ${WAITED} -ge ${WAIT_LIMIT} ]; then
        echo "ERROR: Timed out waiting for ${COORD_FILE}" >&2
        exit 1
    fi
done

ORCH_HOST=$(cut -d: -f1 "${COORD_FILE}")
echo "  Orchestrator address: ${ORCH_HOST}:${ORCH_PORT}"

# Poll until orchestrator gRPC is actually listening
echo "  Polling orchestrator gRPC readiness..."
python3 -c "
import grpc, time, sys
addr = '${ORCH_HOST}:${ORCH_PORT}'
for attempt in range(60):
    try:
        ch = grpc.insecure_channel(addr)
        grpc.channel_ready_future(ch).result(timeout=2)
        ch.close()
        print(f'  Orchestrator ready at {addr}')
        sys.exit(0)
    except Exception:
        time.sleep(1)
print(f'ERROR: orchestrator at {addr} not reachable', file=sys.stderr)
sys.exit(1)
"

# ---------------------------------------------------------------------------
# Step 2: Start migrater in background
# ---------------------------------------------------------------------------
echo "  Starting migrater..."
python3 -u -m src.migrater.server \
    --orchestrator-addr "${ORCH_HOST}" \
    --orchestrator-port "${ORCH_PORT}" \
    > "${MIGRATER_LOG}" 2>&1 &

MIGRATER_PID=$!
echo "  Migrater PID: ${MIGRATER_PID}  log: ${MIGRATER_LOG}"

# Give the migrater time to connect and exchange the first heartbeat/instruction
sleep 3

# ---------------------------------------------------------------------------
# Step 3: Run the orchestrated worker benchmark
# ---------------------------------------------------------------------------
export WORKER_ID="${SLURM_ARRAY_TASK_ID}"

# Set Jitter to 0

SLEEP_START=0

if [[ "${USE_JITTER:-false}" == "true" ]]; then
    SLEEP_START=$((RANDOM % 20))
    echo "  [Workload] Aplicando jitter inicial de ${SLEEP_START}s"
    sleep "${SLEEP_START}"
fi

if [[ "${COMPUTE_TIME}" == "random" ]]; then
    CURRENT_COMPUTE=$((10 + RANDOM % 21))
else
    CURRENT_COMPUTE="${COMPUTE_TIME:-0}"
fi

python3 -u -m benchmarks.e2e.orchestrated_worker \
    --experiment-tag  "${EXPERIMENT_TAG}"        \
    --worker-id       "${SLURM_ARRAY_TASK_ID}"   \
    --n-checkpoints   "${N_CHECKPOINTS}"         \
    --checkpoint-size "${CHECKPOINT_SIZE_BYTES}" \
    --local-dir       "${LOCAL_DIR}"             \
    --pfs-dir         "${PFS_DIR}"               \
    --results-dir     "${RESULTS_DIR}"           \
    --policy          "${POLICY}"                \
    --pfs-bw-bps      "${PFS_BW_BPS}"            \
    --n-workers       "${N_WORKERS}"             \
    --dummy-src       "${DUMMY_SRC}"             \
    --compute-time    "${CURRENT_COMPUTE}"       \
    --jitter          "${SLEEP_START}"

# ---------------------------------------------------------------------------
# Step 4: Stop migrater
# ---------------------------------------------------------------------------
echo ""
echo "  Stopping migrater (PID ${MIGRATER_PID})..."
kill "${MIGRATER_PID}" 2>/dev/null || true
wait "${MIGRATER_PID}" 2>/dev/null || true

# Clean up local staging directory
rm -rf "${LOCAL_DIR}"

echo "Worker ${SLURM_ARRAY_TASK_ID} completed at $(date -Iseconds)"
