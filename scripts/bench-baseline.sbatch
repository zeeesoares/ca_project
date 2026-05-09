#!/bin/bash
# =============================================================================
# bench-baseline.sbatch — Baseline checkpoint worker (direct PFS write)
#
# Each array task runs one worker that writes N_CHECKPOINTS synthetic
# checkpoint files directly to the PFS (Lustre), measuring the blocking
# stall time incurred by training.
#
# Submit via scripts/submit-baseline.sh — do NOT submit directly.
# Required --export vars: EXPERIMENT_TAG, N_CHECKPOINTS, CHECKPOINT_SIZE_BYTES,
#                         PFS_DIR, RESULTS_DIR, N_WORKERS
# =============================================================================
#SBATCH --job-name=bench_baseline
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=normal-arm
#SBATCH --account=f202500010hpcvlabuminhoa
#SBATCH --time=01:00:00
#SBATCH --output=out/bench_baseline_%A_%a.out
#SBATCH --error=out/bench_baseline_%A_%a.err

set -euo pipefail

echo "=========================================="
echo "  bench-baseline worker"
echo "  Job    : ${SLURM_JOB_ID}"
echo "  Task   : ${SLURM_ARRAY_TASK_ID}"
echo "  Host   : $(hostname)"
echo "  Date   : $(date -Iseconds)"
echo "=========================================="

module load "Python/3.12.3-GCCcore-13.3.0"
source venv/bin/activate

# Verify env vars are set
: "${EXPERIMENT_TAG:?  EXPERIMENT_TAG must be set}"
: "${N_CHECKPOINTS:?   N_CHECKPOINTS must be set}"
: "${CHECKPOINT_SIZE_BYTES:? CHECKPOINT_SIZE_BYTES must be set}"
: "${PFS_DIR:?         PFS_DIR must be set}"
: "${RESULTS_DIR:?     RESULTS_DIR must be set}"
: "${N_WORKERS:?       N_WORKERS must be set}"
: "${DUMMY_SRC:?       DUMMY_SRC must be set}"
mkdir -p "$PFS_DIR" out

export WORKER_ID="${SLURM_ARRAY_TASK_ID}"

echo "  Experiment tag : ${EXPERIMENT_TAG}"
echo "  Worker id      : ${WORKER_ID}"
echo "  N checkpoints  : ${N_CHECKPOINTS}"
echo "  Checkpoint size: ${CHECKPOINT_SIZE_BYTES} bytes"
echo "  PFS dir        : ${PFS_DIR}"
echo "  Results dir    : ${RESULTS_DIR}"
echo "  Dummy src      : ${DUMMY_SRC}"
echo ""

python3 -u -m benchmarks.e2e.baseline_worker \
    --experiment-tag      "${EXPERIMENT_TAG}"       \
    --worker-id           "${SLURM_ARRAY_TASK_ID}"  \
    --n-checkpoints       "${N_CHECKPOINTS}"         \
    --checkpoint-size     "${CHECKPOINT_SIZE_BYTES}" \
    --pfs-dir             "${PFS_DIR}"               \
    --results-dir         "${RESULTS_DIR}"           \
    --n-workers           "${N_WORKERS}"             \
    --dummy-src           "${DUMMY_SRC}"

echo ""
echo "Worker ${SLURM_ARRAY_TASK_ID} completed at $(date -Iseconds)"
