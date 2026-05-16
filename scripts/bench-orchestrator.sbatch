#!/bin/bash
# =============================================================================
# bench-orchestrator.sbatch — Orchestrator node (exclusive)
#
# Runs the orchestrator.server on a dedicated node for the duration of the
# benchmark. Writes its hostname:port to a shared coordination file on the
# PFS so that worker nodes can discover it.
#
# Submit via scripts/submit-orchestrated.sh — do NOT submit directly.
# Required --export vars: EXPERIMENT_TAG, POLICY, PFS_BW, ORCH_PORT,
#                         COORD_FILE
# =============================================================================
#SBATCH --job-name=bench_orchestrator
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=normal-arm
#SBATCH --account=f202500010hpcvlabuminhoa
#SBATCH --time=02:00:00
#SBATCH --output=out/bench_orchestrator_%j.out
#SBATCH --error=out/bench_orchestrator_%j.err

set -euo pipefail

echo "=========================================="
echo "  bench-orchestrator"
echo "  Job    : ${SLURM_JOB_ID}"
echo "  Host   : $(hostname)"
echo "  Date   : $(date -Iseconds)"
echo "=========================================="

module load "Python/3.12.3-GCCcore-13.3.0"
source venv/bin/activate

: "${EXPERIMENT_TAG:? EXPERIMENT_TAG must be set}"
: "${POLICY:?         POLICY must be set}"
: "${PFS_BW:?         PFS_BW must be set (e.g. 500MB)}"
: "${ORCH_PORT:?      ORCH_PORT must be set}"
: "${COORD_FILE:?     COORD_FILE must be set}"

mkdir -p out "$(dirname "${COORD_FILE}")" logs

echo "  Experiment tag : ${EXPERIMENT_TAG}"
echo "  Policy         : ${POLICY}"
echo "  PFS bandwidth  : ${PFS_BW}"
echo "  Port           : ${ORCH_PORT}"
echo "  Coord file     : ${COORD_FILE}"
echo ""

# Write hostname:port to coordination file BEFORE starting the server.
# Worker jobs poll this file to discover the orchestrator address.
ORCH_ADDR="$(hostname):${ORCH_PORT}"
echo "${ORCH_ADDR}" > "${COORD_FILE}"
echo "  Written orchestrator address: ${ORCH_ADDR}"
echo ""

# Run orchestrator — stays alive until SLURM cancels or time limit
python3 -u -m src.orchestrator.server \
    --port   "${ORCH_PORT}" \
    --policy "${POLICY}" \
    --pfs-bw "${PFS_BW}"

echo "Orchestrator exited at $(date -Iseconds)"
