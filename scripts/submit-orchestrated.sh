#!/bin/bash
# =============================================================================
# submit-orchestrated.sh — Submit an orchestrated checkpoint benchmark
#
# Launches two SLURM submissions:
#   1. bench-orchestrator.sbatch  — exclusive node running orchestrator.server
#   2. bench-worker.sbatch        — job array (N_WORKERS tasks), each running
#                                   migrater + orchestrated_worker.py
#
# Worker array starts after the orchestrator job begins executing
# (--dependency=after:ORCH_JOB_ID), then polls a shared coordination file
# written by the orchestrator to discover its hostname.
#
# Usage:
#   ./scripts/submit-orchestrated.sh [OPTIONS]
#
# Options:
#   -n, --n-workers           Concurrent workers                [default: 4]
#   -c, --n-checkpoints       Checkpoints per worker            [default: 3]
#   -s, --checkpoint-size     Bytes per checkpoint              [default: 268435456]
#   --policy                  Orchestrator policy               [default: uniform-fair-share]
#   --pfs-bw                  Total PFS bandwidth (e.g. 500MB)  [default: 500MB]
#   --pfs-bw-bps              Same as --pfs-bw but in bytes/s   (auto-computed if --pfs-bw set)
#   -p, --pfs-dir             PFS destination directory
#   -r, --results-dir         Results output directory
#   -t, --experiment-tag      Run label [default: orch_<policy>_<timestamp>]
#   --orch-port               Orchestrator port                 [default: 50052]
#
# Example — uniform fair share, 4 workers, 500 MB/s total:
#   ./scripts/submit-orchestrated.sh \
#       --n-workers 4 --n-checkpoints 3 --checkpoint-size 268435456 \
#       --policy uniform-fair-share --pfs-bw 500MB
#
# Example — sweep multiple policies:
#   for POLICY in uniform-fair-share active-fair-share age-priority epoch-priority; do
#       ./scripts/submit-orchestrated.sh --policy $POLICY --pfs-bw 500MB --n-workers 4
#   done
# =============================================================================

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# ---- Defaults ---------------------------------------------------------------
N_WORKERS=4
N_CHECKPOINTS=3
CHECKPOINT_SIZE_BYTES=$((5 * 1024 * 1024 * 1024))   # 5 GB
POLICY="uniform-fair-share"
PFS_BW_HUMAN="500MB"
BENCH_BASE="/projects/F202500010HPCVLABUMINHO/josesoares/pca/ca_bench"
PFS_DIR="${BENCH_BASE}/pfs"
RESULTS_DIR="${BENCH_BASE}/results"
ORCH_PORT=50052
DUMMY_SRC="checkpoints/random_large.bin"

# ---- Argument parsing -------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--n-workers)           N_WORKERS="$2";             shift 2 ;;
        -c|--n-checkpoints)       N_CHECKPOINTS="$2";         shift 2 ;;
        -s|--checkpoint-size)     CHECKPOINT_SIZE_BYTES="$2"; shift 2 ;;
        --policy)                 POLICY="$2";                shift 2 ;;
        --pfs-bw)                 PFS_BW_HUMAN="$2";          shift 2 ;;
        --pfs-bw-bps)             PFS_BW_BPS_OVERRIDE="$2";   shift 2 ;;
        -p|--pfs-dir)             PFS_DIR="$2";               shift 2 ;;
        -r|--results-dir)         RESULTS_DIR="$2";           shift 2 ;;
        -t|--experiment-tag)      EXPERIMENT_TAG_OVERRIDE="$2"; shift 2 ;;
        --orch-port)              ORCH_PORT="$2";              shift 2 ;;
        -d|--dummy-src)           DUMMY_SRC="$2";            shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ---- Derive bytes/s from human-readable PFS BW ------------------------------
if [[ -n "${PFS_BW_BPS_OVERRIDE:-}" ]]; then
    PFS_BW_BPS="${PFS_BW_BPS_OVERRIDE}"
else
    PFS_BW_BPS=$(python3 -c "
from utils.size_parser import parse_size
print(parse_size('${PFS_BW_HUMAN}'))
")
fi

EXPERIMENT_TAG="${EXPERIMENT_TAG_OVERRIDE:-orch_${POLICY//\//-}_$(date +%Y%m%d_%H%M%S)}"

# Coordination file lives on the PFS so all nodes can read it
COORD_FILE="${RESULTS_DIR}/${EXPERIMENT_TAG}/orch_addr.txt"

# ---- Pre-flight -------------------------------------------------------------
mkdir -p out "${RESULTS_DIR}/${EXPERIMENT_TAG}"

echo "============================================================"
echo "  Submitting orchestrated benchmark"
echo "  Experiment tag  : ${EXPERIMENT_TAG}"
echo "  Policy          : ${POLICY}"
echo "  PFS bandwidth   : ${PFS_BW_HUMAN}  (${PFS_BW_BPS} B/s)"
echo "  Workers         : ${N_WORKERS}"
echo "  Checkpoints     : ${N_CHECKPOINTS} × $((CHECKPOINT_SIZE_BYTES / 1024 / 1024)) MB"
echo "  PFS dir         : ${PFS_DIR}"
echo "  Results dir     : ${RESULTS_DIR}"
echo "  Coord file      : ${COORD_FILE}"
echo "  Orch port       : ${ORCH_PORT}"
echo "  Dummy src       : ${DUMMY_SRC}"
echo "============================================================"
echo ""

# ---- Submit orchestrator ----------------------------------------------------
ORCH_JOB_ID=$(sbatch \
    --export=ALL,\
EXPERIMENT_TAG="${EXPERIMENT_TAG}",\
POLICY="${POLICY}",\
PFS_BW="${PFS_BW_HUMAN}",\
ORCH_PORT="${ORCH_PORT}",\
COORD_FILE="${COORD_FILE}" \
    scripts/bench-orchestrator.sbatch \
    | awk '{print $NF}')

echo "  Submitted orchestrator job: ${ORCH_JOB_ID}"

# ---- Submit worker array (starts after orchestrator begins executing) --------
WORKER_JOB_ID=$(sbatch \
    --array="1-${N_WORKERS}" \
    --dependency="after:${ORCH_JOB_ID}" \
    --export=ALL,\
EXPERIMENT_TAG="${EXPERIMENT_TAG}",\
N_CHECKPOINTS="${N_CHECKPOINTS}",\
CHECKPOINT_SIZE_BYTES="${CHECKPOINT_SIZE_BYTES}",\
DUMMY_SRC="${DUMMY_SRC}",\
PFS_DIR="${PFS_DIR}/${EXPERIMENT_TAG}",\
RESULTS_DIR="${RESULTS_DIR}",\
POLICY="${POLICY}",\
PFS_BW_BPS="${PFS_BW_BPS}",\
N_WORKERS="${N_WORKERS}",\
COORD_FILE="${COORD_FILE}",\
ORCH_PORT="${ORCH_PORT}" \
    scripts/bench-worker.sbatch \
    | awk '{print $NF}')

echo "  Submitted worker array    : ${WORKER_JOB_ID}"
echo "  (workers depend on orchestrator: after:${ORCH_JOB_ID})"
echo ""
echo "  Monitor:"
echo "    squeue -j ${ORCH_JOB_ID},${WORKER_JOB_ID}"
echo ""
echo "  Outputs:"
echo "    out/bench_orchestrator_${ORCH_JOB_ID}.out"
echo "    out/bench_worker_${WORKER_JOB_ID}_*.out"
echo ""
echo "  Cancel experiment (orchestrator + workers):"
echo "    scancel ${ORCH_JOB_ID} ${WORKER_JOB_ID}"
echo ""
echo "  Once all worker tasks finish, collect results:"
echo ""
echo "    python3 -m benchmarks.e2e.collect \\"
echo "        --results-dir ${RESULTS_DIR}/${EXPERIMENT_TAG} \\"
echo "        --output-csv  ${RESULTS_DIR}/${EXPERIMENT_TAG}/summary.csv"


for POLICY in uniform-fair-share active-fair-share age-priority epoch-priority; do
    ./scripts/submit-orchestrated.sh --policy $POLICY --pfs-bw 500MB --n-workers 4
done