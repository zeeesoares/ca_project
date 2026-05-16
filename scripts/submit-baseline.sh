#!/bin/bash
# =============================================================================
# submit-baseline.sh — Submit a baseline checkpoint benchmark on Deucalion
#
# Launches N_WORKERS concurrent SLURM jobs, each writing N_CHECKPOINTS
# synthetic checkpoint files directly to the PFS (no migrater/orchestrator).
# This is the "no local staging" reference for comparison against the
# orchestrated solution.
#
# Usage:
#   ./scripts/submit-baseline.sh [OPTIONS]
#
# Options:
#   -n, --n-workers           Number of concurrent workers  [default: 4]
#   -c, --n-checkpoints       Checkpoints per worker        [default: 3]
#   -s, --checkpoint-size     Size per checkpoint (bytes)   [default: 268435456 = 256 MB]
#   -p, --pfs-dir             PFS destination directory
#   -r, --results-dir         Results output directory
#   -t, --experiment-tag      Human-readable run label      [default: baseline_<timestamp>]
#   -d, --dummy-src           Path to dummy checkpoint source file
#
# Example:
#   ./scripts/submit-baseline.sh \
#       --n-workers 4 \
#       --n-checkpoints 3 \
#       --checkpoint-size 268435456 \
#       --pfs-dir /projects/f202500010hpcvlabuminhoa/ca_bench/pfs \
#       --results-dir /projects/f202500010hpcvlabuminhoa/ca_bench/results
# =============================================================================

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."  # run from project root

# ---- Defaults ---------------------------------------------------------------
N_WORKERS=4
N_CHECKPOINTS=3
CHECKPOINT_SIZE_BYTES=$((5 * 1024 * 1024 * 1024))   # 5 GB
DUMMY_SRC="checkpoints/random_large.bin"
BENCH_BASE="/projects/F202500010HPCVLABUMINHO/josesoares/pca/ca_bench"
PFS_DIR="${BENCH_BASE}/pfs"
RESULTS_DIR="${BENCH_BASE}/results"
EXPERIMENT_TAG="baseline_$(date +%Y%m%d_%H%M%S)"

# ---- Argument parsing -------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--n-workers)       N_WORKERS="$2";             shift 2 ;;
        -c|--n-checkpoints)   N_CHECKPOINTS="$2";         shift 2 ;;
        -s|--checkpoint-size) CHECKPOINT_SIZE_BYTES="$2"; shift 2 ;;
        -p|--pfs-dir)         PFS_DIR="$2";               shift 2 ;;
        -r|--results-dir)     RESULTS_DIR="$2";           shift 2 ;;
        -t|--experiment-tag)  EXPERIMENT_TAG="$2";        shift 2 ;;
        -d|--dummy-src)       DUMMY_SRC="$2";             shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ---- Pre-flight checks ------------------------------------------------------
mkdir -p out "${RESULTS_DIR}/${EXPERIMENT_TAG}"

echo "============================================================"
echo "  Submitting baseline benchmark"
echo "  Experiment tag  : ${EXPERIMENT_TAG}"
echo "  Workers         : ${N_WORKERS}"
echo "  Checkpoints     : ${N_CHECKPOINTS} × $((CHECKPOINT_SIZE_BYTES / 1024 / 1024)) MB"
echo "  PFS dir         : ${PFS_DIR}"
echo "  Results dir     : ${RESULTS_DIR}"
echo "============================================================"
echo ""

# ---- Submit job array -------------------------------------------------------
JOB_ID=$(sbatch \
    --array="1-${N_WORKERS}" \
    --export=ALL,\
EXPERIMENT_TAG="${EXPERIMENT_TAG}",\
N_CHECKPOINTS="${N_CHECKPOINTS}",\
CHECKPOINT_SIZE_BYTES="${CHECKPOINT_SIZE_BYTES}",\
PFS_DIR="${PFS_DIR}/${EXPERIMENT_TAG}",\
RESULTS_DIR="${RESULTS_DIR}",\
N_WORKERS="${N_WORKERS}",\
DUMMY_SRC="${DUMMY_SRC}" \
    scripts/bench-baseline.sbatch \
    | awk '{print $NF}')

echo "  Submitted job array: ${JOB_ID}"
echo ""
echo "  Monitor:  squeue -j ${JOB_ID}"
echo "  Outputs:  out/bench_baseline_${JOB_ID}_*.out"
echo ""
echo "  Once all tasks finish, collect results:"
echo ""
echo "    python3 -m benchmarks.e2e.collect \\"
echo "        --results-dir ${RESULTS_DIR}/${EXPERIMENT_TAG} \\"
echo "        --output-csv  ${RESULTS_DIR}/${EXPERIMENT_TAG}/summary.csv"
