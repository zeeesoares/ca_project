#!/bin/bash
#SBATCH --job-name=llm_checkpointing_orchestrator
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=dev-arm
#SBATCH --account=f202500010hpcvlabuminhoa
#SBATCH --time=00:10:00
#SBATCH --output=out/llm_checkpointing_orchestrator_%j.out

module load "Python/3.12.3-GCCcore-13.3.0"

POLICY=$1
BW=$2
ORCHESTRATOR_PORT=$3

echo "Running orchestrator with policy: $POLICY and bandwidth: $BW"

# Get host
HOST=$(hostname)
echo "Orchestrator running on host: $HOST"

source venv/bin/activate
python3 -m orchestrator.server \
    --policy "$POLICY" \
    --bandwidth "$BW" \
    --port "$ORCHESTRATOR_PORT"
