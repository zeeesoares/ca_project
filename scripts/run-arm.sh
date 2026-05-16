#!/bin/bash
#SBATCH --job-name=train_llm
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=dev-arm
#SBATCH --account=f202500010hpcvlabuminhoa
#SBATCH --time=00:10:00
#SBATCH --output=out/train_%j.out

module load "Python/3.12.3-GCCcore-13.3.0"

ORCH_ADDR=$1
ORCH_PORT=$2

source venv/bin/activate

python3 -u -m src.migrater.server    \
    --orchestrator-addr "$ORCH_ADDR" \
    --orchestrator-port "$ORCH_PORT" > "logs/migrater_$(hostname).log" 2>&1 &
MIGRATER_PID=$!

echo "Migrater on host $(hostname) with PID $MIGRATER_PID"

sleep 2

python3 -u -m src.train.train \
    --profile                 \
    --total-steps         50  \
    --checkpoint-interval 10

kill $MIGRATER_PID
