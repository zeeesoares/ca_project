#!/bin/bash

NUM_WORKERS=${1:-2}
ORCH_HOST=${2:-"localhost"} 
ORCH_PORT=${3:-50051}

echo "Lançando $NUM_WORKERS workers..."

for i in $(seq 1 "$NUM_WORKERS"); do
    sbatch --export=ALL,ORCH_ADDR="$ORCH_HOST",ORCH_PORT="$ORCH_PORT",WORKER_ID="$i" run-arm.sh
done
