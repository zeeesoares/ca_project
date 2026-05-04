#!/bin/bash
#SBATCH --job-name=benchmark_token_bucket_x86
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=normal-x86
#SBATCH --account=f202500010hpcvlabuminhox
#SBATCH --time=02:30:00
#SBATCH --output=out/benchmark_token_bucket_x86_%j.out
#SBATCH --error=out/benchmark_token_bucket_x86_%j.err

set -eux

EXECUTABLE="./scripts/benchmark-token-bucket.sh"

if [ ! -f $EXECUTABLE ]; then
    echo "Error: $EXECUTABLE not found."
    echo "Run this with sbatch from the root of the repository."
    exit 1
fi

$EXECUTABLE
