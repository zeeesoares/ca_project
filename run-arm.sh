#!/bin/bash
#SBATCH --job-name=chk_arm
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=dev-arm
#SBATCH --account=f202500010hpcvlabuminhoa
#SBATCH --time=00:10:00
#SBATCH --output=out/chk_arm_%j.out

set -euxo pipefail

source venv/bin/activate

python3 -m train.train
