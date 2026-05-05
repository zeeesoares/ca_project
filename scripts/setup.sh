#!/bin/bash
#SBATCH --job-name=create_venv
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=dev-arm
#SBATCH --account=f202500010hpcvlabuminhoa
#SBATCH --time=00:10:00
#SBATCH --output=out/chk_arm_%j.out

set -e  # Exit immediately if a command exits with a non-zero status

echo "Loading Python module..."
module load "Python/3.12.3-GCCcore-13.3.0"

echo "which python: $(which python)"
echo "which python3: $(which python3)"

echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

exit 0
