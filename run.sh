#!/bin/bash

# NOTE Set HF_TOKEN environment variable for higher rate limits.

# NOTE
# On GPU partitions torch needs to be installed with CUDA support, example:
# pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CPU partitions:
# pip install torch --index-url https://download.pytorch.org/whl/cpu

# Fail on error
set -e

echo "Starting training experiment..."

python -m train.train \
    --enable-async \
    --enable-compression

echo "Experiment finished."
