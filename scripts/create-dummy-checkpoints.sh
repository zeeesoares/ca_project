#!/bin/bash

CP_DIR="checkpoints"
DD_IF="/dev/urandom"

set -euo pipefail

# Ensure checkpoints directory exists
if [ ! -d $CP_DIR ]; then
    echo "Directory '$CP_DIR' does not exist. Creating it..."
    mkdir -p $CP_DIR
fi

# Create dummy checkpoint files with random data
dd if=$DD_IF of=$CP_DIR/random_tiny.bin   bs=1M count=128  && echo -e "Created random_tiny.bin   (128 MB)\n"
dd if=$DD_IF of=$CP_DIR/random_small.bin  bs=1M count=512  && echo -e "Created random_small.bin  (512 MB)\n"
dd if=$DD_IF of=$CP_DIR/random_medium.bin bs=1M count=1024 && echo -e "Created random_medium.bin   (1 GB)\n"
dd if=$DD_IF of=$CP_DIR/random_large.bin  bs=1M count=5120 && echo -e "Created random_large.bin    (5 GB)"
