#!/bin/bash
# Chạy training end-to-end cho 1 scene BTS
set -e
CONFIG=${1:-configs/default.yaml}
python train.py --config "$CONFIG"
