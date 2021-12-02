#!/usr/bin/bash

set -e

export CUDA_VISIBLE_DEVICES="1,4,5,6,7"

python run_main.py --config=deeprte/config.py