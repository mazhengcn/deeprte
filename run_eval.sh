#!/usr/bin/bash

set -e

export CUDA_VISIBLE_DEVICES="0,1,2,3,4"

python deeprte/train.py --config=deeprte/config.py \
                    --config.experiment_kwargs.config.dataset.data_path=data/rte/rte_example2_converted.npz \
                    --jaxline_mode=eval