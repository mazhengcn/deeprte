#!/usr/bin/bash

set -e

export CUDA_VISIBLE_DEVICES="0,1,2,3,4"

python deeprte/train.py --config=deeprte/config.py \
                    --config.experiment_kwargs.config.dataset.data_path=data/rte/rte_2d_bc_delta_funcs_converted.npz \
                    --jaxline_mode=train_eval_multithreaded