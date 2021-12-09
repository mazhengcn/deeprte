#!/usr/bin/bash

set -e

export CUDA_VISIBLE_DEVICES="2"

python train.py --config=deeprte/config.py --config.experiment_kwargs.config.dataset.data_path=data/rte/rte_2d_converted.npz --jaxline_mode=train_eval_multithreaded