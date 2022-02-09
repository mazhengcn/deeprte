#!/usr/bin/bash
# Copyright 2022 Zheng Ma
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -e

export CUDA_VISIBLE_DEVICES="0,3,4,5,7"

TIMESTAMP=${1:-"$(date --iso-8601="seconds")"}
DATAPATH=${2:-"data/train/square_full_1.npz"}

python deeprte/train.py \
    --config=deeprte/config.py \
    --config.experiment_kwargs.config.dataset.data_path=${DATAPATH} \
    --config.experiment_kwargs.config.training.batch_size="30" \
    --config.checkpoint_dir="data/experiments/square_full_1_${TIMESTAMP%+*}" \
    --jaxline_mode="train_eval_multithreaded" \
    --alsologtostderr="true"
