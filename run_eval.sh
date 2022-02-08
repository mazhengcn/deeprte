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

export CUDA_VISIBLE_DEVICES="4,5,6,7"

RESTORE_PATH="data/experiments/rect_delta_bc_r_2022-01-23T22:46:41/models/latest/step_400000_2022-01-24T03:31:39"
TEST_DATA_PATH="data/experiments/test/rte_example2_converted.npz"
EVAL_CKPT_DIR="data/experiments/eval"

python deeprte/train.py \
    --config=deeprte/config.py \
    --config.experiment_kwargs.config.dataset.data_path=${TEST_DATA_PATH} \
    --config.checkpoint_dir=${EVAL_CKPT_DIR} \
    --config.restore_path=${RESTORE_PATH} \
    --config.one_off_evaluate="true" \
    --jaxline_mode="eval"