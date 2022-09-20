#!/bin/bash
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

export CUDA_VISIBLE_DEVICES="0"

RESTORE_PATH=${1:-"./ckpts/square_full_1_2022-02-09T00:04:32/models/latest/step_500000_2022-02-10T08:51:07"}
TEST_DATA_PATH=${2:-"./data/train/square_full_2.npz"}
EVAL_CKPT_DIR=${3:-"./ckpts/eval_ckpts"}

python run_deeprte.py \
	--config=deeprte/config.py \
	--config.experiment_kwargs.config.dataset.data_path="${TEST_DATA_PATH}" \
	--config.experiment_kwargs.config.evaluation.batch_size="10" \
	--config.checkpoint_dir="${EVAL_CKPT_DIR}" \
	--config.restore_path="${RESTORE_PATH}" \
	--config.one_off_evaluate="true" \
	--jaxline_mode="eval"
