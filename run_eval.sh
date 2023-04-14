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

export CUDA_VISIBLE_DEVICES="4,5,6,7"

RESTORE_DIR=${1:-"/workspaces/deeprte/ckpts/square_full_it_2023-04-03T12:47:33/models/latest/step_375000_2023-04-07T00:31:21"}
EVAL_CKPT_DIR=${3:-"./ckpts/eval_ckpts"}

python deeprte/train.py \
	--config=deeprte/config.py \
	--config.experiment_kwargs.config.evaluation.batch_size="4" \
	--config.experiment_kwargs.config.dataset.split_percentage="99%" \
	--config.checkpoint_dir="${EVAL_CKPT_DIR}" \
	--config.restore_dir="${RESTORE_DIR}" \
	--config.one_off_evaluate="true" \
	--jaxline_mode="eval"
