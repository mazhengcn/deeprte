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

CUDA_DEVICES=${1:-"0,1,2,3"}
DATASET_NAME=${2:-"g0.5-sigma_a3-sigma_t6"}
RESTORE_DIR=${3:-"ckpts/g0.5-sigma_a3-sigma_t6_2023-05-12T23:11:39/models/latest/step_25000_2023-05-13T06:22:58"}
EVAL_CKPT_DIR=${4:-"ckpts/eval_ckpts"}

DEVICES=($(tr "," " " <<< "${CUDA_DEVICES}"))
BATCH_SIZE=${#DEVICES[@]}

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" python deeprte/train.py \
	--config=deeprte/config.py \
	--config.experiment_kwargs.config.dataset.name=rte/${DATASET_NAME} \
	--config.experiment_kwargs.config.evaluation.batch_size=${BATCH_SIZE} \
	--config.checkpoint_dir="${EVAL_CKPT_DIR}" \
	--config.restore_dir="${RESTORE_DIR}" \
	--config.one_off_evaluate="true" \
	--jaxline_mode="eval"
