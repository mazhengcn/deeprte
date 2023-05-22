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

DATASET_NAME=${1:-"g0.5-sigma_a3-sigma_t6"}
BATCH_SIZE=${2:-"8"}
RESTORE_DIR=${3:-"None"}
CUDA_DEVICES=${4:-""}

if [ -n "${CUDA_DEVICES}" ]; then
	export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
	DEVICES=($(tr "," " " <<< "${CUDA_DEVICES}"))
	ACCUM_GRADS_STEPS=$((BATCH_SIZE / ${#DEVICES[@]}))
else
	ACCUM_GRADS_STEPS="1"
fi

TRAIN_ARGS="--config=deeprte/config.py \
	--config.experiment_kwargs.config.dataset.name=rte/${DATASET_NAME} \
	--config.experiment_kwargs.config.training.batch_size=${BATCH_SIZE} \
	--config.experiment_kwargs.config.training.accum_grads_steps=${ACCUM_GRADS_STEPS} \
	--jaxline_mode=train \
	--alsologtostderr=true
	"

if [ "${RESTORE_DIR}" = "None" ]; then
	TIMESTAMP="$(date --iso-8601="seconds")"
	CKPT_NAME="${DATASET_NAME}_${TIMESTAMP%+*}"
	TRAIN_ARGS="${TRAIN_ARGS} --config.checkpoint_dir=ckpts/${CKPT_NAME}"
else
	CKPT_DIR="${RESTORE_DIR%%/models*}"
	CKPT_NAME="${CKPT_DIR##ckpts/}"
	TRAIN_ARGS="${TRAIN_ARGS} --config.checkpoint_dir=${CKPT_DIR} --config.restore_dir=${RESTORE_DIR}"
fi

if ! type screen > /dev/null 2>&1; then
    apt-get update
    echo "Installing screen..."
    apt-get -y install --no-install-recommends screen
    # Clean up
    apt-get clean -y
    rm -rf /var/lib/apt/lists/*
fi

screen -S "${CKPT_NAME}" python deeprte/train.py ${TRAIN_ARGS}