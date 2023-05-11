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

export CUDA_VISIBLE_DEVICES="1,2,3,4"

TIMESTAMP="$(date --iso-8601="seconds")"
DATASET_NAME=${1:-"g0.5-sigma_a3-sigma_t6"}
TFDS_DIR=${2:-"data/tfds"}

python deeprte/train.py \
	--config=deeprte/config.py \
	--config.checkpoint_dir="ckpts/${DATASET_NAME}_${TIMESTAMP%+*}" \
	--config.experiment_kwargs.config.dataset.name="rte/${DATASET_NAME}" \
	--config.experiment_kwargs.config.dataset.data_dir="${TFDS_DIR}" \
	--jaxline_mode="train" \
	--alsologtostderr="true"
