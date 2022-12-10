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

export CUDA_VISIBLE_DEVICES="2,3"

TIMESTAMP="$(date --iso-8601="seconds")"
SOURCE_DIR=${1:-"./rte_data/rte_data/matlab/train-delta"}
DATA_NAME_LIST=${2:-"train_delta_1.mat,train_delta_2.mat"}
TEST_DATA_SAVE_PATH=${3:-"./rte_data/test/train_delta_${TIMESTAMP%+*}"}

python run_deeprte.py \
	--config=deeprte/config.py \
	--source_dir="${SOURCE_DIR}" \
	--data_name_list="${DATA_NAME_LIST}" \
	--save_path="${TEST_DATA_SAVE_PATH}" \
	--config.checkpoint_dir="./ckpts/square_full_it_${TIMESTAMP%+*}" \
	--jaxline_mode="train_eval_multithreaded" \
	--alsologtostderr="true"
