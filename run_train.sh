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

TIMESTAMP="$(date --iso-8601="seconds")"
SOURCE_DIR=${1:-"./data/tfds"}
# TEST_DATA_SAVE_PATH=${3:-"./data/test_data/train_random_kernel.npz"}
python run_deeprte.py \
	--config=deeprte/config.py \
	--tfds_dir="${SOURCE_DIR}" \
	--config.checkpoint_dir="./ckpts/square_full_it_${TIMESTAMP%+*}" \
	# --config.restore_dir="./ckpts/square_full_it_2023-04-01T21:39:01/models/latest/step_75000_2023-04-02T18:25:34" \
	--jaxline_mode="train" \
	--alsologtostderr="true"
