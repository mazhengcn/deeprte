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
DATA_DIR=${2:-"data/raw_data/g0.1-sigma_a3-sigma_t6"}
DATA_FILENAMES=${3:-"g0.1-sigma_a3-sigma_t6.mat"}
MODEL_DIR=${4:-"ckpts/g0.5-sigma_a3-sigma_t6_2023-05-11T22:23:28/models/latest/step_300_2023-05-11T22:29:10"}

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" python run_deeprte.py \
    --output_dir="results" \
    --data_dir="data/raw_data/g0.1-sigma_a3-sigma_t6" \
    --data_filenames="g0.1-sigma_a3-sigma_t6.mat" \
    --model_dir="ckpts/g0.5-sigma_a3-sigma_t6_2023-05-11T22:23:28/models/latest/step_300_2023-05-11T22:29:10"
