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

DATA_DIR=${1:-"/root/projects/deeprte/data/raw_data/train/g0.1-sigma_a3-sigma_t6"}
DATA_FILENAMES=${2:-"g0.1-sigma_a3-sigma_t6_eval.mat"}
MODEL_DIR=${3:-"/root/projects/deeprte/data/ckpts/g0.1-sigma_a3-sigma_t6_2023-05-14T18:50:04/models/latest/step_500000_2023-05-23T12:05:36"}
CUDA_DEVICES=${4:-""}

if [ -n "${CUDA_DEVICES}" ]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
fi

TIMESTAMP="$(date --iso-8601="seconds")"

python run_deeprte.py \
    --output_dir="test/${TIMESTAMP%+*}" \
    --data_dir="${DATA_DIR}" \
    --data_filenames="${DATA_FILENAMES}" \
    --model_dir="${MODEL_DIR}" \