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

DATA_PATH=${1:-"./data/raw/test/sin-rv-g0.5-amplitude5-wavenumber10/sin-rv-g0.5-amplitude5-wavenumber10.mat"}
MODEL_DIR=${2:-"./models/v1.0.1/g0.5"}
OUTPUT_DIR=${3:-"./outputs"}

TIMESTAMP="$(date --iso-8601="seconds")"

python run_deeprte.py \
  --model_dir="${MODEL_DIR}" \
  --data_path="${DATA_PATH}" \
  --output_dir="${OUTPUT_DIR}/${TIMESTAMP%+*}"
