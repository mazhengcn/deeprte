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

export CUDA_VISIBLE_DEVICES="6"

python run_deeprte.py \
    --output_dir="results" \
    --data_dir="data/raw_data/eval_data/0311" \
    --data_filenames="test_random_kernel_0311.mat" \
    --model_dir="ckpts/saved_models/no_ln"
