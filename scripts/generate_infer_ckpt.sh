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

TRAIN_STATE_DIR=${1:-"./data/interim/ckpts/v1.0.1/S_v4/200000"}
CKPT_DIR=${2:-"./models/v1.0.1/S_v4"}

python generate_param_only_checkpoint.py \
    --train_state_dir=${TRAIN_STATE_DIR} \
    --checkpoint_dir=${CKPT_DIR}
