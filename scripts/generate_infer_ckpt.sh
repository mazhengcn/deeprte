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

TRAIN_STATE_DIR=${1:-"/workspaces/deeprte/ckpts/g0.1-gaussian-rte0202/141000"}
CKPT_DIR=${2:-"/workspaces/deeprte/ckpts/g0.1-gaussian-rte0202/infer"}

TRAIN_CKPT_DIR="$(dirname "${TRAIN_STATE_DIR}")"
cp $TRAIN_CKPT_DIR/config.yaml $TRAIN_CKPT_DIR/temp.yaml
echo "load_full_state_path: ${TRAIN_STATE_DIR}/train_state" >> $TRAIN_CKPT_DIR/temp.yaml

python deeprte/train_lib/generate_param_only_checkpoint.py \
    --config=${TRAIN_CKPT_DIR}/temp.yaml \
    --checkpoint_dir=${CKPT_DIR}

rm -f $TRAIN_CKPT_DIR/temp.yaml
