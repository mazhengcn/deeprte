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

CONFIG_PATH=${1:-"./configs/g0.5.yaml"}
CKPT_DIR=${2:-"./data/interim/ckpts/v1.0.1/g0.5"}

python run_train.py --config=${CONFIG_PATH} --workdir=${CKPT_DIR}
