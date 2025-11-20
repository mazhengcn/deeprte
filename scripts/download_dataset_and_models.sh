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

DATA_DIR=${1:-"./data"}
MODELS_DIR=${2:-"./models"}

DATASET_REPO="mazhengcn/rte-dataset"
MODEL_REPO="mazhengcn/deeprte"

# Download rte-dataset
hf download mazhengcn/rte-dataset \
    --exclude=interim/* \
    --local-dir=${DATA_DIR} \
    --repo-type=dataset

# Download models
hf download mazhengcn/deeprte \
    --local-dir=${MODELS_DIR} \
    --repo-type=model
