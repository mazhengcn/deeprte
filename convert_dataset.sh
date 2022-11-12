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

SOURCE_DIR=${1:-"./data/rte_data/matlab/1030/"}
DATAFILES=${2:-"test_sin.mat"}
SAVE_PATH=${3:-"./data/train/1030/test_sin.npz"}

python deeprte/data_convert.py \
	--source_dir="${SOURCE_DIR}" \
	--datafiles="${DATAFILES}" \
	--save_path="${SAVE_PATH}"
