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

SOURCE_DIR=${1:-"data/matlab/"}
DATAFILES=${2:-"e1_L_delta_1.mat,e1_R_delta_1.mat,e1_B_delta_1.mat,e1_T_delta_1.mat"}
SAVE_PATH=${3:-"data/train/square_full_1.npz"}

python deeprte/data_adapter.py \
	--source_dir="${SOURCE_DIR}" \
	--datafiles="${DATAFILES}" \
	--save_path="${SAVE_PATH}"
