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

SOURCE_DIR=${1:-"./data/rte_data/matlab/eval-data/scattering-kernel/fixed-kernel/"}
DATAFILES=${2:-"test_bc1.mat, test_sin_xv.mat"}
SAVE_PATH=${3:-"./data/train/scattering-kernel/test_fixed_kernel.mat"}

python deeprte/data_convert.py \
	--source_dir="${SOURCE_DIR}" \
	--datafiles="${DATAFILES}" \
	--save_path="${SAVE_PATH}"
