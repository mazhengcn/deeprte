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

RAW_DATA_DIR=${1:-"data/raw_data"}
TFDS_DIR=${2:-"data/tfds"}
REMOTE_HOST=${3:-"matjxt-mz@sydata.hpc.sjtu.edu.cn"}
REMOTE_DIR=${4:-"/dssg/home/acct-matjxt/matjxt-mz/data/rte_data/raw_data"}
OVERWRITE=${5:-"True"}

# Use rsync to copy data to destination host
rsync -rlptzv --archive --progress "${REMOTE_HOST}:${REMOTE_DIR}/" "${RAW_DATA_DIR}"

find "${RAW_DATA_DIR}/train" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; > deeprte/data/tfds/rte/CONFIGS.txt

TFDS_ARGS="--data_dir=${TFDS_DIR} --manual_dir=${RAW_DATA_DIR}/train"

if [ "${OVERWRITE}" = "True" ]; then
	TFDS_ARGS="${TFDS_ARGS} --overwrite"
fi

tfds build deeprte/data/tfds/rte ${TFDS_ARGS} --file_format="array_record"
