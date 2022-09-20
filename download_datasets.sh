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
#
# Download RTE datasets from sy-sjtu server
#
# Usage: bash download_datasets.sh /path/to/download/directory
set -e

SERVER_URL=${1:-"matjxt-mz@sydata.hpc.sjtu.edu.cn"}
REMOTE_DATA_DIR=${2:-"/dssg/home/acct-matjxt/matjxt-mz/data/rte_data/"}
DOWNLOAD_DIR=${3:-"./data/"}

# Password: 2&gpTKPd
rsync -rlptzv --progress --delete --exclude=.git "${SERVER_URL}:${REMOTE_DATA_DIR}" "${DOWNLOAD_DIR}"
