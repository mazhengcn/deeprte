#!/usr/bin/bash
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
# Download RTE datasets from math-02 server
#
# Usage: bash download_rte_datasets.sh /path/to/download/directory
set -e

SERVER_URL=${1:-"xuzhiqin_02@202.120.13.117"}
REMOTE_DATA_DIR=${2:-"/cluster/home/xuzhiqin_02/rte_data/"}
DOWNLOAD_DIR=${3:-"data/matlab/"}

rsync -rlptzv --progress --delete --exclude=.git "${SERVER_URL}:${REMOTE_DATA_DIR}" "${DOWNLOAD_DIR}"
