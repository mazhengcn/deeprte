#!/bin/bash

# Set config and generator paths
CONFIG_DIR='/workspaces/deeprte/generator/2d-sweeping/configs'
GENERATOR_PATH='/workspaces/deeprte/generator/2d-sweeping'

# Set data save directory and destination host directory
DATA_SAVE_DIR='/workspaces/deeprte/generator/data/raw_data'
DESTINATION_HOST_DIR='matjxt-mz@sydata.hpc.sjtu.edu.cn:/dssg/home/acct-matjxt/matjxt-mz/data/rte_data/raw_data'

# Generate data
GEN_CONFIG_PATH="${GENERATOR_PATH}/config.m"
if [ -d "${CONFIG_DIR}" ]; then
    for file in "${CONFIG_DIR}"/*; do
        echo "Generating data with config file: ${file}"
        cp "${file}" "${GEN_CONFIG_PATH}"

        matlab -nodisplay -r "run ${GENERATOR_PATH}/generator.m; exit"

        rm -f "${GEN_CONFIG_PATH}"
    done
else
    echo "Error: Config directory ${CONFIG_DIR} does not exist"
    exit 1
fi

# Use rsync to copy data to destination host
sync_data () {
    local DATA_SAVE_DIR="$1"
    local DESTINATION_HOST_DIR="$2"

    if [ -d "${DATA_SAVE_DIR}" ]; then
        ls "${DATA_SAVE_DIR}" > remote_list.txt
        cat remote_list.txt | xargs --max-args=1 --max-procs=10 --replace=% rsync -r --archive --partial "${DATA_SAVE_DIR}/%" "${DESTINATION_HOST_DIR}"
    else
        echo "Error: Data save directory ${DATA_SAVE_DIR} does not exist"
        exit 1
    fi
}

sync_data "${DATA_SAVE_DIR}" "${DESTINATION_HOST_DIR}/train"
# sync_data "${DATA_SAVE_DIR}/isotropic" "${DESTINATION_HOST_DIR}/isotropic"
