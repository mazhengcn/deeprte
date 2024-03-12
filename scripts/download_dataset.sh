#!/bin/bash

# Set data save directory and destination host directory
DATA_SAVE_DIR='/nfs/my/projects/deeprte/data'
DESTINATION_HOST_DIR='matjxt-mz@sydata.hpc.sjtu.edu.cn:/dssg/home/acct-matjxt/matjxt-mz/data/rte_data/raw_data'

DESTINATION_HOST=${DESTINATION_HOST_DIR%:*}
DIR=${DESTINATION_HOST_DIR#*:}

ssh "${DESTINATION_HOST}" ls "${DIR}" > remote_list.txt
cat remote_list.txt | xargs --max-args=1 --max-procs=10 --replace=% rsync -r --archive --partial "${DESTINATION_HOST_DIR}/%" "${DATA_SAVE_DIR}"
