set -e

RAW_DATA_DIR=${1:-"/workspaces/deeprte/data/raw_data/train/g0.1-gaussian-alpha5"}
TFDS_DIR=${2:-"/workspaces/deeprte/tfds-data"}
# GRAIN_DIR=${3:-"assets/grain"}

echo "RAW_DATA_DIR: ${RAW_DATA_DIR}"
echo "Checking contents of train directory:"

# find "${RAW_DATA_DIR}/train" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; > /usr/local/lib/python3.10/site-packages/rte_dataset/builders/tfds/rte/CONFIGS.txt

# find "${RAW_DATA_DIR}" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; > /workspaces/deeprte/notebooks/eff_rte/CONFIGS.txt

TFDS_ARGS="--data_dir=${TFDS_DIR} --manual_dir=${RAW_DATA_DIR}"

# Build tfrecord dataset
tfds build /workspaces/deeprte/notebooks/eff_rte ${TFDS_ARGS}