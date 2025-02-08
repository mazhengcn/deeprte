set -e

MODEL_DIR=${1:-"/workspaces/deeprte/ckpts/g0.1-gaussian-source0121"}
DATA_PATH=${2:-"/workspaces/deeprte/data/raw_data/test/source-g0.1-qconstant/g0.1-qconstant.npz"}
OUTPUT_DIR=${3:-"/workspaces/deeprte/test"}

TIMESTAMP="$(date --iso-8601="seconds")"

python data_postprocess.py \
    --model_path="${MODEL_DIR}" \
    --data_path="${DATA_PATH}" \
    --output_dir="${OUTPUT_DIR}/${TIMESTAMP%+*}"


