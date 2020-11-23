CLASSIFICATION_DATA_PATH=$1
BACKBONE=$2
BASE_DIR="$(dirname $(dirname $(readlink -f $0)))"
CLASSIFIER_SCRIPT="${BASE_DIR}/src/sagemaker_defect_detection/classifier.py"
MFN_LOGS="${BASE_DIR}/logs/"

python ${CLASSIFIER_SCRIPT} \
    --data-path=${CLASSIFICATION_DATA_PATH} \
    --save-path=${MFN_LOGS} \
    --backbone=${BACKBONE} \
    --gpus=1 \
    --learning-rate=1e-3 \
    --epochs=50
