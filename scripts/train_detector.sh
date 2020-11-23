NOW=$(date +"%x %r %Z")
echo "Time: ${NOW}"
DETECTION_DATA_PATH=$1
BACKBONE=$2
BASE_DIR="$(dirname $(dirname $(readlink -f $0)))"
DETECTOR_SCRIPT="${BASE_DIR}/src/sagemaker_defect_detection/detector.py"
LOG_DIR="${BASE_DIR}/logs"
MFN_LOGS="${LOG_DIR}/classification_logs"
RPN_LOGS="${LOG_DIR}/rpn_logs"
ROI_LOGS="${LOG_DIR}/roi_logs"
FINETUNED_RPN_LOGS="${LOG_DIR}/finetune_rpn_logs"
FINETUNED_ROI_LOGS="${LOG_DIR}/finetune_roi_logs"
FINETUNED_FINAL_LOGS="${LOG_DIR}/finetune_final_logs"
EXTRA_FINETUNED_RPN_LOGS="${LOG_DIR}/extra_finetune_rpn_logs"
EXTRA_FINETUNED_ROI_LOGS="${LOG_DIR}/extra_finetune_roi_logs"
EXTRA_FINETUNING_STEPS=3

function find_best_ckpt() {
    python ${BASE_DIR}/scripts/find_best_ckpt.py $1 $2
}

function train_step() {
    echo "training step $1"
    case $1 in
    "1")
        echo "skipping step 1 and use 'train_classifier.sh'"
        ;;
    "2") # train rpn
        python ${DETECTOR_SCRIPT} \
            --data-path=${DETECTION_DATA_PATH} \
            --backbone=${BACKBONE} \
            --train-rpn \
            --pretrained-mfn-ckpt=$(find_best_ckpt "${MFN_LOGS}" "max") \
            --save-path=${RPN_LOGS} \
            --gpus=-1 --distributed-backend=ddp \
            --epochs=100
        ;;
    "3") # train roi
        python ${DETECTOR_SCRIPT} \
            --data-path=${DETECTION_DATA_PATH} \
            --backbone=${BACKBONE} \
            --train-roi \
            --pretrained-rpn-ckpt=$(find_best_ckpt "${RPN_LOGS}" "min") \
            --save-path=${ROI_LOGS} \
            --gpus=-1 --distributed-backend=ddp \
            --epochs=100
        ;;
    "4") # finetune rpn
        python ${DETECTOR_SCRIPT} \
            --data-path=${DETECTION_DATA_PATH} \
            --backbone=${BACKBONE} \
            --finetune-rpn \
            --pretrained-rpn-ckpt=$(find_best_ckpt "${RPN_LOGS}" "min") \
            --pretrained-roi-ckpt=$(find_best_ckpt "${ROI_LOGS}" "min") \
            --save-path=${FINETUNED_RPN_LOGS} \
            --gpus=-1 --distributed-backend=ddp \
            --learning-rate=1e-4 \
            --epochs=100
        # --resume-from-checkpoint=$(find_best_ckpt "${FINETUNED_RPN_LOGS}" "max")
        ;;
    "5") # finetune roi
        python ${DETECTOR_SCRIPT} \
            --data-path=${DETECTION_DATA_PATH} \
            --backbone=${BACKBONE} \
            --finetune-roi \
            --finetuned-rpn-ckpt=$(find_best_ckpt "${FINETUNED_RPN_LOGS}" "max") \
            --pretrained-roi-ckpt=$(find_best_ckpt "${ROI_LOGS}" "min") \
            --save-path=${FINETUNED_ROI_LOGS} \
            --gpus=-1 --distributed-backend=ddp \
            --learning-rate=1e-4 \
            --epochs=100
        # --resume-from-checkpoint=$(find_best_ckpt "${FINETUNED_ROI_LOGS}" "max")
        ;;
    "extra_rpn") # initially EXTRA_FINETUNED_*_LOGS is a copy of FINETUNED_*_LOGS
        python ${DETECTOR_SCRIPT} \
            --data-path=${DETECTION_DATA_PATH} \
            --backbone=${BACKBONE} \
            --finetune-rpn \
            --finetuned-rpn-ckpt=$(find_best_ckpt "${EXTRA_FINETUNED_RPN_LOGS}" "max") \
            --finetuned-roi-ckpt=$(find_best_ckpt "${EXTRA_FINETUNED_ROI_LOGS}" "max") \
            --save-path=${EXTRA_FINETUNED_RPN_LOGS} \
            --gpus=-1 --distributed-backend=ddp \
            --learning-rate=1e-4 \
            --epochs=100
        ;;
    "extra_roi") # initially EXTRA_FINETUNED_*_LOGS is a copy of FINETUNED_*_LOGS
        python ${DETECTOR_SCRIPT} \
            --data-path=${DETECTION_DATA_PATH} \
            --backbone=${BACKBONE} \
            --finetune-roi \
            --finetuned-rpn-ckpt=$(find_best_ckpt "${EXTRA_FINETUNED_RPN_LOGS}" "max") \
            --finetuned-roi-ckpt=$(find_best_ckpt "${EXTRA_FINETUNED_ROI_LOGS}" "max") \
            --save-path=${EXTRA_FINETUNED_ROI_LOGS} \
            --gpus=-1 --distributed-backend=ddp \
            --learning-rate=1e-4 \
            --epochs=100
        ;;
    "joint") # final
        python ${DETECTOR_SCRIPT} \
            --data-path=${DETECTION_DATA_PATH} \
            --backbone=${BACKBONE} \
            --finetuned-rpn-ckpt=$(find_best_ckpt "${EXTRA_FINETUNED_RPN_LOGS}" "max") \
            --finetuned-roi-ckpt=$(find_best_ckpt "${EXTRA_FINETUNED_ROI_LOGS}" "max") \
            --save-path="${FINETUNED_FINAL_LOGS}" \
            --gpus=-1 --distributed-backend=ddp \
            --learning-rate=1e-3 \
            --epochs=300
        # --resume-from-checkpoint=$(find_best_ckpt "${FINETUNED_FINAL_LOGS}" "max")
        ;;

    *) ;;
    esac
}

function train_wait_to_finish() {
    train_step $1 &
    BPID=$!
    wait $BPID
}

function run() {
    if [ "$1" != "" ]; then
        train_step $1
    else
        nvidia-smi | grep python | awk '{ print $3 }' | xargs -n1 kill -9 >/dev/null 2>&1
        read -p "Training all steps from scratch? (Y/N): " confirm && [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]
        if [ "$confirm" == "Y" ]; then
            for i in {1..5}; do
                train_wait_to_finish $i
            done
            echo "finished all the training steps"
            mkdir -p "${EXTRA_FINETUNED_RPN_LOGS}" && cp -r "${FINETUNED_RPN_LOGS}/"* "${EXTRA_FINETUNED_RPN_LOGS}"
            mkdir -p "${EXTRA_FINETUNED_ROI_LOGS}" && cp -r "${FINETUNED_ROI_LOGS}/"* "${EXTRA_FINETUNED_ROI_LOGS}"
        fi
        echo "repeating extra finetuning steps ${EXTRA_FINETUNING_STEPS} more times"
        for i in {1..${EXTRA_FINETUNING_STEPS}}; do
            train_wait_to_finish "extra_rpn"
            train_wait_to_finish "extra_roi"
        done
        echo "final joint training"
        train_step "joint"
    fi
    exit 0
}

run $1
