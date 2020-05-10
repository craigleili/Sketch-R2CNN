CL_DATE=`date '+%y-%m-%d_%H-%M'`

#CL_MODEL="densenet161"
#CL_MODEL="resnet101"
CL_MODEL="resnet50"
#CL_MODEL="sketchanet"

CL_CKPT_PREFIX="quickdraw_${CL_MODEL}"

CL_DATASET="quickdraw"
CL_DATASET_ROOT="<QuickDraw_Root>"
CL_LOG_DIR="<Log_Root>"

CL_RUNNAME="${CL_DATE}-${CL_DATASET}-r2cnn-${CL_MODEL}"
mkdir "${CL_LOG_DIR}/${CL_RUNNAME}"

sudo nvidia-docker run --rm \
    --network=host \
    --shm-size 8G \
    -v /:/host \
    -v /tmp/torch_extensions:/tmp/torch_extensions \
    -v /tmp/torch_models:/root/.torch \
    -w "/host$PWD" \
    -e PYTHONUNBUFFERED=x \
    -e CUDA_CACHE_PATH=/host/tmp/cuda-cache \
    craigleili/sketch-r2cnn:latest \
    python quickdraw_r2cnn_train.py \
        --ckpt_prefix "/host${CL_LOG_DIR}/${CL_CKPT_PREFIX}" \
        --dataset_fn ${CL_DATASET} \
        --dataset_root "/host${CL_DATASET_ROOT}" \
        --intensity_channels 8 \
        --log_dir "/host${CL_LOG_DIR}/${CL_RUNNAME}" \
        --model_fn ${CL_MODEL} \
        --num_epochs 1 \
    2>&1 | tee -a "${CL_LOG_DIR}/${CL_RUNNAME}/train.log"
