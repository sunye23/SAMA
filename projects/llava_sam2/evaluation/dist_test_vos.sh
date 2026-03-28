#!/usr/bin/env bash

FILE=$1
MODEL=$2
GPUS=$3
DATASET=$4
WORK_DIR=$5
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export PYTHONPATH=/Sa2VA/reproduce/Sa2VA:$PYTHONPATH

if command -v torchrun &> /dev/null
then
  echo "Using torchrun mode."
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    torchrun --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=25034 \
    --nproc_per_node=${GPUS} \
    ${FILE} ${MODEL} --launcher pytorch --work_dir ${WORK_DIR} --dataset ${DATASET} "${@:6}"
else
  echo "Using launch mode."
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -m torch.distributed.launch \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=25034 \
    --nproc_per_node=${GPUS} \
    ${FILE} ${MODEL} --launcher pytorch --work_dir ${WORK_DIR} --dataset ${DATASET}  "${@:6}"
fi

