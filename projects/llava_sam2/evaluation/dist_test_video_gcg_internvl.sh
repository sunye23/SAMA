#!/usr/bin/env bash
# 以下 3 个参数由外部脚本 eval_gcg.sh 传进来
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
FILE=$1    # Python脚本路径 (gcg_eval.py)
MODEL=$2   # 模型路径 (例如 /lustre/.../Sa2VA-1B)
GPUS=$3    # GPU 数量
MAX_FRAMES=$4    # GPU 数量
MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
echo "Using dynamically assigned master port: ${MASTER_PORT}"
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# 保证 Python 能找到你的项目路径
export PYTHONPATH=/Sa2VA/reproduce/Sa2VA:$PYTHONPATH

if command -v torchrun &> /dev/null
then
  echo "Using torchrun mode."
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    torchrun --nnodes=${NNODES} \
             --node_rank=${NODE_RANK} \
             --master_addr=${MASTER_ADDR} \
             --master_port=${MASTER_PORT} \
             --nproc_per_node=${GPUS} \
             ${FILE} \
             --model_path "${MODEL}" \
             --max_frames "${MAX_FRAMES}" \
             --launcher pytorch \
             "${@:5}"
else
  echo "Using launch mode."
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -m torch.distributed.launch \
           --nnodes=${NNODES} \
           --node_rank=${NODE_RANK} \
           --master_addr=${MASTER_ADDR} \
           --master_port=${MASTER_PORT} \
           --nproc_per_node=${GPUS} \
           ${FILE} \
           --model_path "${MODEL}" \
           --max_frames "${MAX_FRAMES}" \
           --launcher pytorch \
           "${@:5}"
fi
