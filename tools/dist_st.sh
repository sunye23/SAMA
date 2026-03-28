#!/usr/bin/env bash

set -x

FILE=$1
CONFIG=$2
NAME=$3
WORK_DIR=$4
GPUs=$5
## ========== 关键部分：自动安全生成 MASTER_PORT ==========
#if [ -z "$MASTER_PORT" ]; then
#    if [ "${SLURM_PROCID}" -eq 0 ]; then
#        MASTER_PORT=$((12000 + RANDOM % 20000))  # 生成12000～31999的随机端口
#        echo "Generated MASTER_PORT=${MASTER_PORT} on rank 0"
#    fi
#    # 广播环境变量到所有节点（确保同步）
#    export MASTER_PORT
#fi
#echo "Using master port: ${MASTER_PORT}"
## ==========================================================
DEEPSPEED=${DEEPSPEED:-deepspeed_zero2}
# 兼容非 SLURM 环境
NNODES=${SLURM_NNODES:-1}
NODE_RANK=${SLURM_PROCID:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-25032}

echo "MASTER_ADDR=$MASTER_ADDR"
echo "NNODES=$NNODES"
echo "NODE_RANK=$NODE_RANK"

# 指定你的权重文件所在的工作目录
############### 查找和解析最新的 iter_xxx.pth 目录 ###############
# 方法1: 使用 find 来枚举所有匹配 iter_*.pth 的目录，再根据名字提取数字
ckpts=( $(find "$WORK_DIR" -maxdepth 1 -type d -name "iter_*.pth" 2>/dev/null) )

export PYTHONPATH=/Sa2VA/reproduce/Sa2VA:$PYTHONPATH

if [ ${#ckpts[@]} -eq 0 ]; then
  echo "No checkpoint directory (iter_*.pth) found in $WORK_DIR, continuing without resume..."
  RESUME_ARG=""
else
  best_iter=-1
  best_ckpt=""
  for ckpt_dir in "${ckpts[@]}"; do
    # ckpt_dir 形如 /path/to/iter_800.pth
    dir_name="$(basename "$ckpt_dir")"    # 例如 iter_800.pth
    # 去掉前缀 iter_，再去掉后缀 .pth，得到具体数字
    number="${dir_name#iter_}"           # -> 800.pth
    number="${number%.pth}"              # -> 800

    # 判断是否比当前 best_iter 更大
    if [ "$number" -gt "$best_iter" ]; then
      best_iter="$number"
      best_ckpt="$ckpt_dir"
    fi
  done

  echo "Found latest checkpoint directory: $best_ckpt (iter_$best_iter)"
  RESUME_ARG="--resume $best_ckpt"
fi
#
#if command -v torchrun &> /dev/null
#then
#  echo "Using torchrun mode."
#  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
#    torchrun --nproc_per_node=${GPUs} \
#    --nnodes=${SLURM_NNODES} \
#    --node_rank=${SLURM_PROCID} \
#    --master_addr=${MASTER_ADDR} \
#    --master_port=25032 \
#    tools/${FILE}.py \
#    ${CONFIG} \
#    $RESUME_ARG \
#    --launcher pytorch \
#    --work-dir "$WORK_DIR" \
#    --deepspeed $DEEPSPEED \
#    "${@:6}"
#else
#  echo "Using launch mode."
#  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
#    python -m torch.distributed.launch \
#    --nproc_per_node=${GPUs} \
#    --nnodes=${SLURM_NNODES} \
#    --node_rank=${SLURM_PROCID} \
#    --master_addr=${MASTER_ADDR} \
#    --master_port=25032 \
#    tools/${FILE}.py \
#    ${CONFIG} \
#    $RESUME_ARG \
#    --launcher pytorch \
#    --work-dir "$WORK_DIR" \
#    --deepspeed $DEEPSPEED \
#    "${@:6}"
#fi
if command -v torchrun &> /dev/null
then
  echo "Using torchrun mode."
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    torchrun --nproc_per_node=${GPUs} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    tools/${FILE}.py \
    ${CONFIG} \
    $RESUME_ARG \
    --launcher pytorch \
    --work-dir "$WORK_DIR" \
    --deepspeed $DEEPSPEED \
    "${@:6}"
else
  echo "Using launch mode."
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -m torch.distributed.launch \
    --nproc_per_node=${GPUs} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    tools/${FILE}.py \
    ${CONFIG} \
    $RESUME_ARG \
    --launcher pytorch \
    --work-dir "$WORK_DIR" \
    --deepspeed $DEEPSPEED \
    "${@:6}"
fi
