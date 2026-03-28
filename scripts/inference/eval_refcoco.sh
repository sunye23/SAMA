#!/usr/bin/env bash
set -x

FILE=/home/ubuntu/sunye/project/video_project/SAMA/projects/llava_sam2/evaluation/refcoco_eval_st.py
MODEL=/home/ubuntu/sunye/project/video_project/SAMA/work_dirs/sama_1b

GPUS=8
DATASET='refcoco'
WORK_DIR="${MODEL}/${DATASET}"
mkdir -p "$WORK_DIR"
output_logs="${WORK_DIR}/output.log"
echo "$output_logs"

# 3) 调用 dist.sh 并将输出重定向到 $output_logs
bash /home/ubuntu/sunye/project/video_project/SAMA/projects/llava_sam2/evaluation/dist_test_refcoco.sh "$FILE" "$MODEL" "$GPUS" "$DATASET" \
    >> "$output_logs" 2>&1