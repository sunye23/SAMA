#!/usr/bin/env bash
#set -x
export PYTHONPATH=/PATH/TO/SAMA:$PYTHONPATH
NAME=sama_1b
FILE=train
CONFIG=./projects/llava_sam2/configs/sama_1b_base.py
HOME_WORK_DIR=/SAMA/work_dirs/sama_1b
GPUs=8
WORK_DIR="${HOME_WORK_DIR}/${NAME}"
mkdir -p "$WORK_DIR"

output_logs="${WORK_DIR}/output.log"
echo "Output logs at: $output_logs"


bash ./tools/dist_st.sh "$FILE" "$CONFIG" "$NAME" "$WORK_DIR" "$GPUs" \
    >> "$output_logs" 2>&1
