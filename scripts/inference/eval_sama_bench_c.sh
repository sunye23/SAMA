#!/usr/bin/env bash
set -e

export PYTHONPATH=/Sa2VA/reproduce/Sa2VA:$PYTHONPATH
export OPENAI_API_KEY=""

#MODEL_=$1
#MAX_FRAMES=$2
MODEL=/home/ubuntu/sunye/project/video_project/Sa2VA_NVIDIA/work_dirs/SAMA_1B
FILE=/Sa2VA/reproduce/Sa2VA/projects/llava_sam2/evaluation/video_region_cap_eval_video_wise.py
GPUS=8
MAX_FRAMES=32

SAVE_DIR="${MODEL}/video_region_cap_pred"
mkdir -p "$SAVE_DIR"

JSON_DIR=/Sa2VA/reproduce/Sa2VA/video_region_meta/benchmark_split
JSON_FILES=(
    mevis.json
    lvvis_0.json
    lvvis_1.json
    ref_youtube_vos_0.json
    ref_youtube_vos_1.json
    VidSTG_0.json
    VidSTG_1.json
    VidSTG_2.json
    VidSTG_3.json
    VidSTG_4.json
    VidSTG_5.json
    VidSTG_6.json
    VidSTG_7.json
)

# 多 JSON 逐个处理
for json_file in "${JSON_FILES[@]}"; do
    FULL_PATH="${JSON_DIR}/${json_file}"
    BASENAME=$(basename "$json_file" .json)
    LOG_FILE="${MODEL}/output_video_cap_${BASENAME}.log"
    FINAL_METRIC_FILE="${MODEL}/video_caption_final_metrics/${BASENAME}_final.json"

    if [ -f "$FINAL_METRIC_FILE" ]; then
        echo "[SKIP] ${json_file} 已处理完毕，跳过。"
        continue
    fi

    echo ">>> Processing: $FULL_PATH"
    echo ">>> Logging to: $LOG_FILE"

    bash /Sa2VA/reproduce/Sa2VA/projects/llava_sam2/evaluation/dist_test_video_gcg.sh \
        "$FILE" \
        "$MODEL" \
        "$GPUS" \
        "$MAX_FRAMES" \
        --save_dir "$SAVE_DIR" \
        --json_file "$FULL_PATH" \
        >> "$LOG_FILE" 2>&1
done