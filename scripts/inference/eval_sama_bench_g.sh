#!/usr/bin/env bash
set -x
export PYTHONPATH=/home/ubuntu/sunye/project/video_project/Sa2VA_NVIDIA:$PYTHONPATH
export OPENAI_API_KEY=""

MODEL=/home/ubuntu/sunye/project/video_project/Sa2VA_NVIDIA/work_dirs/SAMA_1B
MAX_FRAMES=32

FILE=/home/ubuntu/sunye/project/video_project/Sa2VA_NVIDIA/projects/llava_sam2/evaluation/video_gcg_eval_video_wise.py

# 推理保存目录
SAVE_DIR="${MODEL}/video_gcg_pred_video_wise"
mkdir -p "$SAVE_DIR"
# GPU配置
GPUS=8

# 数据目录与JSON列表
JSON_DIR=/home/ubuntu/sunye/project/video_project/Sa2VA_NVIDIA/video_region_meta/benchmark_split
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

# 循环逐个处理每个子JSON文件
for json_file in "${JSON_FILES[@]}"; do
    FULL_PATH="${JSON_DIR}/${json_file}"
    BASENAME=$(basename "$json_file" .json)
    LOG_FILE="${MODEL}/output_video_GCG_${BASENAME}.log"
    FINAL_METRIC_FILE="${MODEL}/video_gcg_eval_metrics/video_gcg_eval_${BASENAME}_final.json"

    # ✅ 跳过已处理过的文件
    if [ -f "$FINAL_METRIC_FILE" ]; then
        echo "[SKIP] ${json_file} 已处理完毕，跳过。"
        continue
    fi
    echo ">>> Processing: $FULL_PATH"
    echo ">>> Logging to: $LOG_FILE"

    bash ./projects/llava_sam2/evaluation/dist_test_video_gcg.sh \
        "$FILE" \
        "$MODEL" \
        "$GPUS" \
        "$MAX_FRAMES" \
        --save_dir "$SAVE_DIR" \
        --json_file "$FULL_PATH" \
        >> "$LOG_FILE" 2>&1
done


