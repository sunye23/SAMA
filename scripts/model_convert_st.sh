#!/usr/bin/env bash
export PYTHONPATH=/PATH/TO/SAMA:$PYTHONPATH
SCRIPT_PATH="./projects/llava_sam2/hf/convert_to_hf_st.py"

MODEL_DIRS=(
  "/path/to/sama_1b/iter_xxx.pth"
)

CONFIG_FILES=(
  "/path/to/sama_1b/sama_1b_base.py"
)

for idx in "${!MODEL_DIRS[@]}"; do
    MODEL_DIR="${MODEL_DIRS[$idx]}"
    CONFIG_FILE="${CONFIG_FILES[$idx]}"

    BASE_DIR="$(dirname "$MODEL_DIR")"
    BASE_FOLDER_NAME="$(basename "$BASE_DIR")"
    FILE_NAME="$(basename "$MODEL_DIR" .pth)"
    OUTPUT_DIR="$(dirname "$BASE_DIR")/${BASE_FOLDER_NAME}/${BASE_FOLDER_NAME}_${FILE_NAME}"

    mkdir -p "$OUTPUT_DIR"

    echo "BASE_DIR:         $BASE_DIR"
    echo "BASE_FOLDER_NAME: $BASE_FOLDER_NAME"
    echo "FILE_NAME:        $FILE_NAME"
    echo "OUTPUT_DIR:       $OUTPUT_DIR"
    echo "CONFIG_FILE:      $CONFIG_FILE"

    python "$SCRIPT_PATH" \
        --config "$CONFIG_FILE" \
        --pth_model "$MODEL_DIR" \
        --save_path "$OUTPUT_DIR"
done
