#!/usr/bin/env bash
set -x  
EXP_PATH="SAMA/data/video_datas/revos/meta_expressions_valid_.json"
MASK_PATH="SAMA/data/video_datas/revos/mask_dict.json"
FG_PATH="SAMA/data/video_datas/revos/mask_dict_foreground.json"

PRED_PATH="SAMA/Sa2VA-8B/REVOS/Annotations"
SAVE_JSON_FILE="${PRED_PATH}/revos_result.json"
SAVE_CSV_FILE="${PRED_PATH}/revos_result.csv"

export PYTHONPATH=SAMA:$PYTHONPATH

python SAMA/tools/eval_video_seg/eval_revos.py \
  --visa_pred_path "$PRED_PATH" \
  --visa_exp_path "$EXP_PATH" \
  --visa_mask_path "$MASK_PATH" \
  --visa_foreground_mask_path "$FG_PATH" \
  --save_json_name "$SAVE_JSON_FILE" \
  --save_csv_name  "$SAVE_CSV_FILE"
