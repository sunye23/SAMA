#!/usr/bin/env bash
set -x

EXP_PATH="Sa2VA/data/video_datas/davis17/meta_expressions/valid/meta_expressions.json"
MASK_PATH="Sa2VA/data/video_datas/davis17/valid/mask_dict.pkl"
PRED_PATH="PATH/to/DAVIS/Annotations"
SAVE_FILE="${PRED_PATH}/davis17_result.json"

python ./tools/eval_video_seg/eval_davis17.py \
  --mevis_exp_path "$EXP_PATH" \
  --mevis_mask_path "$MASK_PATH" \
  --mevis_pred_path "$PRED_PATH" \
  --save_name "$SAVE_FILE"