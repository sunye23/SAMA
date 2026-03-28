#!/usr/bin/env bash
set -x

EXP_PATH="/SAMA/data/video_datas/davis17/meta_expressions/valid/meta_expressions.json"
MASK_PATH="/SAMA/data/video_datas/davis17/valid/mask_dict.pkl"
PRED_PATH="sama_1b/DAVIS/Annotations"
SAVE_FILE="${PRED_PATH}/davis17_result.json"

export PYTHONPATH=/SAMA:$PYTHONPATH

python SAMA/tools/eval_video_seg/eval_davis17.py \
  --mevis_exp_path "$EXP_PATH" \
  --mevis_mask_path "$MASK_PATH" \
  --mevis_pred_path "$PRED_PATH" \
  --save_name "$SAVE_FILE"