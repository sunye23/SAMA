#!/usr/bin/env bash
set -x

EXP_PATH="/SAMA/data/video_datas/mevis/valid_u/meta_expressions.json"
MASK_PATH="/SAMA/data/video_datas/mevis/valid_u/mask_dict.json"
PRED_PATH="/SAMA/work_dirs/sama_1b/MEVIS_U/Annotations"

SAVE_FILE="${PRED_PATH}/mevis_result.json"

export PYTHONPATH=/SAMA:$PYTHONPATH

python ./tools/eval_video_seg/eval_mevis.py \
  --mevis_exp_path $EXP_PATH \
  --mevis_mask_path $MASK_PATH \
  --mevis_pred_path $PRED_PATH \
  --save_name "$SAVE_FILE"
