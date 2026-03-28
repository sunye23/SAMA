#!/usr/bin/env bash
set -x
PRED_PATH="SAMA-1B/MEVIS/Annotations"
OUT_ZIP_PATH="SAMA-1B/MEVIS/Annotations.zip"

python ./tools/eval_video_seg/zip_mp_mevis.py "$PRED_PATH" "$OUT_ZIP_PATH"