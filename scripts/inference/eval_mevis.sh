#!/usr/bin/env bash
set -x

MODEL="/home/ubuntu/sunye/project/video_project/SAMA/work_dirs/sama_1b"

export PYTHONPATH=/home/ubuntu/sunye/project/video_project/SAMA:$PYTHONPATH
FILE="/home/ubuntu/sunye/project/video_project/SAMA/projects/llava_sam2/evaluation/ref_vos_eval_st.py"

GPUS=8

TOTAL_RESULTS_FILE="${MODEL}/total_results.txt"
DATASETS=("MEVIS")

for DATASET in "${DATASETS[@]}"; do

    WORK_DIR="${MODEL}/${DATASET}"
    mkdir -p "$WORK_DIR"

    output_logs="${WORK_DIR}/output_${DATASET}.log"
    echo "Processing dataset: $DATASET, logs: $output_logs"

    bash /home/ubuntu/sunye/project/video_project/SAMA/projects/llava_sam2/evaluation/dist_test_vos.sh \
         "$FILE" "$MODEL" "$GPUS" "$DATASET" "$WORK_DIR" \
         >> "$output_logs" 2>&1

    if [[ "$DATASET" == "DAVIS" ]]; then
        echo "DAVIS:" >> "$TOTAL_RESULTS_FILE"
        EXP_PATH="/home/ubuntu/sunye/project/video_project/SAMA/data/video_datas/davis17/meta_expressions/valid/meta_expressions.json"
        MASK_PATH="/home/ubuntu/sunye/project/video_project/SAMA/data/video_datas/davis17/valid/mask_dict.pkl"
        PRED_PATH="${MODEL}/DAVIS/Annotations"
        SAVE_FILE="${PRED_PATH}/davis17_result.json"

        python /home/ubuntu/sunye/project/video_project/SAMA/tools/eval_video_seg/eval_davis17.py \
          --mevis_exp_path "$EXP_PATH" \
          --mevis_mask_path "$MASK_PATH" \
          --mevis_pred_path "$PRED_PATH" \
          --save_name "$SAVE_FILE" \
          >> "$TOTAL_RESULTS_FILE" 2>&1
    elif [[ "$DATASET" == "MEVIS_U" ]]; then
        echo "MEVIS_U:" >> "$TOTAL_RESULTS_FILE"
        EXP_PATH="/home/ubuntu/sunye/project/video_project/SAMA/data/video_datas/mevis/valid_u/meta_expressions.json"
        MASK_PATH="/home/ubuntu/sunye/project/video_project/SAMA/data/video_datas/mevis/valid_u/mask_dict.json"
        PRED_PATH="${MODEL}/MEVIS_U/Annotations"
        SAVE_FILE="${PRED_PATH}/mevis_result.json"

        python /home/ubuntu/sunye/project/video_project/SAMA/tools/eval_video_seg/eval_mevis.py \
          --mevis_exp_path "$EXP_PATH" \
          --mevis_mask_path "$MASK_PATH" \
          --mevis_pred_path "$PRED_PATH" \
          --save_name "$SAVE_FILE" \
          >> "$TOTAL_RESULTS_FILE" 2>&1
      elif [[ "$DATASET" == "REVOS" ]]; then
          echo "REVOS:" >> "$TOTAL_RESULTS_FILE"
          EXP_PATH="/home/ubuntu/sunye/project/video_project/SAMA/data/video_datas/revos/meta_expressions_valid_.json"
          MASK_PATH="/home/ubuntu/sunye/project/video_project/SAMA/data/video_datas/revos/mask_dict.json"
          FG_PATH="/home/ubuntu/sunye/project/video_project/SAMA/data/video_datas/revos/mask_dict_foreground.json"

          PRED_PATH="${MODEL}/REVOS/Annotations"
          SAVE_JSON_FILE="${PRED_PATH}/revos_result.json"
          SAVE_CSV_FILE="${PRED_PATH}/revos_result.csv"

          python /home/ubuntu/sunye/project/video_project/SAMA/tools/eval_video_seg/eval_revos.py \
            --visa_pred_path "$PRED_PATH" \
            --visa_exp_path "$EXP_PATH" \
            --visa_mask_path "$MASK_PATH" \
            --visa_foreground_mask_path "$FG_PATH" \
            --save_json_name "$SAVE_JSON_FILE" \
            --save_csv_name  "$SAVE_CSV_FILE"
      fi
done

