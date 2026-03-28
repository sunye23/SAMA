#!/bin/bash
JSON_FILE=split_3.json
python automatic_annotation.py \
  --json_path "/home/ubuntu/sunye/project/video_project/gemini_analysis/Benchmark/mevis_train/split/${JSON_FILE}" \
  --save_path "/home/ubuntu/sunye/project/video_project/gemini_analysis/Benchmark/mevis_train/mevis_annotated_${JSON_FILE}.json" \
  --skip_index_file_path "/home/ubuntu/sunye/project/video_project/gemini_analysis/Benchmark/mevis_train/error_${JSON_FILE}.txt" \
  --response_txt "/home/ubuntu/sunye/project/video_project/gemini_analysis/Benchmark/mevis_train/response_${JSON_FILE}.json"
