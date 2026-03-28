import os
import json
import argparse
import torch
import torch.distributed as dist
from tqdm import tqdm
from collections import defaultdict
import re

from projects.llava_sam2.evaluation.eval_gcg_metrics_demo import (
    eval_caption_quality,
    eval_caption_quality_with_clair
)

def remove_special_tokens(sentence, tokens=None):
    if tokens is None:
        tokens = ["<p>", "</p>", "[SEG]"]
    pattern = "(" + "|".join(map(re.escape, tokens)) + ")"
    cleaned = re.sub(pattern, "", sentence)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def truncate_caption(caption, max_words=200):
    return ' '.join(caption.strip().split()[:max_words])

def load_predictions_grouped_by_video(pred_dir, gt_dir):
    video_to_texts = defaultdict(lambda: {"gt": [], "pred": []})
    files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.json')])

    for fname in tqdm(files, desc='Loading prediction files'):
        pred_path = os.path.join(pred_dir, fname)
        gt_path = os.path.join(gt_dir, fname)

        if not os.path.exists(gt_path):
            print(f"[Warning] Missing GT for {fname}, skip.")
            continue

        with open(pred_path, 'r', encoding='utf-8') as f_pred, open(gt_path, 'r', encoding='utf-8') as f_gt:
            pred_json = json.load(f_pred)
            gt_json = json.load(f_gt)

        file_name = gt_json.get("file_name")
        if not file_name:
            continue

        pred = pred_json.get("prediction", "").strip()
        description_dict = gt_json.get("description", {})
        gt_values = [v.strip() for v in description_dict.values() if v.strip()] if isinstance(description_dict, dict) else []
        gt = gt_values[0] if gt_values else ""

        if not gt:
            continue
        if not pred:
            pred = "None"

        clean_pred = remove_special_tokens(pred)
        clean_gt = remove_special_tokens(gt)

        video_to_texts[file_name]["gt"].append(clean_gt)
        video_to_texts[file_name]["pred"].append(clean_pred)

    return video_to_texts

def main(args):
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    pred_dir = args.pred_dir
    gt_dir = args.gt_dir
    save_dir = args.save_dir
    final_save_file = args.final_save_file
    use_clair = args.use_clair

    os.makedirs(save_dir, exist_ok=True)
    video_to_texts = load_predictions_grouped_by_video(pred_dir, gt_dir)

    all_video_scores_gcg = []
    all_video_scores_clair = [] if use_clair else None

    video_names = sorted(video_to_texts.keys())
    shard = [vn for idx, vn in enumerate(video_names) if idx % world_size == rank]

    for video_name in tqdm(shard, desc=f"Evaluating [Rank {rank}]"):
        save_file = os.path.join(save_dir, f"{video_name.replace('/', '_')}.json")

        if os.path.exists(save_file):
            with open(save_file, 'r') as f:
                result = json.load(f)
            all_video_scores_gcg.append(result['GCG_standard'])
            if use_clair and 'Clair_standard' in result:
                all_video_scores_clair.append(result['Clair_standard'])
            continue

        try:
            gt_texts = video_to_texts[video_name]["gt"]
            pred_texts = video_to_texts[video_name]["pred"]

            video_score_gcg = eval_caption_quality(gt_texts, pred_texts)
            all_video_scores_gcg.append(video_score_gcg)

            save_result = {"GCG_standard": video_score_gcg}

            if use_clair:
                video_score_clair = eval_caption_quality_with_clair(gt_texts, pred_texts)
                all_video_scores_clair.append(video_score_clair)
                save_result["Clair_standard"] = video_score_clair

            with open(save_file, 'w') as f:
                json.dump(save_result, f, indent=2)
        except Exception as e:
            print(f"[Warning] Failed to evaluate {video_name}, skipping. Error: {e}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str, required=True)
    parser.add_argument('--gt_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--final_save_file', type=str, required=True)
    parser.add_argument('--use_clair', action='store_true')
    args = parser.parse_args()
    main(args)
