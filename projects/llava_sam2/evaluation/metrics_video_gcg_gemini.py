import os
import json
import argparse
from tqdm import tqdm
from pathlib import Path
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

def truncate(text: str, max_w: int = 200) -> str:
    ws = text.split()
    return " ".join(ws[:max_w]) if len(ws) > max_w else text

def load_predictions_grouped_by_video(pred_dir: Path, gt_dir: Path):
    video_to_texts = defaultdict(lambda: {"gt": [], "pred": []})
    pred_files = sorted(pred_dir.glob("*.json"))
    assert pred_files, f"No JSON files found in {pred_dir}"

    filter_idx_file = Path("/Sa2VA/reproduce/Sa2VA/video_region_meta/question_ids_list.json")
    with open(filter_idx_file, 'r') as f:
        valid_ids = set(str(idx) for idx in json.load(f))

    for pf in tqdm(pred_files, desc="Loading predictions"):
        gid = pf.stem
        if gid not in valid_ids:
            continue

        gt_path = gt_dir / f"{gid}.json"
        if not gt_path.exists():
            print(f"[Warning] GT not found for id {gid}, skip.")
            continue

        with open(pf, "r", encoding="utf-8") as f:
            pred_json = json.load(f)
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_json = json.load(f)

        pred_cap = remove_special_tokens(pred_json.get("prediction", ""))
        gt_cap = remove_special_tokens(gt_json.get("answer", ""))
        file_name = gt_json.get("file_name", None)
        if "VidSTG" not in file_name:           # TODO：注意这里做了修改
            continue
        if pred_cap and gt_cap and file_name:
            video_to_texts[file_name]["gt"].append(gt_cap)
            video_to_texts[file_name]["pred"].append(pred_cap)

    return video_to_texts

def main(args):
    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    save_dir = Path(args.save_dir)
    final_save_file = Path(args.final_save_file)
    use_clair = args.use_clair
    import torch
    import torch.distributed as dist

    # 初始化分布式
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    os.makedirs(save_dir, exist_ok=True)
    video_to_texts = load_predictions_grouped_by_video(pred_dir, gt_dir)
    all_video_scores_gcg = []
    all_video_scores_clair = [] if use_clair else None

    video_names = sorted(video_to_texts.keys())
    shard = [v for idx, v in enumerate(video_names) if idx % world_size == rank]
    for video_name in tqdm(shard, desc=f"Evaluating @ Rank {rank}"):
        texts = video_to_texts[video_name]
        safe_video_name = video_name.replace("/", "_")
        save_file = save_dir / f"{safe_video_name}.json"
        if save_file.exists():
            with open(save_file, 'r') as f:
                result = json.load(f)
            all_video_scores_gcg.append(result['GCG_standard'])
            if use_clair and 'Clair_standard' in result:
                all_video_scores_clair.append(result['Clair_standard'])
            continue

        try:
            gt_texts = texts["gt"]
            pred_texts = texts["pred"]
            video_score_gcg = eval_caption_quality(gt_texts, pred_texts)
            all_video_scores_gcg.append(video_score_gcg)
            save_result = {"GCG_standard": video_score_gcg}

            if use_clair:
                video_score_clair = eval_caption_quality_with_clair(gt_texts, pred_texts)
                all_video_scores_clair.append(video_score_clair)
                save_result["Clair_standard"] = video_score_clair
            print(f"[INFO] {video_name} result: {save_result}")
            with open(save_file, 'w') as f:
                json.dump(save_result, f, indent=2)

        except Exception as e:
            print(f"[Warning] Failed to evaluate {video_name}, skipping. Error: {e}")
            continue

    dist.barrier()  # 等待所有进程完成
    caption_keys = all_video_scores_gcg[0].keys()
    mean_scores_gcg = {k: sum(d[k] for d in all_video_scores_gcg) / len(all_video_scores_gcg) for k in caption_keys}

    final_result = {
        "GCG_standard": mean_scores_gcg
    }

    if use_clair and all_video_scores_clair:
        mean_scores_clair = sum(all_video_scores_clair) / len(all_video_scores_clair)
        final_result["Clair_standard"] = mean_scores_clair

    os.makedirs(final_save_file.parent, exist_ok=True)
    with open(final_save_file, 'w') as f:
        json.dump(final_result, f, indent=2)

    print(f"\n✅ Final caption metrics saved to: {final_save_file}")

    print("\n--- GCG Standard ---")
    for k, v in mean_scores_gcg.items():
        print(f"{k}: {v:.4f}")
    if use_clair and all_video_scores_clair:
        print("\n--- Clair Standard ---")
        print(f"{mean_scores_clair:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str, required=True)
    parser.add_argument('--gt_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--final_save_file', type=str, required=True)
    parser.add_argument('--use_clair', action='store_true')
    parser.set_defaults(use_clair=False)
    args = parser.parse_args()
    main(args)
