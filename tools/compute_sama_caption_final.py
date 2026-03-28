import os
import json

folder_path = "/Sa2VA/reproduce/Sa2VA/work_dirs/AblationExperiments/rebuttal/s2a_1B_loss100/s2a_1B_loss100_iter_58772/video_region_cap_pred/video_caption_metrics"

# 初始化累计指标字典 as
metric_sums = {
    "Bleu_1": 0.0,
    "Bleu_2": 0.0,
    "Bleu_3": 0.0,
    "Bleu_4": 0.0,
    "METEOR": 0.0,
    "ROUGE_L": 0.0,
    "CIDEr": 0.0,
    "SPICE": 0.0,
    "clair": 0.0
}

# 获取所有 JSON 文件
json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
print("[INFO] length json_files: {}".format(len(json_files)))
# assert len(json_files) == 522, f"期望522个文件，实际找到{len(json_files)}个文件"

# 遍历每个文件，累加指标
for file_name in json_files:
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'r') as f:
        data = json.load(f)
        caption_scores = data.get("caption_scores", {})
        for key in caption_scores:
            metric_sums[key] += caption_scores[key]
        metric_sums["clair"] += data.get("clair", 0.0)

# 计算均值
average_metrics = {key: value / len(json_files) for key, value in metric_sums.items()}

# 打印结果
print("[INFO] folder_path: {}".format(folder_path))
print(f"[INFO] 指标均值（基于{len(json_files)}个文件）：")
for key, value in average_metrics.items():
    print(f"{key}: {value:.6f}")
