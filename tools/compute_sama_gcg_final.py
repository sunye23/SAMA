import os
import json

# 设置路径
folder_path = "/Sa2VA/reproduce/ml-ferret/gpt4_result/video_gcg/ferret_13b_gcg_metrics"

# 初始化指标累加器
metric_sums = {
    "miou": 0.0,
    "recall": 0.0,
    "METEOR": 0.0,
    "CIDEr": 0.0,
    "clair": 0.0,
    "Bleu_1": 0.0,
    "Bleu_2": 0.0,
    "Bleu_3": 0.0,
    "Bleu_4": 0.0,
    "ROUGE_L": 0.0,
    "SPICE": 0.0
}

# 获取所有 json 文件
# json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
json_files = [
    os.path.join(folder_path, f)
    for f in os.listdir(folder_path)
    if f.endswith('.json') and 'VidSTG' in f
]
num_files = len(json_files)
assert num_files > 0, "文件夹中没有找到 JSON 文件"

# 遍历文件累加指标
num_video = 0
for file_name in json_files:
    file_path = os.path.join(folder_path, file_name)
    num_video = num_video + 1
    with open(file_path, 'r') as f:
        data = json.load(f)
        metric_sums["miou"] += data.get("miou", 0.0)
        metric_sums["recall"] += data.get("recall", 0.0)
        metric_sums["clair"] += data.get("clair", 0.0)
        caption_scores = data.get("caption_scores", {})
        for key in caption_scores:
            metric_sums[key] += caption_scores[key]

# 计算平均值
average_metrics = {key: value / num_files for key, value in metric_sums.items()}

# 打印平均指标
print("[INFO] folder_path: {}".format(folder_path))
print(f"[INFO] VidSTG num video: {num_video}")
print(f"[INFO] 指标均值（基于 {num_files} 个文件）:")
for key, value in average_metrics.items():
    print(f"{key}: {value:.6f}")
