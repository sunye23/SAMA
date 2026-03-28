#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import json
import re
import cv2
import torch
import numpy as np
import skimage
from tqdm import tqdm
from PIL import Image

from tqdm import tqdm
from projects.llava_sam2.evaluation.utils.clair import clair
# ==============================
# 1) 定义与评估指标相关的函数
# ==============================

##########################################################################
#### Calculate mask-mIoU (计算mask的mIoU)

def compute_iou(mask1, mask2):
    '''
        mask1: (H,W) 或 (T,H,W)
        mask2: (H,W) 或 (T,H,W)
    '''
    intersection = np.logical_and(mask1, mask2)  # 计算两个掩码的交集
    union = np.logical_or(mask1, mask2)  # 计算两个掩码的并集
    if np.sum(union) == 0:
        return 0.0
    iou = np.sum(intersection) / np.sum(union)  # IOU = 交集像素数 / 并集像素数
    return iou


def compute_miou(pred_masks, gt_masks):
    '''
        pred_masks  : N1 x (H,W) 或 N1 x (T,H,W)
        gt_masks    : N2 x (H,W) 或 N2 x (T,H,W)
    '''
    # 创建一个二维矩阵，用于存储每个pred_mask与每个gt_mask的IOU
    pred_masks = list(pred_masks)  # 确保是可迭代list
    gt_masks = list(gt_masks)
    # 新增的鲁棒性判断：
    if len(pred_masks) == 0 or len(gt_masks) == 0:
        return 0.0
    iou_matrix = np.zeros((len(pred_masks), len(gt_masks)))
    for i, pm in enumerate(pred_masks):
        for j, gm in enumerate(gt_masks):
            iou_matrix[i, j] = compute_iou(pm, gm)

    # 进行“1对1”匹配，将最优的匹配IOU挑选出来后，从矩阵中删除对应行列
    paired_iou = []
    while iou_matrix.size > 0 and np.max(iou_matrix) > 0:
        max_iou_idx = np.unravel_index(np.argmax(iou_matrix, axis=None), iou_matrix.shape)
        paired_iou.append(iou_matrix[max_iou_idx])
        iou_matrix = np.delete(iou_matrix, max_iou_idx[0], axis=0)  # 删除该pred对应的行
        iou_matrix = np.delete(iou_matrix, max_iou_idx[1], axis=1)  # 删除该gt对应的列

    return np.mean(paired_iou) if paired_iou else 0.0

def evaluate_mask_miou(all_pred_masks, all_gt_masks):
    mious = []

    for pred_masks, gt_masks in tqdm(zip(all_pred_masks, all_gt_masks), total=len(all_pred_masks)):
        if len(pred_masks) == 0 or len(gt_masks) == 0:
            mious.append(0.0)
        else:
            mious.append(compute_miou(pred_masks.values(), gt_masks.values()))

    mean_miou = np.mean(mious) if mious else 0.0
    print(f"\033[92mMean IoU (mIoU) across all videos: {mean_miou}\033[0m")
    return mean_miou

##########################################################################
#### Calculate Recall (计算召回率)

def compute_iou_matrix(pred_masks, gt_masks):
    '''
        pred_masks  : N1 x (H,W) 或 N1 x (T,H,W)
        gt_masks    : N2 x (H,W) 或 N2 x (T,H,W)
    '''
    pred_masks = list(pred_masks)
    gt_masks = list(gt_masks)

    iou_matrix = np.zeros((len(pred_masks), len(gt_masks)))
    for i, pm in enumerate(pred_masks):
        for j, gm in enumerate(gt_masks):
            iou_matrix[i, j] = compute_iou(pm, gm)
    return iou_matrix


# ========== 加载BERT并计算文本相似度 ==========
# 若离线环境无法访问transformers模型，可在此处进行相应的本地模型路径配置
from transformers import AutoTokenizer, AutoModel

from sklearn.metrics.pairwise import cosine_similarity
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")

def get_bert_embedding(text):
    inputs = bert_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = bert_model(**inputs)
    sentence_embedding = torch.mean(outputs.last_hidden_state[0], dim=0).detach().numpy()
    return sentence_embedding

def text_similarity_bert(str1, str2):
    emb1 = get_bert_embedding(str1)
    emb2 = get_bert_embedding(str2)
    return cosine_similarity([emb1], [emb2])[0, 0]
#
# def find_best_matches(gt_masks, gt_labels, pred_masks, pred_labels, iou_threshold=0.5, text_sim_threshold=0.5):
#     gt_masks = list(gt_masks)
#     pred_masks = list(pred_masks)
#
#     # 1) 计算IOU矩阵
#     ious = compute_iou_matrix(pred_masks, gt_masks)
#     # 2) 计算文本相似度矩阵
#     text_sims = np.zeros((len(gt_labels), len(pred_labels)))
#     for i, gt_label in enumerate(gt_labels):
#         for j, dt_label in enumerate(pred_labels):
#             text_sims[i, j] = text_similarity_bert(gt_label, dt_label)
#
#     # 3) 在IOU和文本相似度都达标的前提下，找出“1对1”最佳匹配
#     best_matches = []
#     while ious.size > 0:
#         max_iou_idx = np.unravel_index(np.argmax(ious), ious.shape)
#         if ious[max_iou_idx] < iou_threshold or text_sims[max_iou_idx] < text_sim_threshold:
#             break
#         best_matches.append(max_iou_idx)
#         ious[max_iou_idx[0], :] = 0
#         ious[:, max_iou_idx[1]] = 0
#         text_sims[max_iou_idx[0], :] = 0
#         text_sims[:, max_iou_idx[1]] = 0
#
#     return best_matches

def find_best_matches(gt_masks, gt_labels, pred_masks, pred_labels, iou_threshold=0.5, text_sim_threshold=0.5):
    gt_masks = list(gt_masks)
    pred_masks = list(pred_masks)

    ious = compute_iou_matrix(pred_masks, gt_masks)

    text_sims = np.zeros((len(gt_labels), len(pred_labels)))
    for i, gt_label in enumerate(gt_labels):
        for j, dt_label in enumerate(pred_labels):
            text_sims[i, j] = text_similarity_bert(gt_label, dt_label)

    # 新增安全检查 (防止维度不一致)
    min_rows = min(ious.shape[0], text_sims.shape[0])
    min_cols = min(ious.shape[1], text_sims.shape[1])
    ious = ious[:min_rows, :min_cols]
    text_sims = text_sims[:min_rows, :min_cols]

    best_matches = []
    while ious.size > 0:
        max_iou_idx = np.unravel_index(np.argmax(ious), ious.shape)
        if ious[max_iou_idx] < iou_threshold or text_sims[max_iou_idx] < text_sim_threshold:
            break
        best_matches.append(max_iou_idx)
        ious[max_iou_idx[0], :] = 0
        ious[:, max_iou_idx[1]] = 0
        text_sims[max_iou_idx[0], :] = 0
        text_sims[:, max_iou_idx[1]] = 0

    return best_matches

def evaluate_recall_with_mapping(all_gt_masks, all_gt_phrases, all_pred_masks, all_pred_phrases,
                                 iou_threshold=0.5, text_sim_threshold=0.5):
    true_positives = 0
    actual_positives = 0

    for gt_masks, gt_labels, pred_masks, pred_labels in tqdm(zip(all_gt_masks, all_gt_phrases, all_pred_masks, all_pred_phrases),
                                                             total=len(all_gt_masks)):
        if len(pred_masks) == 0 or len(pred_labels) == 0:
            continue  # 无预测mask，跳过即可
        actual_positives += len(gt_labels)
        best_matches = find_best_matches(gt_masks.values(), gt_labels,
                                         pred_masks.values(), pred_labels,
                                         iou_threshold, text_sim_threshold)
        true_positives += len(best_matches)

    recall = true_positives / actual_positives if actual_positives > 0 else 0
    print(f"\033[92mRecall: {recall:.3f}\033[0m")
    return recall
##########################################################################
#### Caption Quality (评估字幕质量)

def eval_caption_quality(all_gt_references, all_pred_captions):
    # 使用COCO官方的字幕评估方式
    from tqdm import tqdm
    references = {}
    captions = {}

    k = 0
    for gt_ref, pred_caption in tqdm(zip(all_gt_references, all_pred_captions), total=len(all_gt_references)):
        # 如果文本过长，只截断前2000字符
        if len(gt_ref) > 2000:
            gt_ref = gt_ref[:2000]
        if len(pred_caption) > 2000:
            pred_caption = pred_caption[:2000]
        references[str(k)] = [gt_ref]
        captions[str(k)] = [pred_caption]
        k += 1

    # 保存为json
    import json
    new_cap = []
    for k, v in captions.items():
        new_cap.append({'image_id': k, 'caption': v[0]})
    new_ref = {'images': [], 'annotations': []}
    for kk, refs in references.items():
        new_ref['images'].append({'id': kk})
        for ref_ in refs:
            new_ref['annotations'].append({'image_id': kk, 'id': kk, 'caption': ref_})

    with open('tmp_references.json', 'w') as fgts:
        json.dump(new_ref, fgts)
    with open('tmp_captions.json', 'w') as fres:
        json.dump(new_cap, fres)

    # 使用coco官方评测代码
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap
    annotation_file = 'tmp_references.json'
    results_file = 'tmp_captions.json'

    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.evaluate()
    caption_dict = dict()
    for metric, score in coco_eval.eval.items():
        caption_dict[metric] = score
        print(f'\033[92m{metric}: {score:.3f}\033[0m')
    return caption_dict

def eval_caption_quality_with_clair(all_gt_references, all_pred_captions):
    # 如果要使用clair来评估，需要自定义的utils.clair

    references = {}
    captions = {}

    k = 0
    for gt_ref, pred_caption in tqdm(zip(all_gt_references, all_pred_captions), total=len(all_gt_references)):
        if len(gt_ref) > 2000:
            gt_ref = gt_ref[:2000]
        if len(pred_caption) > 2000:
            pred_caption = pred_caption[:2000]
        references[str(k)] = [gt_ref]
        captions[str(k)] = [pred_caption]
        k += 1

    sum_ = 0
    count_ = 0
    for k_ in references:
        clair_score, reason = clair(captions[k_], references[k_], model='chat-gpt')
        sum_ += clair_score
        count_ += 1

    avg_score = sum_ / count_
    print(f'\033[92m CLAIR Score: {avg_score:.3f}\033[0m')
    return avg_score


# ==============================
# 2) 这里模拟“ValGCGDataset”的引用
#    为了可运行，若你有自己的utils.dataset可自行保留。
#    暂时简单注释或保留一个空壳子。
# ==============================
try:
    from utils.dataset import ValGCGDataset
except ImportError:
    # 若无法import，给出一个空壳，保证脚本能运行
    class ValGCGDataset:
        def __init__(self, video_dataset_dir, val_datasets='video_gcg||mevis_gcg||vidstg_gcg'):
            # 模拟一下有长度2
            self._length = 2
        def __len__(self):
            return self._length
        def __getitem__(self, idx):
            # 这里只是示意
            return {}


# ==============================
# 3) 解析命令行参数
# ==============================
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GCG Task")
    parser.add_argument("--video_dataset_dir", default='./video_dataset', type=str)
    parser.add_argument("--vis_save_path", default="./vis_output/eval_gcg", type=str)
    parser.add_argument("--dataset_name", default="video_gcg", type=str, choices=["video_gcg"])
    parser.add_argument("--eval_miou", action="store_true", default=False)
    parser.add_argument("--eval_recall", action="store_true", default=False)
    parser.add_argument("--eval_caption", action="store_true", default=False)
    parser.add_argument("--use_clair", action="store_true", default=False)

    # 新增一个可选参数 --demo_data 用来直接生成伪数据调试
    parser.add_argument("--demo_data", action="store_true", default=False,
                        help="If set, skip reading real data/res.json and directly use fake data.")
    return parser.parse_args()


# ==============================
# 4) 主函数
# ==============================
def main():
    args = parse_args()

    # 如果不是demo模式，就尝试走正常数据集
    if not args.demo_data:
        # 实际加载数据集
        if args.dataset_name == "video_gcg":
            eval_dataset = ValGCGDataset(args.video_dataset_dir, val_datasets='video_gcg||mevis_gcg||vidstg_gcg')
        else:
            raise ValueError(f"Invalid dataset name: {args.dataset_name}")

        print('eval_dataset', len(eval_dataset))  # 打印验证集的长度

        # 下面尝试从硬盘读取res.json
        all_res = []
        for idx in tqdm(range(len(eval_dataset))):
            save_dir_for_current_video = os.path.join(args.vis_save_path, args.dataset_name, f"{idx:06d}")
            # print(save_dir_for_current_video)
            saved_file = os.path.join(save_dir_for_current_video, "res.json")

            try:
                if os.path.exists(saved_file):
                    with open(saved_file, 'r') as file:
                        res = json.load(file)
                        all_res.append(res)
                else:
                    all_res.append(None)
            except Exception as e:
                print(f"Error in processing {idx}: {e}")
                all_res.append(None)

        print('all_res', len(all_res))

        # ============ 在 all_res 基础上计算指标 ============
        evaluate_all_metrics(args, all_res)

    else:
        print("==== Running in DEMO mode: Generating fake data for debugging ====\n")
        import numpy as np

        # ------------------------
        # 1) 构造 2 个视频对应的文本 (res.json风格)
        #    让它们在字幕上有较多 n-gram 重叠，以获取非零的 BLEU / METEOR / CIDEr / ROUGE。
        # ------------------------
        all_res = []

        # ---- 视频1：GT 与 Pred 大部分相似，但不是完全相同 ----
        res_video_1 = {
            "gt_text": "GT文本：一只猫在沙发上休息，一只狗在门口。",
            "pred_text": "Pred文本：有一只猫在沙发上休息，还有一条狗在门口。",
            "gt_text_cleaned": "一只猫在沙发上休息，一只狗在门口",
            "pred_text_cleaned": "有一只猫在沙发上休息，还有一条狗在门口",
            # 短语上让两边相同，以保证Recall时的文本匹配通过
            "gt_phrases": ["cat", "dog"],
            "pred_phrases": ["cat", "dog"]
        }

        # ---- 视频2：GT 与 Pred 完全相同，便于观察指标分数 ----
        res_video_2 = {
            "gt_text": "GT文本：大象在河边玩耍，一只鸟从天空飞过。",
            "pred_text": "Pred文本：大象在河边玩耍，一只鸟从天空飞过。",
            "gt_text_cleaned": "大象在河边玩耍，一只鸟从天空飞过",
            "pred_text_cleaned": "大象在河边玩耍，一只鸟从天空飞过",
            "gt_phrases": ["elephant", "bird"],
            "pred_phrases": ["elephant", "bird"]
        }

        all_res.append(res_video_1)
        all_res.append(res_video_2)

        # ------------------------
        # 2) 构造 GT & Pred mask：让 iouThreshold=0.5 下部分通过、部分不通过，从而 Recall≠0。
        #    每个视频都有 2个目标(obj_id=0,1)，帧数 T=2, 尺寸 4x4。
        # ------------------------
        all_gt_masks = []
        all_pred_masks = []

        # ========== 视频1 ==========
        # 让video1的2个对象都高于0.5 iou，保证Recall=1
        # 对象0：GT和Pred完全一致 => iou=1
        v1_obj0_gt = np.array([
            [[1, 1, 0, 0],
             [1, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]],

            [[1, 1, 1, 0],
             [1, 1, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]]
        ], dtype=bool)
        v1_obj0_pred = v1_obj0_gt.copy()  # 完全相同 => iou=1

        # 对象1：做部分重叠 => iou>0.5
        v1_obj1_gt = np.array([
            [[1, 1, 1, 1],
             [1, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]],

            [[1, 1, 1, 0],
             [1, 1, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]]
        ], dtype=bool)
        v1_obj1_pred = np.array([
            [[1, 1, 1, 0],
             [1, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]],

            [[1, 1, 1, 0],
             [1, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]]
        ], dtype=bool)

        all_gt_masks.append({0: v1_obj0_gt, 1: v1_obj1_gt})
        all_pred_masks.append({0: v1_obj0_pred, 1: v1_obj1_pred})

        # ========== 视频2 ==========
        # 让video2中，obj0的IOU>0.5 => pass， obj1的IOU<0.5 => fail => Recall= 1/2=0.5
        # 对象0：整体 iou>0.5
        v2_obj0_gt = np.array([
            [[1, 1, 1, 0],
             [1, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]],

            [[1, 1, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]]
        ], dtype=bool)
        v2_obj0_pred = np.array([
            [[1, 1, 1, 0],
             [1, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]],

            [[1, 0, 0, 0],
             [1, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]]
        ], dtype=bool)
        # 经手动计算 => iou ~ 0.7，足以>0.5

        # 对象1：让iou<0.5 => fail
        v2_obj1_gt = np.array([
            [[1, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]],

            [[1, 1, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]]
        ], dtype=bool)
        v2_obj1_pred = np.array([
            [[0, 1, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]],

            [[0, 1, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]]
        ], dtype=bool)
        # overlap很少 => iou<0.5

        all_gt_masks.append({0: v2_obj0_gt, 1: v2_obj1_gt})
        all_pred_masks.append({0: v2_obj0_pred, 1: v2_obj1_pred})

        # ------------------------
        # 3) 直接调用主评估函数 (demo)
        # ------------------------
        evaluate_all_metrics_demo(args, all_res, all_gt_masks, all_pred_masks)

def evaluate_all_metrics(args, all_res):
    """
    当我们从硬盘读取了 all_res (其中存有各种res.json内容),
    并且想要加载对应的 mask 文件夹进行评估时，就会运行这里。
    """
    # 下列流程与原代码保持一致
    import os
    import cv2
    import skimage
    import re
    from tqdm import tqdm
    from numpy import np
    from PIL import Image

    if args.eval_miou:
        mious = []
    if args.eval_recall:
        iou_threshold = 0.5
        text_sim_threshold = 0.5
        true_positives = 0
        actual_positives = 0
    if args.eval_caption:
        all_gt_references = []
        all_pred_captions = []

    for idx in tqdm(range(len(all_res))):
        res = all_res[idx]
        if not res:
            continue

        try:
            gt_text = res['gt_text']
            pred_text = res['pred_text']
            gt_text_cleaned = res['gt_text_cleaned']
            pred_text_cleaned = res['pred_text_cleaned']
            gt_phrases = res['gt_phrases']
            pred_phrases = res['pred_phrases']
        except:
            # 如果字段缺失，就跳过
            continue

        # 构造当前视频的可视化输出路径（与用户原逻辑相同）
        save_dir_for_current_video = os.path.join(args.vis_save_path, args.dataset_name, f"{idx:06d}")
        img_frames_dir = os.path.join(save_dir_for_current_video, "img_frames")
        if not os.path.exists(img_frames_dir):
            continue

        filenames = os.listdir(img_frames_dir)
        sorted_filenames = sorted(filenames, key=lambda x: int(re.search(r'\d+', x).group()))
        # 如果需要评估mIoU或Recall，就加载mask
        if args.eval_miou or args.eval_recall:
            images = []
            for filename in sorted_filenames:
                if filename.endswith(".jpg"):
                    image_path = os.path.join(img_frames_dir, filename)
                    if not os.path.exists(image_path):
                        continue
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(image)

            # 加载GT mask
            gt_masks = {}
            for obj_id in range(len(gt_phrases)):
                obj_dir = os.path.join(save_dir_for_current_video, f"gt_masks_{obj_id}")
                if not os.path.exists(obj_dir):
                    continue
                gt_masks[obj_id] = []
                for ti in range(len(images)):
                    mask_path = os.path.join(obj_dir, f"mask_{ti}.png")
                    if os.path.exists(mask_path):
                        mask_img = skimage.io.imread(mask_path)
                        gt_masks[obj_id].append(mask_img)
                    else:
                        # 若缺失，用空白掩码顶上
                        gt_masks[obj_id].append(np.zeros_like(images[ti][:,:,0], dtype=bool))

            # 加载pred mask
            pred_masks = {}
            for obj_id in range(len(pred_phrases)):
                obj_dir = os.path.join(save_dir_for_current_video, f"pred_masks_{obj_id}")
                if not os.path.exists(obj_dir):
                    continue
                pred_masks[obj_id] = []
                for ti in range(len(images)):
                    mask_path = os.path.join(obj_dir, f"mask_{ti}.png")
                    if os.path.exists(mask_path):
                        mask_img = skimage.io.imread(mask_path)
                        pred_masks[obj_id].append(mask_img)
                    else:
                        pred_masks[obj_id].append(np.zeros_like(images[ti][:,:,0], dtype=bool))

        # 计算 mIoU
        if args.eval_miou:
            # 需要先把 list 转 bool
            for k in gt_masks:
                gt_masks[k] = np.array(gt_masks[k], dtype=bool)  # (T,H,W)
            for k in pred_masks:
                pred_masks[k] = np.array(pred_masks[k], dtype=bool)
            mious.append(compute_miou(pred_masks.values(), gt_masks.values()))

        # 计算 Recall
        if args.eval_recall:
            actual_positives += len(gt_phrases)
            best_matches = find_best_matches(gt_masks.values(), gt_phrases,
                                             pred_masks.values(), pred_phrases,
                                             iou_threshold=0.5, text_sim_threshold=0.5)
            true_positives += len(best_matches)

        # 计算字幕
        if args.eval_caption:
            all_gt_references.append(gt_text_cleaned)
            all_pred_captions.append(pred_text_cleaned)

    if args.eval_miou:
        mean_miou = np.mean(mious) if mious else 0.0
        print(f"\033[92mMean IoU (mIoU) across all videos: {mean_miou}\033[0m")

    if args.eval_recall:
        recall = true_positives / actual_positives if actual_positives > 0 else 0
        print(f"\033[92mRecall: {recall:.3f}\033[0m")

    if args.eval_caption and not args.use_clair:
        print('Evaluating caption quality...')
        eval_caption_quality(all_gt_references, all_pred_captions)

    if args.eval_caption and args.use_clair:
        print('Evaluating caption quality with CLAIR...')
        eval_caption_quality_with_clair(all_gt_references, all_pred_captions)


def evaluate_all_metrics_demo(args, all_res, all_gt_masks, all_pred_masks):
    """
    当我们在demo_data模式下直接生成了all_res和对应的GT/Pred mask，就会调用这里进行演示。
    """
    # 先把 mask、phrases 传给 evaluate_mask_miou & evaluate_recall_with_mapping
    # 以及 caption 部分
    # -- 先组装
    all_gt_phrases = []
    all_pred_phrases = []

    for r in all_res:
        all_gt_phrases.append(r["gt_phrases"])
        all_pred_phrases.append(r["pred_phrases"])

    import re

    def extract_groundings(sentence, start_token="<g_s>", end_token="<g_e>"):
        pattern = re.escape(start_token) + r"(.*?)" + re.escape(end_token)
        matches = re.findall(pattern, sentence, flags=re.DOTALL)
        return list(map(lambda s: s.strip().lower(), matches))

    def remove_special_tokens(sentence, tokens=None):
        if tokens is None:
            tokens = ["<g_s>", "<g_e>", "<seg>"]
        # 构造正则表达式，匹配所有待去除的标识符
        pattern = "(" + "|".join(map(re.escape, tokens)) + ")"
        # 替换标识符为空字符串
        cleaned = re.sub(pattern, "", sentence)
        # 合并连续的空白字符为一个空格，并去掉首尾空白
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    text = ("Given the man's <g_s> In the background <g_e> <seg>focused attention on the woman, "
            "and also <g_s> behind her <g_e> something is happening.")
    # groundings = extract_groundings(text)

    # result = remove_special_tokens(text)
    def test_fun(data):
        print("Grounding: ")
        print(extract_groundings(data))
        print("Cleaned data...")
        print(remove_special_tokens(data))
    # 计算mIoU
    if args.eval_miou:
        print("\n======= 评价 mIoU  =======")
        evaluate_mask_miou(all_pred_masks, all_gt_masks)

    # 计算Recall
    if args.eval_recall:
        print("\n======= 评价 Recall =======")
        evaluate_recall_with_mapping(all_gt_masks, all_gt_phrases, all_pred_masks, all_pred_phrases,
                                     iou_threshold=0.5, text_sim_threshold=0.5)

    # 计算字幕质量
    if args.eval_caption:
        print("\n======= 评价 Caption =======")
        all_gt_references = [x["gt_text_cleaned"] for x in all_res]
        all_pred_captions = [x["pred_text_cleaned"] for x in all_res]

        if args.eval_caption and not args.use_clair:
            print('Evaluating caption quality...')
            eval_caption_quality(all_gt_references, all_pred_captions)

        if args.eval_caption and args.use_clair:
            print('Evaluating caption quality with CLAIR...')
            eval_caption_quality_with_clair(all_gt_references, all_pred_captions)
        # if not args.use_clair:
        #     eval_caption_quality(all_gt_references, all_pred_captions)
        # else:
        #     eval_caption_quality_with_clair(all_gt_references, all_pred_captions)

    print("\n[Demo] 完成伪数据的指标演示，结束。")


if __name__ == "__main__":
    main()
