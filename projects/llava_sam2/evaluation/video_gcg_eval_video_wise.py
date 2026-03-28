import argparse  # 用于解析命令行参数
import copy
import json
import math
import os
import pickle
import re
import time
import random
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import TABLEAU_COLORS
import numpy as np
import pycocotools.mask as maskUtils
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from pycocotools import mask as mask_utils  # COCO格式的掩码处理库，支持RLE等编码
from transformers import (  # HuggingFace提供的transformers库，
    AutoModel,  # 可自动加载预训练模型
    #   用于加载causal language model
    AutoTokenizer,  # 自动加载相应模型的tokenizer
    #   量化相关的配置（此处未实际使用）
    #   CLIP图像预处理器（此处未实际使用）
    #   CLIP视觉模型（此处未实际使用）
    #   文本生成的配置
)
import argparse
import copy
import os.path as osp
import torch
from mmengine.dist import (collect_results, get_dist_info, get_rank, init_dist,
                           master_only)
from xtuner.registry import BUILDER
from xtuner.configs import cfgs_name_path
from xtuner.model.utils import guess_load_checkpoint
from mmengine.config import Config
from mmengine.fileio import PetrelBackend, get_file_backend
from mmengine.config import ConfigDict
import os
from torch.utils.data import Dataset
from utils import _init_dist_pytorch, get_dist_info, collect_results_cpu  # 自定义工具函数，分布式初始化、获取rank、汇总结果等
from projects.llava_sam2.evaluation.utils.clair import clair
from projects.llava_sam2.evaluation.eval_gcg_metrics_demo import (evaluate_mask_miou, evaluate_recall_with_mapping,
                                                                  eval_caption_quality, eval_caption_quality_with_clair)

from projects.llava_sam2.evaluation.utils import barrier
def get_sparse_indices_uniform(vid_len, interval=3, max_frames=32, min_frames=8):
    if vid_len <= 0:
        return []

    # 按固定间隔采样
    sample_indices = list(range(0, vid_len, interval))

    # 若超过最大帧数，则均匀间隔取max_frames个
    if len(sample_indices) > max_frames:
        step = len(sample_indices) / max_frames
        sample_indices = [sample_indices[int(i * step)] for i in range(max_frames)]

    # 若不足最小帧数，则均匀补齐
    if len(sample_indices) < min_frames:
        additional_needed = min_frames - len(sample_indices)
        extra_step = vid_len / (additional_needed + 1)
        extra_frames = [int((i + 1) * extra_step) for i in range(additional_needed)]
        sample_indices.extend(extra_frames)
        sample_indices = sorted(set(sample_indices))

    return sorted(sample_indices)


def parse_args():
    parser = argparse.ArgumentParser(description='GCG')               # 创建命令行参数解析器，描述“GCG”任务
    parser.add_argument('--model_path', help='hf model path.')        # 模型路径参数
    parser.add_argument('--json_file', type=str, required=True,
                        help='path to json file to process')  # ✨新增
    parser.add_argument('--max_frames', type=int, default=128)
    parser.add_argument(
        '--split',
        default='val',
        help='Specify a split')                                       # 数据集的split名称，默认“val”
    parser.add_argument(
        '--save_dir',
        default='./gcg_pred/',
        help='save path')                                             # 推理结果的保存目录，默认'./gcg_pred/'
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')                                          # 分布式启动器类型，可选'none','pytorch','slurm','mpi'
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)  # 本地进程的rank（分布式相关）
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)               # 若没有LOCAL_RANK，则写入环境变量
    return args

# 指定一个图像文件夹路径，放置需要推理的图像
IMAGE_FOLDER = '/SAMA/reproduce/SAMA/data/glamm_data/images/grandf/val_test'

def truncate_caption(caption, max_words=80):
    """
    将过长的caption截断到指定的单词数上限。
    你也可以选择按字符数截断，如 len(caption) > N 时切掉后半部分。
    """
    words = caption.split()
    if len(words) > max_words:
        words = words[:max_words]
        return ' '.join(words)
    else:
        return caption

# todo: 三个数据集 1. Region-level Video description 2. Video grounded chat 3. hallucination mutli-choice QA

class GCGVideoInferenceDataset(Dataset):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    FAST_IMG_CONTEXT_TOKEN = '<FAST_IMG_CONTEXT>'
    FAST_IMG_START_TOKEN = '<fast_img>'
    FAST_IMG_END_TOKEN = '</fast_img>'

    LIMIT = ''

    VP_START_TOKEN = '<vp>'
    VP_END_TOKEN = '</vp>'
    def __init__(self, video_folder, data_json, qa_per_item=5, max_frames=32, args=None):
        """
        参数:
          video_folder: 存放视频帧的根目录，比如 /remote-home/sunye/video_project/video_dataset/mevis/train/JPEGImages/
          data_json: 包含所有视频及对话信息的JSON文件路径 (你的示例文件)
          qa_per_item: 每个记录最多包含多少轮问答（默认为 5）
        """
        self.video_folder = video_folder
        self.qa_per_item = qa_per_item
        self.image_size = 448
        self._system = ''
        self.downsample_ratio = 0.5
        self.dense_len = 5
        patch_size = 14
        self.args = args
        self.patch_size = patch_size
        self.patch_token = int((self.image_size // patch_size) ** 2 * (self.downsample_ratio ** 2))
        self.dataset_dicts = self.json_file_preprocess(data_json, args)
        random.Random(0).shuffle(self.dataset_dicts)
        mevis_json = '/SAMA/reproduce/SAMA/video_region_meta/mask_dict/Val/Mevis_mask_dict_val.json'
        lvvis_json = '/SAMA/reproduce/SAMA/video_region_meta/mask_dict/Val/LVVIS_mask_dict_val.json'
        ref_youtube_json = '/SAMA/reproduce/SAMA/video_region_meta/mask_dict/Val/RefYoutube_mask_dict_val.json'
        vidstg_json = '/SAMA/reproduce/SAMA/video_region_meta/mask_dict/Val/VidSTG_mask_dict_val_updated.json'
        self.max_frames = max_frames
        self.mask_dicts = dict()
        with open(mevis_json, 'r') as fp:
            self.mask_dicts['mevis'] = json.load(fp)
        with open(lvvis_json, 'r') as fp:
            self.mask_dicts['lvvis'] = json.load(fp)
        with open(ref_youtube_json, 'r') as fp:
            self.mask_dicts['ref_youtube'] = json.load(fp)
        with open(vidstg_json, 'r') as fp:
            self.mask_dicts['vidstg'] = json.load(fp)
        print("ALL mask dict json file has been loaded....")

    def __len__(self):
        """数据集大小，即可迭代的样本数。"""
        return len(self.dataset_dicts)

    def json_file_preprocess(self, data_path, args=None):
        with open(data_path, 'r', encoding='utf-8') as fp:
            data = json.load(fp)

        dataset_dicts = []
        data_id = 0  # 用于给每条数据分配唯一的 id
        filter_num = 0
        sum_total = 0
        round_per_data = self.qa_per_item
        index = 0
        for video in tqdm(data, desc='Registering Conv BenchMark...'):
            video_info = {
                'height': video.get('height', None),
                'width': video.get('width', None),
                'frames': video.get('frames', None),
                'anno_map': video.get('anno_map', None),
                'video': video.get('video', None),
                'obj_masks': video.get('obj_masks', None),
                'groundtruth': video.get('groundtruth', None),
                'conversations': video.get('conversation', None),
            }
            # 如果 conversations 不存在，跳过该视频
            if video_info['conversations'] is None:
                continue
            conversations = video_info['conversations']
            # 确保 conversations 的长度为偶数，如果不是，抛弃最后一条消息
            if len(conversations) % 2 != 0:
                conversations = conversations[:-1]

            # 计算对话轮数（每两条消息为一轮）
            total_rounds = len(conversations) // 2
            sum_total = sum_total + total_rounds
            # 计算需要分割成多少个子列表，每个子列表最多包含 5 轮对话
            num_splits = (total_rounds + round_per_data - 1) // round_per_data  # 向上取整

            for i in range(num_splits):
                # 计算子列表的起始和结束索引（以消息为单位，每轮两条消息）
                start_idx = i * round_per_data * 2  # 每轮 2 条消息，5 轮共 10 条消息
                end_idx = min(start_idx + round_per_data * 2, len(conversations))
                sub_conversations = conversations[start_idx:end_idx]
                sub_obj_masks = video_info['obj_masks'][start_idx // 2:end_idx // 2]
                sub_groundtruth = video_info['groundtruth'][start_idx // 2:end_idx // 2]

                # 获取子列表的对话内容
                cleaned_conversations = []
                cleaned_obj_masks = []
                cleaned_groundtruth = []
                current_round = min(round_per_data, len(sub_conversations) // 2)
                question_id = None
                for idx_conv in range(current_round):
                    current = idx_conv * 2
                    question = sub_conversations[current]
                    if question['from'] == 'human':
                        question_value = question['value']
                        question_id = question['question_id']
                    else:
                        filter_num = filter_num + 1
                        continue
                    answer = sub_conversations[current + 1]
                    if answer['from'] == 'gpt':
                        answer_value = answer['value']
                    else:
                        filter_num = filter_num + 1
                        continue
                    current_obj_mask = sub_obj_masks[idx_conv]
                    current_gt = sub_groundtruth[idx_conv]
                    num_mask = question_value.count("<mask>")
                    if num_mask != len(current_obj_mask):
                        filter_num = filter_num + 1
                        continue
                    num_seg = answer_value.count("[SEG]")
                    valid_seg = 0
                    for gt_item in current_gt:
                        if len(gt_item) != 0:
                            valid_seg = valid_seg + 1
                    if num_seg != valid_seg:
                        filter_num = filter_num + 1
                        continue
                    if num_seg > 0 and 'with interleaved segmentation masks.' not in question['value']:
                        question['value'] = question['value'] + ' Response with interleaved segmentation masks.'

                    cleaned_conversations.append(question)
                    cleaned_conversations.append(answer)
                    cleaned_obj_masks.append(current_obj_mask)
                    cleaned_groundtruth.append(current_gt)
                # 支持多个数据集筛选，如 mevis,lvvis 或 all
                if 'VidSTG' in video_info['video']:
                    video_info['frames'] = [
                        os.path.splitext(item["raw_filename"])[0] if isinstance(item, dict) and "raw_filename" in item
                        else os.path.splitext(item)[0]
                        for item in video_info['frames']
                    ]

                if len(cleaned_conversations) != 0:
                    record = {
                        'id': data_id,
                        'question_id': question_id,
                        'height': video_info['height'],
                        'width': video_info['width'],
                        'frames': video_info['frames'],
                        'annotation': video_info['anno_map'],
                        'obj_masks': cleaned_obj_masks,
                        'groundtruth': cleaned_groundtruth,
                        'file_name': video_info['video'],
                        'conversations': cleaned_conversations,
                    }
                    dataset_dicts.append(record)
                    data_id += 1
        print("[INFO] Total conversation num: ", sum_total, "filtered conversation num: ", filter_num) #
        print(f"[INFO] Total conv evaluation length: {len(dataset_dicts)}")
        print(f"[INFO] Total conv evaluation length: {len(dataset_dicts)}")
        return dataset_dicts

    def decode_mask(self, video_masks, image_size):
        ret_masks = []
        for object_masks in video_masks:
            # None object
            if len(object_masks) == 0:
                if len(ret_masks) != 0:
                    _object_masks = ret_masks[0] * 0
                else:
                    _object_masks = np.zeros(
                        (self.dense_len, image_size[0], image_size[1]), dtype=np.uint8)
            else:
                _object_masks = []
                for i_frame in range(len(object_masks[0])):
                    _mask = np.zeros(image_size, dtype=np.uint8)
                    for i_anno in range(len(object_masks)):
                        if object_masks[i_anno][i_frame] is None:
                            continue
                        m = maskUtils.decode(object_masks[i_anno][i_frame])
                        if m.ndim == 3:
                            m = m.sum(axis=2).astype(np.uint8)
                        else:
                            m = m.astype(np.uint8)
                        if m.shape != image_size:
                            # 注意：cv2.resize需要 (width, height) 顺序
                            m = cv2.resize(m, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
                        _mask = _mask | m
                    _object_masks.append(_mask)
                _object_masks = np.stack(_object_masks, axis=0)
            # if self.pad_image_to_square:
            #     _object_masks = expand2square_mask(_object_masks)
            ret_masks.append(_object_masks)
        _shape = ret_masks[0].shape
        for item in ret_masks:
            if item.shape != _shape:
                print([_ret_mask.shape for _ret_mask in ret_masks])
                return None
        ret_masks = np.stack(ret_masks, axis=0)  # (n_obj, n_frames, h, w)

        ret_masks = torch.from_numpy(ret_masks)
        # ret_masks = F.interpolate(ret_masks, size=(self.image_size // self.down_ratio,
        #                           self.image_size // self.down_ratio), mode='nearest')
        ret_masks = ret_masks.flatten(0, 1)
        return ret_masks
    #
    # def _get_region_infos(self, masks):
    #     # masks tensor, (n_obj, h, w)
    #     masks_112_112 = F.interpolate(masks.unsqueeze(0), size=(64, 64), mode='nearest').squeeze(0)
    #     masks = F.interpolate(
    #         masks.unsqueeze(0),
    #         size=(int(self.image_size // self.patch_size * self.downsample_ratio),
    #               int(self.image_size // self.patch_size * self.downsample_ratio)),
    #         mode='nearest').squeeze(0)
    #     return masks, masks_112_112


    def _get_region_infos(self, masks):
        """
        将原本的掩码转换为对应的 bounding-box 区域全为 1，再对该结果进行插值。
        :param masks: (n_obj, H, W) 的张量，每个通道是一张二值掩码
        :return:
            masks_small:    (n_obj, target_h, target_w)
            masks_64_64:    (n_obj, 64, 64)
        """
        n_obj, h, w = masks.shape
        box_masks = []

        # 1) 将每个mask的非零区域替换为bbox区域
        for i_obj in range(n_obj):
            cur_mask = masks[i_obj]  # shape: (H, W)
            nonzero_rows, nonzero_cols = torch.where(cur_mask > 0)
            if len(nonzero_rows) == 0:
                # 空掩码，无非零像素
                box_masks.append(cur_mask)  # 或者 torch.zeros_like(cur_mask)
            else:
                # 找到 bbox (min_row, min_col) ~ (max_row, max_col)
                min_row, max_row = nonzero_rows.min(), nonzero_rows.max()
                min_col, max_col = nonzero_cols.min(), nonzero_cols.max()
                # 构造一个全0张量，然后把bbox区域置为1
                new_mask = torch.zeros_like(cur_mask)
                new_mask[min_row:max_row + 1, min_col:max_col + 1] = 1
                box_masks.append(new_mask)

        # 在dim=0上拼回 (n_obj, H, W)
        box_masks = torch.stack(box_masks, dim=0)

        # 2) 第一次插值: (n_obj, 64, 64)
        masks_64_64 = F.interpolate(
            box_masks.unsqueeze(0),
            size=(112, 112),
            mode='nearest'
        ).squeeze(0)  # -> (n_obj, 64, 64)

        # 3) 第二次插值: (n_obj, target_h, target_w)
        target_h = int(self.image_size // self.patch_size * self.downsample_ratio)
        target_w = int(self.image_size // self.patch_size * self.downsample_ratio)
        masks_small = F.interpolate(
            box_masks.unsqueeze(0),
            size=(target_h, target_w),
            mode='nearest'
        ).squeeze(0)  # -> (n_obj, target_h, target_w)

        return masks_small, masks_64_64


    def prepare_text(self, n_frames, conversations, num_image_tokens=256, n_fast_images=50, sparse_len=32):
        text_prompts = []
        num_context_token = sparse_len
        # if self.window_size > 0:
        #     num_context_token = math.ceil(num_context_token * 32 / self.window_stride) * 32
        # else:
        #     num_context_token = 32
        # context_str = ''
        context_str = self.IMG_CONTEXT_TOKEN * num_context_token

        frame_token_str = f'{self.IMG_START_TOKEN}' \
                          f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                          f'{self.IMG_END_TOKEN}'
        qa_list = []
        input = ''
        mask_index = 0
        question = None
        # conv_manager = ConversationManager(max_length=self.max_conv_len)
        for i, item in enumerate(conversations):
            if item['from'] == 'human':
                item['value'] = item['value'].replace('<image>\n', '').replace('<video>\n', '').replace('\n<video>', '').replace('\n<image>', '')
                question = item['value'].strip()
                if i == 0:
                    frame_tokens = frame_token_str + '\n'
                    frame_tokens = frame_tokens * n_frames
                    frame_tokens = frame_tokens.strip()
                    input = input + frame_tokens + context_str + item['value']
                    text_prompts.append(question)
                else:
                    input = input + context_str + item['value']
                    text_prompts.append(question)
                mask_count = input.count("<mask>")
                fill_token = []
                for i_idx in range(mask_index, mask_index + mask_count):
                    fill_token.append(self.VP_START_TOKEN + self.IMG_CONTEXT_TOKEN * 2 + self.VP_END_TOKEN)
                input = self.fill_masks(input, fill_token)
                mask_index = mask_index + mask_count
            elif item['from'] == 'gpt':
                answer = item['value'].strip()
                answer = answer.replace('<seg>', '[SEG]').replace('<g_s>', '<p>').replace('<g_e>', '</p>')
                qa_list.append({'input': input, 'output': answer})
                # conv_manager.add_conversation(question=question,answer=answer)
                input = ''
        qa_list[0].update({'system': self._system})
        return qa_list, text_prompts

    def fill_masks(self, question, replacement_tokens):
        token_iterator = iter(replacement_tokens)

        def replace_func(match):
            return next(token_iterator, match.group(0))

        filled_question = re.sub(r'<mask>', replace_func, question)
        return filled_question

    def __getitem__(self, index):
        record = self.dataset_dicts[index]
        data_dict = dict()
        mask_info = dict()
        data_dict['record_id'] = record['question_id']           # 拆分后分配的全局id
        data_dict['video_id'] = record['question_id']
        data_dict['frames'] = record['frames']          # 帧名列表
        masks_mapping = record['annotation']  # anno_map
        data_dict['obj_masks'] = record['obj_masks']    # 当前段落对应的obj_masks
        data_dict['groundtruth'] = record['groundtruth']
        data_dict['file_name'] = record['file_name']    # "mevis/train/JPEGImages/0a2c0bfe6909"
        frames_files = data_dict['frames']
        # TODO: 添加数据集的时候注意调整这里， 或者路径对齐
        if 'lvvis' in data_dict['file_name']:
            dir_path, folder = os.path.split(data_dict['file_name'])
            if folder.isdigit():
                folder = folder.zfill(5)
            new_file_name = os.path.join(dir_path, folder)
            frames_files = [
                os.path.join(self.video_folder, new_file_name, frame_file + ".jpg") for frame_file in frames_files
            ]
        else:
            frames_files = [
                os.path.join(self.video_folder, data_dict['file_name'], frame_file + ".jpg") for frame_file in frames_files
            ]
        images = []
        ori_width, ori_height = None, None
        for frame_idx, frame_path in enumerate(frames_files):
            frame_image = Image.open(frame_path).convert('RGB')
            if ori_height is None:
                ori_width, ori_height = frame_image.size
            else:
                assert ori_width == frame_image.size[0]
                assert ori_height == frame_image.size[1]
            images.append(frame_image)
        data_dict['images'] = images

        file_name = data_dict['file_name']
        if 'mevis' in file_name:
            mask_dict = self.mask_dicts.get('mevis', {})
        elif 'lvvis' in file_name:
            mask_dict = self.mask_dicts.get('lvvis', {})
        elif 'ref_youtube_vos' in file_name:
            mask_dict = self.mask_dicts.get('ref_youtube', {})
        elif 'VidSTG' in file_name:
            mask_dict = self.mask_dicts.get('vidstg', {})

        frame_indices = range(len(frames_files))
        for item in masks_mapping:
            if item == -1:
                continue
            obj_list = []
            obj_list_all = []
            # frames_masks_ = []
            frames_masks_all = []
            mask_annotation = mask_dict[str(item)]
            if 'lvvis' in file_name:
                mask_annotation = mask_annotation.get("segmentations", [])
            else:
                mask_annotation = mask_annotation
            for frame_idx in frame_indices:
                if frame_idx < len(mask_annotation) and mask_annotation[frame_idx] is not None:
                    frames_masks_all.append(copy.deepcopy(mask_annotation[frame_idx]))
                else:
                    empty_array = np.zeros((ori_height, ori_width), dtype=np.uint8)
                    empty_mask = maskUtils.encode(np.asfortranarray(empty_array))
                    frames_masks_all.append(empty_mask)
            obj_list_all.append(frames_masks_all)
            obj_masks_all = self.decode_mask([obj_list_all], image_size=(ori_height, ori_width))

            if item not in mask_info:
                temp_item = str(item)
                mask_info[temp_item] = dict()
                processed_mask, processed_mask_112 = self._get_region_infos(obj_masks_all)
                mask_info[temp_item]['mask'] = obj_masks_all
                mask_info[temp_item]['processed_mask'] = processed_mask
                mask_info[temp_item]['prompt_masks_112'] = processed_mask_112
                mask_info[temp_item]['height'] = ori_height
                mask_info[temp_item]['width'] = ori_width
            else:
                # 记录警告，跳过该重复 mask 的处理
                print(f"Duplicate mask id {item} found in sample {record['id']}, skipping duplicate.")
                continue

        prompt_masks = []
        prompt_masks_112 = []
        mask_count = []
        obj_list = record['obj_masks']
        gt_list = record['groundtruth']
        for item in obj_list:
            mask_count.append(len(item))
            for sub_item in item:
                temp_sub_item = str(sub_item)
                prompt_masks.append(mask_info[temp_sub_item]['processed_mask'])
                prompt_masks_112.append(mask_info[temp_sub_item]['prompt_masks_112'])
        gt_final = []
        for item in gt_list:
            mask_final = None
            for sub_item in item:
                temp_mask_dict = []
                if len(sub_item) != 0 and len(sub_item) == 1:
                    mask_final = mask_info[str(sub_item[0])]['mask']
                elif len(sub_item) > 1:
                    for index, sub_sub_item in enumerate(sub_item):
                        if index == 0:
                            mask_final = mask_info[str(sub_sub_item)]['mask']
                        else:
                            mask_final = mask_final | mask_info[str(sub_sub_item)]['mask']
                gt_final.append(mask_final)
        sparse_len = len(get_sparse_indices_uniform(len(frame_indices), max_frames=self.max_frames))

        qa_list, text_prompts = self.prepare_text(n_frames=self.dense_len, conversations=record['conversations'], num_image_tokens=self.patch_token, sparse_len=sparse_len)

        data_dict['gt_masks'] = gt_final
        data_dict['mask_count'] = mask_count
        data_dict['prompt_masks'] = prompt_masks
        data_dict['prompt_masks_112'] = prompt_masks_112
        data_dict['text_prompts'] = text_prompts
        data_dict['qa_list'] = qa_list
        data_dict['conversations'] = record['conversations']
        return data_dict

def extract_groundings(sentence, start_token="<p>", end_token="</p>"):
    pattern = re.escape(start_token) + r"(.*?)" + re.escape(end_token)
    matches = re.findall(pattern, sentence, flags=re.DOTALL)
    return list(map(lambda s: s.strip().lower(), matches))

def remove_special_tokens(sentence, tokens=None):
    if tokens is None:
        tokens = ["<p>", "</p>", "[SEG]", "<|im_end|>", '<s>', "<|end|>"]
    # 构造正则表达式，匹配所有待去除的标识符
    pattern = "(" + "|".join(map(re.escape, tokens)) + ")"
    # 替换标识符为空字符串
    cleaned = re.sub(pattern, "", sentence)
    # 合并连续的空白字符为一个空格，并去掉首尾空白
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned
from datetime import timedelta


def main():
    args = parse_args()
    video_folder = '/sama_bench'
    dataset = GCGVideoInferenceDataset(video_folder, args.json_file, qa_per_item=1, max_frames=args.max_frames, args=args)

    import torch.distributed as dist
    if args.launcher != 'none' and not dist.is_initialized():  # ← 加这一行判断
        _init_dist_pytorch(backend='nccl',
                           timeout=timedelta(hours=1))
        rank, world_size = get_dist_info()     # 获取当前进程的rank、总进程数
        torch.cuda.set_device(rank)            # 将当前进程绑定到对应的GPU
    else:
        rank = 0
        world_size = 1
    sampler = torch.utils.data.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=1,
        num_workers=8,
        pin_memory=False,
        collate_fn=lambda x:x[0],
    )

    hf_SAMA_model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval().cuda()

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    # 若结果存放目录不存在，则创建
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    results = []                                # 用于收集所有推理的文本结果
    n_samples = len(dataset)                    # 数据集中需要推理的图像数量
    per_rank_samples = math.ceil(n_samples / world_size) + 1  # 每个进程要处理的图像数量（多加1是保险）
    per_rank_ids = range(per_rank_samples * rank,
                         min(n_samples, per_rank_samples * (rank + 1)))  # 当前进程实际需要处理的索引范围

    gt_text = []
    pred_text = []
    gt_text_cleaned = []
    pred_text_cleaned = []
    gt_phases = []
    pred_phases = []
    gt_masks = []
    pred_masks = []
    video_ids = []
    print("length dataset: {}".format(len(dataset)))
    print("max_frames: {}".format(args.max_frames))

    local_counter = 0
    print(f"[Rank {rank}] Start inference...", flush=True)
    for data_batch in tqdm(dataloader):
        local_counter += 1
        if local_counter % 10 == 0 or local_counter == len(dataloader):
            print(f"[Rank {rank}] Processed {local_counter} / {len(dataloader)} batches", flush=True)

        # for idx in tqdm(per_rank_ids):
        if data_batch is None:
            print("Data is skipped....")
            continue
        prepared_data = dict()
        prepared_data['video'] = data_batch['images']
        video_ids.append(data_batch['file_name'])
        prepared_data['prompt_masks'] = data_batch['prompt_masks']
        prepared_data['prompt_masks_112'] = data_batch['prompt_masks_112']
        prepared_data['text_prompts'] = data_batch['text_prompts'][0]
        prepared_data['text'] = data_batch['qa_list'][0]['input']
        prepared_data['mask_count'] = data_batch['mask_count']

        prediction = {'video_id': data_batch['video_id'], 'video_file': data_batch['file_name']}  # 记录当前的文件ID，文件名字，和文件中的哪个问题，最好再记录下gt，边推理边进行结果指标的计算

        w, h = data_batch['images'][0].size                                 # 记录图像宽高，用于后面padding掩码尺寸
        pred_dict = hf_SAMA_model.predict_forward(**prepared_data, tokenizer=tokenizer, max_frames=args.max_frames, prediction_only=False)

        # pred_dict = model.predict_forward(**prepared_data, tokenizer=tokenizer)

        gt_text.append(data_batch['qa_list'][0]['output'])
        gt_text_cleaned.append(remove_special_tokens(data_batch['qa_list'][0]['output']))
        gt_phases.append(extract_groundings(data_batch['qa_list'][0]['output']))
        gt_masks_mapping = dict()
        for idx, gt_mask_item in enumerate(data_batch['gt_masks']):
            gt_masks_mapping[idx] = gt_mask_item.cpu().numpy()
        gt_masks.append(gt_masks_mapping)
        if 'prediction_masks' not in pred_dict.keys() or pred_dict['prediction_masks'] is None or len(pred_dict['prediction_masks']) == 0:
            print("No SEG !!!")
            # 若没有分割掩码就给一个空tensor
            prediction['prediction_masks'] = torch.zeros((0, h, w), dtype=torch.bool)
            pred_text.append(pred_dict['prediction'])
            pred_text_cleaned.append(remove_special_tokens(pred_dict['prediction']))
            pred_phases.append(extract_groundings(pred_dict['prediction']))
            pred_masks.append({})  # 明确追加空字典
        else:
            pred_text.append(pred_dict['prediction'])
            pred_text_cleaned.append(remove_special_tokens(pred_dict['prediction']))
            pred_phases.append(extract_groundings(pred_dict['prediction']))
            pred_mask_mapping = dict()

            for idx, pred_mask_item in enumerate(pred_dict['prediction_masks']):
                if isinstance(pred_mask_item, torch.Tensor):
                    pred_mask_item = pred_mask_item.cpu().numpy()
                pred_mask_mapping[idx] = pred_mask_item
            pred_masks.append(pred_mask_mapping)

        def generate_color_palette(num_colors):
            """
            返回最多 32 个视觉分离度高、风格统一且专业感强的 RGB 颜色。
            颜色灵感来源于 COCO / Pascal VOC / ColorBrewer 等高质量配色方案。
            """
            color_list = [
                (0, 114, 189),  # blue
                (162, 20, 47),  # deep red
                (237, 177, 32),  # orange
                (126, 47, 142),  # purple
                (119, 172, 48),  # green
                (77, 190, 238),  # sky blue
                (0, 128, 128),  # teal
                (240, 228, 66),  # lemon yellow
                (200, 82, 0),  # burnt orange
                (86, 180, 233),  # light blue
                (204, 121, 167),  # light purple
                (128, 128, 128),  # gray
                (34, 139, 34),  # forest green
                (70, 130, 180),  # steel blue
                (255, 105, 180),  # pink
                (0, 191, 255),  # deep sky blue
                (160, 82, 45),  # saddle brown
                (255, 215, 0),  # gold
                (186, 85, 211),  # medium orchid
                (255, 140, 0),  # dark orange
                (64, 224, 208),  # turquoise
                (255, 99, 71),  # tomato red
                (189, 183, 107),  # khaki
                (70, 70, 70),  # dark gray
                (154, 205, 50),  # yellow green
                (72, 61, 139),  # dark slate blue
                (210, 105, 30),  # chocolate
                (255, 20, 147),  # deep pink
                (112, 128, 144),  # slate gray
                (0, 255, 127),  # spring green
                (139, 0, 139),  # dark magenta
            ]

            if num_colors > len(color_list):
                # 循环重复颜色直到达到所需数量
                repeats = (num_colors + len(color_list) - 1) // len(color_list)
                color_list = (color_list * repeats)[:num_colors]
            return color_list[:num_colors]

        def visualize_masks_on_image(image, masks, phrases, colors, alpha=0.6):
            vis_img = np.array(image).copy()
            for mask, phrase, color in zip(masks, phrases, colors):
                colored_mask = np.zeros_like(vis_img)
                colored_mask[mask.astype(bool)] = (np.array(color)).astype(np.uint8)
                vis_img = cv2.addWeighted(vis_img, 1.0, colored_mask, alpha, 0)
                # 在图片上绘制文本标签
                ys, xs = np.where(mask)
                if len(xs) > 0 and len(ys) > 0:
                    cv2.putText(vis_img, phrase, (int(xs.mean()), int(ys.mean())),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            return vis_img

    import torch.distributed as dist
    barrier()
    print(f"[Rank {rank}] Finished inference.", flush=True)

    import uuid
    if rank == 0:
        shared_tmp_dir = os.path.join(args.model_path, "tmpdir", str(uuid.uuid4())[:8])
        os.makedirs(shared_tmp_dir, exist_ok=True)
    else:
        shared_tmp_dir = ""
    shared_tmp_dir_list = [shared_tmp_dir]
    torch.distributed.broadcast_object_list(shared_tmp_dir_list, src=0)
    tmp_dir = shared_tmp_dir_list[0]
    print(f"All ranks use shared tmp_dir: {shared_tmp_dir}")
    gt_text_all = collect_results_cpu(gt_text, len(dataset), tmpdir=f'{tmp_dir}/gt_text')
    gt_text_cleaned_all = collect_results_cpu(gt_text_cleaned, len(dataset), tmpdir=f'{tmp_dir}/gt_text_cleaned')
    pred_text_all = collect_results_cpu(pred_text, len(dataset), tmpdir=f'{tmp_dir}/pred_text')
    pred_text_cleaned_all = collect_results_cpu(pred_text_cleaned, len(dataset), tmpdir=f'{tmp_dir}/pred_text_cleaned')
    gt_phases_all = collect_results_cpu(gt_phases, len(dataset), tmpdir=f'{tmp_dir}/gt_phases')
    pred_phases_all = collect_results_cpu(pred_phases, len(dataset), tmpdir=f'{tmp_dir}/pred_phases')
    gt_masks_all = collect_results_cpu(gt_masks, len(dataset), tmpdir=f'{tmp_dir}/gt_masks')
    pred_masks_all = collect_results_cpu(pred_masks, len(dataset), tmpdir=f'{tmp_dir}/pred_masks')
    video_ids_all = collect_results_cpu(video_ids, len(dataset), tmpdir=f'{tmp_dir}/video_ids')
    video_mious = []
    video_recalls = []
    video_captions = []
    video_clair_scores = []
    video_metric_dir = os.path.join(args.save_dir, "video_metrics")
    os.makedirs(video_metric_dir, exist_ok=True)
    print(f"[INFO] Rank {rank} len video_ids_all: {len(video_ids_all)}")
    print(f"[INFO] Rank {rank} len pred_masks_all: {len(pred_masks_all)}")
    print(f"[INFO] Rank {rank} len gt_masks_all: {len(gt_masks_all)}")
    print(f"[INFO] Rank {rank} len pred_phases_all: {len(pred_phases_all)}")
    print(f"[INFO] Rank {rank} len gt_phases_all: {len(gt_phases_all)}")
    print(f"[INFO] Rank {rank} len pred_text_cleaned_all: {len(pred_text_cleaned_all)}")
    print(f"[INFO] Rank {rank} len pred_text_all: {len(pred_text_all)}")
    print(f"[INFO] Rank {rank} len gt_text_cleaned_all: {len(gt_text_cleaned_all)}")
    print(f"[INFO] Rank {rank} len gt_text_all: {len(gt_text_all)}")
    # ---------- 在这里统一做截断 ---------- #
    MAX_CAPTION_LEN = 300  # 可根据需要调整，比如 60, 80, 100 都可以
    gt_text_cleaned_all = [truncate_caption(txt, MAX_CAPTION_LEN) for txt in gt_text_cleaned_all]
    pred_text_cleaned_all = [truncate_caption(txt, MAX_CAPTION_LEN) for txt in pred_text_cleaned_all]
    # ----------------------------------- #
    video_to_samples = dict()
    for idx, video_name in enumerate(video_ids_all):
        if video_name not in video_to_samples:
            video_to_samples[video_name] = []
        video_to_samples[video_name].append(idx)
    print(f"[INFO] total video_to_samples length: {len(video_to_samples)}")

    video_items = list(video_to_samples.items())
    per_rank_videos = video_items[rank::world_size]
    print(f"[INFO] Rank {rank} processing {len(per_rank_videos)} data....")
    # for idx, (video_name, sample_indices) in enumerate(video_to_samples.items()):
    for idx, (video_name, sample_indices) in enumerate(per_rank_videos):
        # 替换 / 成 _，确定当前视频的结果文件路径
        safe_video_name = video_name.replace("/", "_")
        save_path = os.path.join(video_metric_dir, f"{safe_video_name}.json")

        # ✅ 如果已存在则直接加载
        if os.path.exists(save_path):
            try:
                with open(save_path, 'r') as f:
                    result = json.load(f)
                video_mious.append(result['miou'])
                video_recalls.append(result['recall'])
                video_clair_scores.append(result['clair'])
                video_captions.append(result['caption_scores'])
                continue
            except Exception as e:
                print(f"[WARNING] Failed to load {save_path}: {e}")

        cur_gt_masks = [gt_masks_all[i] for i in sample_indices]
        cur_pred_masks = [pred_masks_all[i] for i in sample_indices]
        cur_gt_phases = [gt_phases_all[i] for i in sample_indices]
        cur_pred_phases = [pred_phases_all[i] for i in sample_indices]
        cur_gt_texts = [gt_text_cleaned_all[i] for i in sample_indices]
        cur_pred_texts = [pred_text_cleaned_all[i] for i in sample_indices]
        try:
            miou = evaluate_mask_miou(cur_pred_masks, cur_gt_masks)
        except:
            miou = 0.0
        try:
            recall = evaluate_recall_with_mapping(cur_gt_masks, cur_gt_phases, cur_pred_masks, cur_pred_phases,
                                              iou_threshold=0.5, text_sim_threshold=0.5)
        except:
            recall = 0.0
        try:
            caption_metrics = eval_caption_quality(cur_gt_texts, cur_pred_texts)
        except:
            caption_metrics = {
                "Bleu_1": 0.0,
                "Bleu_2": 0.0,
                "Bleu_3": 0.0,
                "Bleu_4": 0.0,
                "METEOR": 0.0,
                "ROUGE_L": 0.0,
                "CIDEr": 0.0,
                "SPICE": 0.0
            }
        try:
            clair_score_video = eval_caption_quality_with_clair(cur_gt_texts, cur_pred_texts)
        except:
            clair_score_video = 0.0
        video_mious.append(miou)
        video_recalls.append(recall)
        video_captions.append(caption_metrics)
        video_clair_scores.append(clair_score_video)

        ###
        cur_miou = video_mious[idx]
        cur_recall = video_recalls[idx]
        cur_caption_metrics = video_captions[idx]
        cur_clair_score = video_clair_scores[idx]
        save_result = {
            "miou": cur_miou,
            "recall": cur_recall,
            "caption_scores": cur_caption_metrics,
            "clair": cur_clair_score,
        }
        with open(save_path, 'w') as f:
            json.dump(save_result, f, indent=2)

        time.sleep(random.uniform(1, 1.25))

    mean_miou = sum(video_mious) / (len(video_mious) + 0.00001)
    mean_recall = sum(video_recalls) / (len(video_recalls) + 0.00001)
    mean_clair_score = sum(video_clair_scores) / (len(video_clair_scores) + 0.00001)
    caption_keys = video_captions[0].keys()
    final_caption_scores = dict()
    for k in caption_keys:
        final_caption_scores[k] = sum(d[k] for d in video_captions) / (len(video_captions)+0.00001)
    final_caption_scores["CLAIR"] = mean_clair_score

    json_base_name = os.path.splitext(os.path.basename(args.json_file))[0]
    save_dir = os.path.join(args.model_path, 'video_gcg_eval_metrics')
    os.makedirs(save_dir, exist_ok=True)

    save_json = os.path.join(save_dir, f"{json_base_name}_rank{rank}.json")

    num_videos = len(video_to_samples)

    rank_result_dict = {
        "Processed Videos": num_videos,
        "Mean IoU (mIoU)": mean_miou,
        "Recall": mean_recall,
    }
    rank_result_dict.update(final_caption_scores)

    with open(save_json, 'w') as f:
        json.dump(rank_result_dict, f, indent=2)

    print(f"\033[92m[RNAK] {rank} Evaluation results saved to {save_json}\033[0m")
    barrier()
    print(f"[INFO] Rank {rank} has been finished processing....")

    if rank == 0:
        from collections import defaultdict

        result_dir = os.path.join(args.model_path, 'video_gcg_eval_metrics')
        pattern_prefix = f"{json_base_name}_rank"

        result_files = [f for f in os.listdir(result_dir)
                        if f.startswith(pattern_prefix) and f.endswith(".json")]
        all_metrics = []
        for fname in result_files:
            with open(os.path.join(result_dir, fname), 'r') as f:
                metrics = json.load(f)
                all_metrics.append(metrics)
        # 加权汇总
        total_videos = sum(m.get("Processed Videos", 0) for m in all_metrics)
        final_scores = defaultdict(float)

        for m in all_metrics:
            count = m.get("Processed Videos", 0)
            for k, v in m.items():
                if k != "Processed Videos":
                    final_scores[k] += v * count

        # 计算加权平均
        for k in final_scores:
            final_scores[k] /= total_videos

        final_scores["Processed Videos"] = total_videos
        final_save_json = os.path.join(args.model_path, 'video_gcg_eval_metrics',
                                       f"video_gcg_eval_{json_base_name}_final.json")
        with open(final_save_json, 'w') as f:
            json.dump(final_scores, f, indent=2)

        print(f"\033[92m[INFO] Aggregated results saved to {final_save_json}\033[0m")


def process_and_save_output(output_dir, image_name, text_output, pred_masks):
    """后处理并将结果保存为json，包括文本和RLE格式的掩码。"""
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # 去除多余字符，如<s>，换行等
    text_output = text_output.replace("<s>", "").replace("\n", "").replace("  ", " ")
    # 如果生成文本里有"ASSISTANT: "，只保留最后一次出现后面的文字
    text_output = text_output.split("ASSISTANT: ")[-1]

    # 用正则去除任何形如<...>的标签，保留干净的文本
    cleaned_str = re.sub(r'<.*?>', '', text_output)

    # 提取形如<p>段落</p>的短语
    pattern = re.compile(r'<p>(.*?)<\/p>')
    phrases = pattern.findall(text_output)
    phrases = [p.strip() for p in phrases]

    # 移除 [SEG] 标记
    cleaned_str = cleaned_str.replace('[SEG]', '')

    # 去掉多余空格，以及首尾的引号
    cleaned_str = ' '.join(cleaned_str.split()).strip("'")
    cleaned_str = cleaned_str.strip()

    # 将模型预测的掩码张量转为CPU，做RLE编码
    pred_masks_tensor = pred_masks.cpu()
    uncompressed_mask_rles = mask_to_rle_pytorch(pred_masks_tensor)
    rle_masks = []
    for m in uncompressed_mask_rles:
        rle_masks.append(coco_encode_rle(m))

    # 构建输出字典，包括图像ID、文本caption、提取到的phrases以及分割掩码
    result_dict = {
        "image_id": image_name[:-4],  # 去掉后缀
        "caption": cleaned_str,
        "phrases": phrases,
        "pred_masks": rle_masks
    }

    # 将结果写到对应json文件
    output_path = f"{output_dir}/{image_name[:-4]}.json"

    with open(output_path, 'w') as f:
        json.dump(result_dict, f)

    return

def mask_to_rle_pytorch(tensor: torch.Tensor):
    """
    将分割掩码编码成未压缩RLE格式，便于使用pycocotools进行后续序列化或解码。
    tensor形状为 (b, h, w)。
    """
    # 首先将(h, w)的维度交换，以便flatten成连续的向量
    b, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 1).flatten(1)

    # 计算相邻像素的变化位置（diff非0的地方）
    diff = tensor[:, 1:] ^ tensor[:, :-1]
    change_indices = diff.nonzero()

    # 将变化位置编码成长度信息
    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = torch.cat(
            [torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device), cur_idxs + 1,
             torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device), ]
        )
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if tensor[i, 0] == 0 else [0]
        counts.extend(btw_idxs.detach().cpu().tolist())
        out.append({"size": [h, w], "counts": counts})

    return out

def coco_encode_rle(uncompressed_rle):
    """将未压缩RLE通过pycocotools的frPyObjects进一步转成标准COCO RLE（counts二进制字符串）"""
    h, w = uncompressed_rle["size"]
    rle = mask_utils.frPyObjects(uncompressed_rle, h, w)
    # pycocotools生成的counts是bytes，需要解码成UTF-8字符串方可JSON化
    rle["counts"] = rle["counts"].decode("utf-8")

    return rle

if __name__ == '__main__':
    start_time = time.time()  # 【Added Timing Code】记录程序开始时间
    main()
    end_time = time.time()  # 【Added Timing Code】记录程序结束时间
    total_time = end_time - start_time  # 【Added Timing Code】计算总耗时
    formatted_time = time.strftime("%H:%M:%S", time.gmtime(total_time))  # 【Added Timing Code】
    print(f"Main function run time: {formatted_time}")  # 【Added Timing Code】输出总耗时（时:分:秒格式）
