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
import matplotlib.colors as mcolors  # 确认添加此行
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
# from projects.llava_sam2.datasets.VideoRegionConv_Dataset import VideoRegionConvDataset
# from projects.llava_sam2.datasets.VideoRegionCaptioning_Dataset import VideoRegionDescriptionDataset
from projects.llava_sam2.evaluation.utils.clair import clair
from projects.llava_sam2.evaluation.eval_gcg_metrics_demo import (evaluate_mask_miou, evaluate_recall_with_mapping,
                                                                  eval_caption_quality, eval_caption_quality_with_clair)
from PIL import ImageDraw
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

COLOR = [
    'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta',
    'lime', 'pink', 'brown', 'beige', 'navy', 'teal', 'violet', 'gold',
    'silver', 'coral', 'salmon', 'indigo'
]

# todo 这里的代码需要改进，生成的模型和这里的参数需要一致
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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
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
from google import genai
from google.genai import types
def annotate_video_with_gemini(system_prompt):
    # genai.configure(api_key=GEMINI_API_KEY)
    # model = genai.GenerativeModel(model_name="models/gemini-1.5-pro")
    client = genai.Client(api_key=GEMINI_API_KEY)
    try:
        response = client.models.generate_content(
            # model="gemini-2.0-flash",
            model="gemini-1.5-pro",
            contents=system_prompt)
        return response
    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        return None
def parse_gemini_response(raw: str) -> str:
    """
    解析 Gemini 返回：
    1) 若含合法 JSON 且包含 'answer' 字段，则取其值
    2) 否则返回原文本（去首尾空白）
    """
    if raw is None:
        return ""
    raw = raw.strip()
    # 先尝试整体 JSON
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "answer" in obj:
            return obj["answer"]
    except json.JSONDecodeError:
        pass
    # 再用正则找最外层 {…}
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if match:
        try:
            obj = json.loads(match.group())
            if "answer" in obj:
                return obj["answer"]
        except json.JSONDecodeError:
            pass
    return raw

def sample_five_indices(n: int) -> list[int]:
    """返回长度为 ≤5 的均匀索引列表，保证含首帧(0)。"""
    if n <= 5:
        return list(range(n))
    idx = np.linspace(0, n - 1, num=5, dtype=int)
    # np.linspace 可能因取整导致重复，去重后再补齐
    idx = sorted(set(idx))
    while len(idx) < 5:                       # 极端小 n 情况补帧
        for add in range(1, n):
            if len(idx) == 5:
                break
            if add not in idx:
                idx.append(add)
    return sorted(idx)


# todo: 三个数据集 1. Region-level Video description 2. Video grounded chat 3. hallucination mutli-choice QA
class GCGVideoInferenceDataset:
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
    def __init__(self, video_folder, data_json, description_per_item=5):
        """
        参数:
          video_folder: 存放视频帧的根目录，比如 /remote-home/sunye/video_project/video_dataset/mevis/train/JPEGImages/
          data_json: 包含所有视频及对话信息的JSON文件路径 (你的示例文件)
          qa_per_item: 每个记录最多包含多少轮问答（默认为 5）
        """
        self.description_per_item = 1
        self.video_folder = video_folder
        self.image_size = 448
        self._system = ''
        self.downsample_ratio = 0.5
        self.dense_len = 5
        patch_size = 14
        self.patch_size = patch_size
        self.patch_token = int((self.image_size // patch_size) ** 2 * (self.downsample_ratio ** 2))
        self.dataset_dicts = self.json_file_preprocess(data_json)
        mevis_json = '/SAMA/reproduce/SAMA/video_region_meta/mask_dict/Val/Mevis_mask_dict_val.json'
        lvvis_json = '/SAMA/reproduce/SAMA/video_region_meta/mask_dict/Val/LVVIS_mask_dict_val.json'
        ref_youtube_json = '/SAMA/reproduce/SAMA/video_region_meta/mask_dict/Val/RefYoutube_mask_dict_val.json'
        vidstg_json = '/SAMA/reproduce/SAMA/video_region_meta/mask_dict/Val/VidSTG_mask_dict_val_updated.json'

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

    def json_file_preprocess(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        record_index = 0
        dataset_dicts = []
        description_per_data = self.description_per_item
        for video in tqdm(data, desc='Registering VideoRegionCaptioning BenchMark'):
            video_info = {
                'video': video.get('video', None),
                'video_id': video.get('video_id', None),
                'height': video.get('height', None),
                'width': video.get('width', None),
                'frames': video.get('frames', None),
                'annotation': video.get('anno_map', None),
                'des_id': video.get('description_id', None),
            }
            # if 'VidSTG' in video_info['video']:
            #     continue
            # if 'lvvis' in video_info['video']:
            #     continue
            # if 'ref_youtube_vos' in video_info['video']:
            #     continue
            # if 'mevis' in video_info['video']:
            #     continue
            if 'VidSTG' in video_info['video']:
                video_info['frames'] = [
                    os.path.splitext(item["raw_filename"])[0] if isinstance(item, dict) and "raw_filename" in item
                    else os.path.splitext(item)[0]
                    for item in video_info['frames']
                ]
            new_dict = dict()
            descriptions = video.get('description', {})
            for index, (obj_key, description_text) in enumerate(descriptions.items()):
                new_dict[obj_key] = description_text

                description_id = video_info['des_id'][obj_key]
                if index + 1 == len(descriptions) or (index + 1) % description_per_data == 0 or len(descriptions) == 1:
                    record = {
                        'file_name': video_info['video'],
                        'id': record_index,
                        'height': video_info['height'],
                        'width': video_info['width'],
                        'frames': video_info['frames'],
                        'annotation': video_info['annotation'],
                        'description': new_dict,
                        'description_id': description_id,
                    }
                    dataset_dicts.append(record)
                    record_index = record_index + 1
                    new_dict = dict()
        print(f"**Total des evaluation length: {len(dataset_dicts)}")
        print(f"**Total des evaluation length: {len(dataset_dicts)}")
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

    def __getitem__(self, index):
        record = self.dataset_dicts[index]
        data_dict = dict()
        mask_info = dict()
        data_dict['file_name'] = record['file_name']    # "mevis/train/JPEGImages/0a2c0bfe6909"
        data_dict['video_id'] = record['id']
        data_dict['description_id'] = record['description_id']
        data_dict['frames'] = record['frames']          # 帧名列表
        masks_mapping = record['annotation']  # anno_map
        data_dict['description'] = record['description']
        frames_files = data_dict['frames']
        video_id = data_dict['video_id']
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
        frames_files = frames_files if len(frames_files) <= 5 else [frames_files[0]] + [frames_files[i] for i in
                                                                                        np.linspace(1,
                                                                                                    len(frames_files) - 1,
                                                                                                    4, dtype=int)]

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
        prompt_masks = []
        prompt_masks_112 = []
        mask_count = []

        for obj_key, value in data_dict['description'].items():
            # obj_list = []
            obj_list_all = []
            match = re.search(r"obj(\d+)", obj_key)
            obj_num = int(match.group(1))
            mask_annotation = mask_dict[str(masks_mapping[obj_num])]
            if 'lvvis' in file_name:
                mask_annotation = mask_annotation.get("segmentations", [])
            else:
                mask_annotation = mask_annotation
            frames_masks_ = []
            frames_masks_all = []
            for frame_idx in frame_indices:
                if frame_idx < len(mask_annotation) and mask_annotation[frame_idx] is not None:
                    frames_masks_all.append(copy.deepcopy(mask_annotation[frame_idx]))
                else:
                    empty_array = np.zeros((ori_height, ori_width), dtype=np.uint8)
                    empty_mask = maskUtils.encode(np.asfortranarray(empty_array))
                    frames_masks_all.append(empty_mask)
            obj_list_all.append(frames_masks_all)
            obj_masks_all = self.decode_mask([obj_list_all], image_size=(ori_height, ori_width))

            if obj_masks_all is None:
                zero_mask = np.zeros((len(frame_indices), ori_height, ori_width), dtype=np.uint8)
                obj_masks_all = torch.from_numpy(zero_mask).float()
            if obj_num not in mask_info:
                temp_obj_num = str(obj_num)
                mask_info[temp_obj_num] = dict()
                mask_info[temp_obj_num]['mask'] = obj_masks_all
                mask_info[temp_obj_num]['height'] = ori_height
                mask_info[temp_obj_num]['width'] = ori_width
            else:
                print("mask exist...")
                return None
        raw_bbox_list = []
        bbox_list = []
        mask_list = []
        COLOR = [
            'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta',
            'lime', 'pink', 'brown', 'beige', 'navy', 'teal', 'violet', 'gold',
            'silver', 'coral', 'salmon', 'indigo'
        ]
        system_prompt = 'You are giving five frames sampled from a video. The first frame shows the colored box highlighting the interested object.' \
        ' You need to describe the interested object based on the video content. ' \
        'Answer without mentioning any box or color words as if observing the object in real-time. Response in JSON format like: {"answer":"..."}. ' \
        'Question: '
        question = system_prompt + "Provide me with a description of the <mask>."
        color_index = 0
        obj_colors = []
        image_draw = data_dict['images'][0]
        draw = ImageDraw.Draw(image_draw)
        for key, value in mask_info.items():
            single_mask = mask_info[key]['mask'][0]
            if single_mask.sum() == 0:
                # ⚠️ 构造一个中心小 box 占位
                center_x, center_y = ori_width // 2, ori_height // 2
                box_size = 10
                min_col = max(center_x - box_size // 2, 0)
                max_col = min(center_x + box_size // 2, ori_width - 1)
                min_row = max(center_y - box_size // 2, 0)
                max_row = min(center_y + box_size // 2, ori_height - 1)
                # 构造假的 mask
                placeholder_mask = np.zeros((ori_height, ori_width), dtype=np.uint8)
                placeholder_mask[min_row:max_row + 1, min_col:max_col + 1] = 1
                single_mask = torch.from_numpy(placeholder_mask).float()
            else:
                nonzero = torch.nonzero(single_mask, as_tuple=False)
                min_row = nonzero[:, 0].min().item()
                max_row = nonzero[:, 0].max().item()
                min_col = nonzero[:, 1].min().item()
                max_col = nonzero[:, 1].max().item()

            color = COLOR[color_index % len(COLOR)]
            color_index += 1

            # 在图像上绘制边框
            # 注意：PIL 里坐标是 (x, y) = (列, 行)，即 (min_col, min_row) 到 (max_col, max_row)
            draw.rectangle(
                [(min_col, min_row), (max_col, max_row)],
                outline=color,
                width=3
            )
            obj_colors.append(color)


        # 用 obj_colors 替换 <mask>
        processed_question = process_question_with_colors(question, obj_colors)
        data_dict['processed_question'] = processed_question
        image_np = np.array(image_draw)  # 转为 numpy 数组
        image_restored = Image.fromarray(image_np)  # 重新转为 PIL.Image
        data_dict['images'][0] = image_restored
        return data_dict


def process_question_with_colors(q_text, colors_list):
    parts = q_text.split("<mask>")
    # 如果 question 中没有 <mask>，直接返回原文本
    if len(parts) == 1:
        return q_text

    new_text = parts[0]
    # 每个 <mask> 都要被替换
    for i in range(len(parts) - 1):
        # 如果颜色数用完了，可根据需求决定： as
        # 1) 设一个默认颜色, 比如 "some_color"
        # 2) 或者只替换已有颜色，剩下的 <mask> 保留不动
        if i < len(colors_list):
            color_str = colors_list[i]
        else:
            color_str = "some_color"
        replacement = f"the object within {color_str} box"
        new_text += replacement + parts[i + 1]
    return new_text

def extract_groundings(sentence, start_token="<p>", end_token="</p>"):
    pattern = re.escape(start_token) + r"(.*?)" + re.escape(end_token)
    matches = re.findall(pattern, sentence, flags=re.DOTALL)
    return list(map(lambda s: s.strip().lower(), matches))

def remove_special_tokens(sentence, tokens=None):
    if tokens is None:
        tokens = ["<p>", "</p>", "[SEG]"]
    # 构造正则表达式，匹配所有待去除的标识符
    pattern = "(" + "|".join(map(re.escape, tokens)) + ")"
    # 替换标识符为空字符串
    cleaned = re.sub(pattern, "", sentence)
    # 合并连续的空白字符为一个空格，并去掉首尾空白
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def main():
    args = parse_args()  # 解析命令行参数
    video_folder = '/SAMA/reproduce/Benchmark'
    json_file = '/SAMA/reproduce/SAMA/video_region_meta/BenchMark_merged_final_with_qid_with_descid.json'
    dataset = GCGVideoInferenceDataset(video_folder, json_file, description_per_item=1)

    # 根据launcher参数决定是否做分布式初始化
    if args.launcher != 'none':
        _init_dist_pytorch('nccl')             # 初始化pytorch分布式模式，后端为NCCL
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
        num_workers=16,
        pin_memory=False,
        collate_fn=lambda x:x[0],
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

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
        transforms.Resize((384, 384), interpolation=InterpolationMode.BICUBIC),
    ])
    index = 0
    output_base_dir = '/SAMA/reproduce/Benchmark/evaluation/SAMA_prediction/video_gcg/gemini_1_5_pro_description'
    os.makedirs(output_base_dir, exist_ok=True)
    for data_batch in tqdm(dataloader):
        instruction = data_batch['processed_question']  # ✅ as
        images = data_batch['images']
        file_name = data_batch['file_name']
        record_id = data_batch['description_id']
        prompts = []
        prompts.append(instruction)
        # 保证每张图像都是 CUDA 上的 tensor
        pixel_values = [transform(img) for img in images]
        for item in pixel_values:
            prompts.append(item)
        response = annotate_video_with_gemini(prompts)
        if response is None:
            response = ''
        else:
            response = parse_gemini_response(response.text)#
            # === 保存答案（示例：写入 save_dir/{record_id}.txt）===
        text_json_path = os.path.join(output_base_dir, f"{record_id}.json")
        with open(text_json_path, 'w') as f_json:
            json.dump({"prediction": response}, f_json)
        time.sleep(random.uniform(1.0, 1.5))  # 随机暂停 1–1.5 s


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
