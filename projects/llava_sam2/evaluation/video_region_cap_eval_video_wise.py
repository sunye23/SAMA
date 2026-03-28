import argparse  # 用于解析命令行参数
import copy
import json
import math
import os
import pickle
import re
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
    parser.add_argument('--max_frames', type=int, default=128)
    parser.add_argument('--json_file', type=str, required=True,
                        help='path to json file to process')  # ✨新增
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

DETAILED_QUESTIONS =  [
    'Please describe the region <mask> in the video in detail.',
]

def truncate_caption(caption, max_words=200):
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
    def __init__(self, video_folder, data_json, description_per_item=5, max_frames=32, args=None):
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
        self.dataset_dicts = self.json_file_preprocess(data_json, args)
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
            # 支持多个数据集筛选，如 mevis,lvvis 或 all
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

    def _get_region_infos(self, masks):
        # masks tensor, (n_obj, h, w)
        masks_112_112 = F.interpolate(masks.unsqueeze(0), size=(112, 112), mode='nearest').squeeze(0)
        masks = F.interpolate(
            masks.unsqueeze(0),
            size=(int(self.image_size // self.patch_size * self.downsample_ratio),
                  int(self.image_size // self.patch_size * self.downsample_ratio)),
            mode='nearest').squeeze(0)
        return masks, masks_112_112

    def _process_conversation(self, descriptions, mask_info, sparse_len):
        conversation = []
        question_list = []
        frame_token_str = f'{self.IMG_START_TOKEN}' \
                          f'{self.IMG_CONTEXT_TOKEN * self.patch_token}' \
                          f'{self.IMG_END_TOKEN}'

        num_context_token = sparse_len
        # if self.window_size > 0:
        #     num_context_token = math.ceil(num_context_token * 32 / self.window_stride) * 32
        # else:
        #     num_context_token = 32
        # context_str = ''
        context_str = self.IMG_CONTEXT_TOKEN * num_context_token + '\n'
        for index, (key, value) in enumerate(descriptions.items()):
            question = random.choice(DETAILED_QUESTIONS)
            question_list.append(question.strip())
            match = re.search(r"obj(\d+)", key)
            obj_num = int(match.group(1))
            if index == 0:
                frame_tokens = frame_token_str + '\n'
                # frame_tokens = '=' + ' 's
                frame_tokens = frame_tokens * self.dense_len
                frame_tokens = frame_tokens.strip()
                question =  frame_tokens + context_str + question
            else:
                question = context_str + question
            region_str = self.VP_START_TOKEN + self.IMG_CONTEXT_TOKEN * 2 + self.VP_END_TOKEN
            question = question.replace("<mask>", region_str) + self.LIMIT
            question = question.strip()
            answer = value.strip()
            conversation.append({'input': question, 'output': answer})
        return conversation, question_list

    def __getitem__(self, index):
        record = self.dataset_dicts[index]
        data_dict = dict()
        mask_info = dict()
        data_dict['file_name'] = record['file_name']    # "mevis/train/JPEGImages/0a2c0bfe6909"
        data_dict['video_id'] = record['id']
        data_dict['frames'] = record['frames']          # 帧名列表
        masks_mapping = record['annotation']  # anno_map
        data_dict['description'] = record['description']
        data_dict['description_id'] = record['description_id']
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
                processed_mask, processed_mask_112 = self._get_region_infos(obj_masks_all)
                mask_info[temp_obj_num]['mask'] = obj_masks_all
                mask_info[temp_obj_num]['processed_mask'] = processed_mask
                mask_info[temp_obj_num]['processed_mask_112'] = processed_mask_112
                mask_info[temp_obj_num]['height'] = ori_height
                mask_info[temp_obj_num]['width'] = ori_width
                prompt_masks.append(processed_mask)
                prompt_masks_112.append(processed_mask_112)
            else:
                print("mask exist...")
                return None
        sparse_len = len(get_sparse_indices_uniform(len(frame_indices), max_frames=self.max_frames))

        conversations, question_list = self._process_conversation(data_dict['description'], mask_info, sparse_len)
        data_dict['mask_count'] = [1 for i in range(len(question_list))]
        data_dict['prompt_masks'] = prompt_masks
        data_dict['prompt_masks_112'] = prompt_masks_112
        data_dict['text_prompts'] = question_list
        data_dict['qa_list'] = conversations
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

def main():
    args = parse_args()
    video_folder = '/sama_bench'
    json_file = args.json_file
    dataset = GCGVideoInferenceDataset(video_folder, json_file, description_per_item=1, max_frames=args.max_frames, args=args)  # 【修改2】传入args.dataset

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
        num_workers=8,
        pin_memory=False,
        collate_fn=lambda x: x[0]
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

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    results = []                                # 用于收集所有推理的文本结果
    n_samples = len(dataset)                    # 数据集中需要推理的图像数量
    per_rank_samples = math.ceil(n_samples / world_size) + 1  # 每个进程要处理的图像数量（多加1是保险）
    per_rank_ids = range(per_rank_samples * rank,
                         min(n_samples, per_rank_samples * (rank + 1)))  # 当前进程实际需要处理的索引范围

    gt_text = []
    pred_text = []
    gt_text_cleaned = []
    pred_text_cleaned = []

    video_ids = []
    for data_batch in tqdm(dataloader):
        if data_batch is None:
            print("Data is skipped....")
            continue
        prepared_data = dict()
        prepared_data['video'] = data_batch['images']

        prepared_data['prompt_masks'] = data_batch['prompt_masks']
        prepared_data['prompt_masks_112'] = data_batch['prompt_masks_112']
        prepared_data['text_prompts'] = data_batch['text_prompts'][0]
        prepared_data['text'] = data_batch['qa_list'][0]['input']
        prepared_data['mask_count'] = data_batch['mask_count']
        description_id = data_batch['description_id']
        description = data_batch['description']
        file_name = data_batch['file_name']
        video_ids.append(data_batch['file_name'])
        prediction = {'video_id': data_batch['video_id'], 'video_file': data_batch['file_name']}  # 记录当前的文件ID，文件名字，和文件中的哪个问题，最好再记录下gt，边推理边进行结果指标的计算

        w, h = data_batch['images'][0].size                                 # 记录图像宽高，用于后面padding掩码尺寸
        pred_dict = hf_SAMA_model.predict_forward(**prepared_data, tokenizer=tokenizer, prediction_only=True, max_frames=args.max_frames)
        # save data
        prediction_dir = os.path.join(args.save_dir, "prediction")
        os.makedirs(prediction_dir, exist_ok=True)

        # 构建保存内容
        prediction_data = {
            "pred_text_cleaned": remove_special_tokens(pred_dict['prediction']),
            "description": description,
            "file_name": file_name,
            "description_id": description_id,
        }

        # 保存成 JSON 文件
        output_path = os.path.join(prediction_dir, f"{description_id}.json")
        with open(output_path, 'w') as f:
            json.dump(prediction_data, f, ensure_ascii=False, indent=2)
        # save data end
        gt_text.append(data_batch['qa_list'][0]['output'])
        gt_text_cleaned.append(remove_special_tokens(data_batch['qa_list'][0]['output']))
        pred_text.append(pred_dict['prediction'])
        pred_text_cleaned.append(remove_special_tokens(pred_dict['prediction']))

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

    gt_text_all = collect_results_cpu(gt_text, len(dataset), tmpdir=f'{tmp_dir}/gt_text')
    gt_text_cleaned_all = collect_results_cpu(gt_text_cleaned, len(dataset), tmpdir=f'{tmp_dir}/gt_text_cleaned')
    pred_text_all = collect_results_cpu(pred_text, len(dataset), tmpdir=f'{tmp_dir}/pred_text')
    pred_text_cleaned_all = collect_results_cpu(pred_text_cleaned, len(dataset), tmpdir=f'{tmp_dir}/pred_text_cleaned')
    video_ids_all = collect_results_cpu(video_ids, len(dataset), tmpdir=f'{tmp_dir}/video_ids')

    video_captions = []
    video_clair_scores = []
    video_metric_dir = os.path.join(args.save_dir, "video_caption_metrics")
    os.makedirs(video_metric_dir, exist_ok=True)
    print(f"[INFO] Rank {rank} len gt_text_all: {len(gt_text_all)}")
    print(f"[INFO] Rank {rank} len gt_text_cleaned_all: {len(gt_text_cleaned_all)}")
    print(f"[INFO] Rank {rank} len pred_text_all: {len(pred_text_all)}")
    print(f"[INFO] Rank {rank} len pred_text_cleaned_all: {len(pred_text_cleaned_all)}")

    MAX_CAPTION_LEN = 300
    gt_text_cleaned_all = [truncate_caption(txt, MAX_CAPTION_LEN) for txt in gt_text_cleaned_all]
    pred_text_cleaned_all = [truncate_caption(txt, MAX_CAPTION_LEN) for txt in pred_text_cleaned_all]

    video_to_samples = dict()
    for idx, video_name in enumerate(video_ids_all):
        if video_name not in video_to_samples:
            video_to_samples[video_name] = []
        video_to_samples[video_name].append(idx)
    print(f"[INFO] total video_to_samples length: {len(video_to_samples)}")

    video_items = list(video_to_samples.items())
    per_rank_videos = video_items[rank::world_size]
    print(f"[INFO] Rank {rank} processing {len(per_rank_videos)} data....")

    for idx, (video_name, sample_indices) in enumerate(per_rank_videos):
        # 替换 / 成 _，确定当前视频的结果文件路径
        safe_video_name = video_name.replace("/", "_")
        save_path = os.path.join(video_metric_dir, f"{safe_video_name}.json")

        if os.path.exists(save_path):
            try:
                with open(save_path, 'r') as f:
                    result = json.load(f)
                video_clair_scores.append(result['clair'])
                video_captions.append(result['caption_scores'])
                continue
            except Exception as e:
                print(f"[WARNING] Failed to load {save_path}: {e}")

        cur_gt_texts = [gt_text_cleaned_all[i] for i in sample_indices]
        cur_pred_texts = [pred_text_cleaned_all[i] for i in sample_indices]
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
            print(f"[INFO] caption_metrics: {caption_metrics}")
        try:
            clair_score_video = eval_caption_quality_with_clair(cur_gt_texts, cur_pred_texts)
        except:
            clair_score_video = 0.0
        video_captions.append(caption_metrics)
        video_clair_scores.append(clair_score_video)

        ###
        cur_caption_metrics = video_captions[idx]
        cur_clair_score = video_clair_scores[idx]
        save_result = {
            "caption_scores": cur_caption_metrics,
            "clair": cur_clair_score,
        }
        with open(save_path, 'w') as f:
            json.dump(save_result, f, indent=2)

        time.sleep(random.uniform(1, 1.25))
    mean_clair_score = sum(video_clair_scores) / len(video_clair_scores)
    caption_keys = video_captions[0].keys()
    final_caption_scores = dict()
    for k in caption_keys:
        final_caption_scores[k] = sum(d[k] for d in video_captions) / len(video_captions)
    final_caption_scores["CLAIR"] = mean_clair_score

    json_base_name = os.path.splitext(os.path.basename(args.json_file))[0]
    save_dir = os.path.join(args.model_path, 'video_caption_final_metrics')
    os.makedirs(save_dir, exist_ok=True)

    save_json = os.path.join(save_dir, f"{json_base_name}_rank{rank}.json")

    num_videos = len(video_to_samples)
    rank_result_dict = {
        "Processed Videos": num_videos,
    }
    rank_result_dict.update(final_caption_scores)

    with open(save_json, 'w') as f:
        json.dump(rank_result_dict, f, indent=2)

    print(f"\033[92m[RNAK] {rank} Evaluation results saved to {save_json}\033[0m")
    barrier()
    print(f"[INFO] Rank {rank} has been finished processing....")

    if rank == 0:
        from collections import defaultdict
        # 新的结果文件夹路径
        result_dir = os.path.join(args.model_path, 'video_caption_final_metrics')
        pattern_prefix = f"{json_base_name}_rank"

        # 获取所有 rank 结果文件
        result_files = [f for f in os.listdir(result_dir) if f.startswith(pattern_prefix) and f.endswith(".json")]

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

        final_save_json = os.path.join(args.model_path, 'video_caption_final_metrics', f"{json_base_name}_final.json")
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
    h, w = uncompressed_rle["size"]
    rle = mask_utils.frPyObjects(uncompressed_rle, h, w)
    rle["counts"] = rle["counts"].decode("utf-8")

    return rle

import time
if __name__ == '__main__':
    start_time = time.time()  # 【Added Timing Code】记录程序开始时间
    main()
    end_time = time.time()  # 【Added Timing Code】记录程序结束时间
    total_time = end_time - start_time  # 【Added Timing Code】计算总耗时
    formatted_time = time.strftime("%H:%M:%S", time.gmtime(total_time))  # 【Added Timing Code】
    print(f"Main function run time: {formatted_time}")  # 【Added Timing Code】输出总耗时（时:分:秒格式）
