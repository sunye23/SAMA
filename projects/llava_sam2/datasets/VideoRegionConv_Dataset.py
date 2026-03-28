import logging
import os
from typing import Literal

import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from mmengine import print_log
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import pickle
from xtuner.registry import BUILDER
from xtuner.dataset.huggingface import build_origin_dataset
import copy
import re
from tqdm import tqdm
from .encode_fn import video_lisa_encode_fn
import json
import random
import pycocotools.mask as maskUtils
import cv2
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import torch.nn.functional as F
from projects.llava_sam2.datasets.utils import get_sparse_indices, get_dense_indices, get_sparse_indices_uniform
import math

SEG_QUESTIONS = [
    "Can you segment the {class_name} in this image?",
    "Please segment {class_name} in this image.",
    "What is {class_name} in this image? Please respond with segmentation mask.",
    "What is {class_name} in this image? Please output segmentation mask.",

    "Can you segment the {class_name} in this image",
    "Please segment {class_name} in this image",
    "What is {class_name} in this image? Please respond with segmentation mask",
    "What is {class_name} in this image? Please output segmentation mask",

    "Could you provide a segmentation mask for the {class_name} in this image?",
    "Please identify and segment the {class_name} in this image.",
    "Where is the {class_name} in this picture? Please respond with a segmentation mask.",
    "Can you highlight the {class_name} in this image with a segmentation mask?",

    "Could you provide a segmentation mask for the {class_name} in this image",
    "Please identify and segment the {class_name} in this image",
    "Where is the {class_name} in this picture? Please respond with a segmentation mask",
    "Can you highlight the {class_name} in this image with a segmentation mask",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]
def get_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    frames = []

    frame_id = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

        frame_id += 1

    cap.release()
    return frames

class VideoRegionConvDataset(Dataset):
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

    def __init__(self,
                 image_folder,
                 extra_image_processor=None,
                 tokenizer=None,
                 # offline_processed_text_folder=None,
                 template_map_fn=None,
                 max_length=2048,
                 lazy=True,
                 repeats=1,
                 special_tokens=None,
                 frame_contiguous_sample=False,
                 use_fast=False,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 data_path=None,
                 #####
                 max_conv_len=None,
                 dense_len=None,
                 qa_per_item=None,
                 sparse_len=None,
                 num_temporal_token=None,
                 num_spatial_token=None,
                 window_stride=None,
                 window_size=None,
                 mevis_json=None,
                 lvvis_json=None,
                 ref_youtube_pkl=None,
                 sav_json=None,
                 vidstg_json=None,
                 # only work if use_fast = True
                 n_fast_images=50,
                 fast_pool_size=4,
                 fast_token_after_question=False,
                 max_frames=32,
                 selected_datasets=None,
    ):
        assert lazy is True
        self.tokenizer = BUILDER.build(tokenizer)
        self.dense_len = dense_len
        self.qa_per_item = qa_per_item
        self.max_frames = max_frames
        self.sparse_len = sparse_len
        self.num_temporal_token = num_temporal_token
        self.num_spatial_token = num_spatial_token
        self.window_stride = window_stride
        self.window_size = window_size

        self.selected_datasets = selected_datasets
        self.lazy = lazy
        self.max_conv_len = max_conv_len

        self.max_length = max_length

        self.template_map_fn = template_map_fn
        if isinstance(self.template_map_fn, dict) and self.lazy:
            _type = self.template_map_fn['type']
            del self.template_map_fn['type']
            self.template_map_fn = _type(**self.template_map_fn)

        self.arch_type = arch_type
        if self.arch_type == 'qwen':
            self.IMG_CONTEXT_TOKEN = '<|image_pad|>'
            self.IMG_START_TOKEN = '<|vision_start|>'
            self.IMG_END_TOKEN = '<|vision_end|>'
        elif self.arch_type == 'llava':
            self.IMG_CONTEXT_TOKEN = '<image>'
            self.IMG_START_TOKEN = ''
            self.IMG_END_TOKEN = ''

        self.mask_dicts = dict()
        with open(mevis_json, 'r') as fp:
            self.mask_dicts['mevis'] = json.load(fp)
        with open(lvvis_json, 'r') as fp:
            self.mask_dicts['lvvis'] = json.load(fp)
        with open(ref_youtube_pkl, 'rb') as fp:
            self.mask_dicts['ref_youtube'] = pickle.load(fp)
        with open(sav_json, 'r') as fp:
            self.mask_dicts['sav'] = json.load(fp)
        with open(vidstg_json, 'r') as fp:
            self.mask_dicts['vidstg'] = json.load(fp)
        print("ALL mask dict json file has been loaded....")

        json_data = self.json_file_preprocess(data_path)
        self.text_data = json_data
        random.Random(0).shuffle(self.text_data)
        self.image_folder = image_folder
        if extra_image_processor is not None:
            self.extra_image_processor = BUILDER.build(extra_image_processor)
        self.down_ratio = 1
        self.repeats = repeats

        self._system = ''

        self.downsample_ratio = 0.5
        if self.arch_type == 'llava':
            self.downsample_ratio = 1
        self.image_size = 448
        if self.arch_type == 'llava':
            self.image_size = 336
        patch_size = 14
        self.patch_size = patch_size
        self.patch_token = int((self.image_size // patch_size) ** 2 * (self.downsample_ratio ** 2))
        if self.arch_type == 'qwen':
            self.patch_token = 1

        if preprocessor is None:
            self.transformer = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
            ])
            self.preprocessor = None
        else:
            self.transformer = None
            self.preprocessor = BUILDER.build(preprocessor)

        if special_tokens is not None:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        self.use_fast = use_fast
        self.n_fast_images = n_fast_images
        self.fast_pool_size = fast_pool_size

        self.frame_contiguous_sample = frame_contiguous_sample

        # for visualization debug
        self.save_folder = './work_dirs/video_debug/'
        self.cur_number = 0

        # exist_thr
        self.exist_thr = 8
        self.fast_token_after_question = fast_token_after_question
        if self.fast_token_after_question:
            assert self.use_fast

    def __len__(self):
        return len(self.text_data) * self.repeats

    @property
    def modality_length(self):
        length_list = []
        for data_dict in self.text_data:
            if self.lazy:
                cur_len = 100
            else:
                cur_len = len(data_dict['input_ids'])
                if data_dict.get('image', None) is None:
                    cur_len = -cur_len
            length_list.append(cur_len)
        return length_list * self.repeats

    def real_len(self):
        return len(self.text_data)

    def json_file_preprocess(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as fp:
            data = json.load(fp)

        dataset_dicts = []
        data_id = 0
        filter_num = 0
        sum_total = 0
        round_per_data = self.qa_per_item
        for video in tqdm(data, desc='Registering VideoRegionConv videos'):
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

            if video_info['conversations'] is None:
                continue

            if self.selected_datasets is not None:
                flag = False
                for item in self.selected_datasets:
                    if item in video_info['video']:
                        flag = True
                if flag is True:
                    continue

            conversations = video_info['conversations']

            if len(conversations) % 2 != 0:
                conversations = conversations[:-1]

            total_rounds = len(conversations) // 2
            sum_total = sum_total + total_rounds
            num_splits = (total_rounds + round_per_data - 1) // round_per_data

            for i in range(num_splits):
                start_idx = i * round_per_data * 2
                end_idx = min(start_idx + round_per_data * 2, len(conversations))
                sub_conversations = conversations[start_idx:end_idx]
                sub_obj_masks = video_info['obj_masks'][start_idx // 2:end_idx // 2]
                sub_groundtruth = video_info['groundtruth'][start_idx // 2:end_idx // 2]

                cleaned_conversations = []
                cleaned_obj_masks = []
                cleaned_groundtruth = []
                current_round = min(round_per_data, len(sub_conversations) // 2)
                total_seg = 0
                for idx_conv in range(current_round):
                    current = idx_conv * 2
                    question = sub_conversations[current]
                    if question['from'] == 'human':
                        question_value = question['value']
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
                    total_seg = total_seg + num_seg
                    cleaned_conversations.append(question)
                    cleaned_conversations.append(answer)
                    cleaned_obj_masks.append(current_obj_mask)
                    cleaned_groundtruth.append(current_gt)
                if total_seg == 0:
                    filter_num = filter_num + len(cleaned_obj_masks)
                    continue
                if 'VidSTG' in video_info['video']:
                    video_info['frames'] = [
                        os.path.splitext(item["raw_filename"])[0] if isinstance(item, dict) and "raw_filename" in item
                        else os.path.splitext(item)[0]
                        for item in video_info['frames']
                    ]

                if len(cleaned_conversations) != 0:
                    record = {
                        'id': data_id,
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
        print("Total conversation num: ", sum_total, "filtered conversation num: ", filter_num)

        return dataset_dicts


    def save_visualization_results(self, data_dict):
        base_folder = "/Benchmark/processed_json/Updated/visualization/"
        video_folder_name = data_dict.get("file_name", "unknown_video")
        video_folder = os.path.join(base_folder, video_folder_name)
        os.makedirs(video_folder, exist_ok=True)

        frames_folder = os.path.join(video_folder, "frames")
        gt_masks_folder = os.path.join(video_folder, "gt_masks")
        prompt_masks_folder = os.path.join(video_folder, "prompt_masks_112")
        text_folder = os.path.join(video_folder, "text")
        os.makedirs(frames_folder, exist_ok=True)
        os.makedirs(gt_masks_folder, exist_ok=True)
        os.makedirs(prompt_masks_folder, exist_ok=True)
        os.makedirs(text_folder, exist_ok=True)

        frames_list = []
        pixel_values = None
        if "pixel_values" in data_dict and data_dict["pixel_values"] is not None:
            try:
                pixel_values = data_dict["pixel_values"]
            except Exception as e:
                print(f"获取 pixel_values 时出错: {e}")
        if pixel_values is not None and pixel_values.numel() > 0:
            for i, frame_tensor in enumerate(pixel_values):
                try:
                    frame_np = frame_tensor.cpu().clone().numpy()
                    for c in range(3):
                        frame_np[c] = frame_np[c] * self.IMAGENET_STD[c] + self.IMAGENET_MEAN[c]
                    frame_np = np.clip(frame_np, 0, 1)
                    frame_np = (frame_np * 255).astype(np.uint8)
                    frame_np = frame_np.transpose(1, 2, 0)
                    out_path = os.path.join(frames_folder, f"{i}.jpg")
                    cv2.imwrite(out_path, cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
                    frames_list.append(frame_np)
                except Exception as e:
                    print(f"处理 pixel_values 第 {i} 帧时出错: {e}")
        else:
            print("未能获取 pixel_values，跳过视频帧保存。")

        g_frames_list = []
        if "g_pixel_values" in data_dict and data_dict["g_pixel_values"] is not None:
            for i, g_tensor in enumerate(data_dict["g_pixel_values"]):
                try:
                    g_np = g_tensor.cpu().clone().numpy()
                    g_np = g_np.transpose(1, 2, 0)  # HWC
                    g_frames_list.append(g_np)
                except Exception as e:
                    print(f"处理 g_pixel_values 第 {i} 帧时出错: {e}")
        else:
            print("未能获取 g_pixel_values，gt_masks overlay 将无法进行。")

        # 定义浅黄色色值
        light_yellow = np.array([255, 255, 204], dtype=np.uint8)

        gt_masks_list = data_dict.get("gt_masks", None)
        if gt_masks_list is not None and isinstance(gt_masks_list, list):
            for idx, mask_tensor in enumerate(gt_masks_list):
                subfolder = os.path.join(gt_masks_folder, f"mask_{idx}_overlay")
                os.makedirs(subfolder, exist_ok=True)
                n_frames = mask_tensor.shape[0]
                for j in range(n_frames):
                    try:
                        mask_np = mask_tensor[j].cpu().numpy()
                        if mask_np.max() <= 1:
                            mask_np = (mask_np * 255).astype(np.uint8)
                        else:
                            mask_np = mask_np.astype(np.uint8)
                        colored_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
                        colored_mask[mask_np > 128] = light_yellow
                        if j < len(g_frames_list):
                            orig_frame = g_frames_list[j]
                            H, W = orig_frame.shape[:2]
                            resized_mask = cv2.resize(colored_mask, (W, H))
                            overlay = cv2.addWeighted(orig_frame, 0.5, resized_mask, 0.5, 0)
                        else:
                            overlay = colored_mask
                        out_mask_path = os.path.join(subfolder, f"{j}.png")
                        cv2.imwrite(out_mask_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                    except Exception as e:
                        print(f"保存 gt_masks overlay 第 {idx} 个 mask, 第 {j} 帧时出错: {e}")
        else:
            print("data_dict 中没有 gt_masks，或格式不正确。")

        prompt_masks_list = data_dict.get("prompt_masks_112", None)
        if prompt_masks_list is not None and isinstance(prompt_masks_list, list):
            for idx, mask_tensor in enumerate(prompt_masks_list):
                subfolder = os.path.join(prompt_masks_folder, f"mask_{idx}_overlay")
                os.makedirs(subfolder, exist_ok=True)
                n_frames = mask_tensor.shape[0]
                for j in range(n_frames):
                    try:
                        mask_np = mask_tensor[j].cpu().numpy()
                        if mask_np.max() <= 1:
                            mask_np = (mask_np * 255).astype(np.uint8)
                        else:
                            mask_np = mask_np.astype(np.uint8)
                        colored_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
                        colored_mask[mask_np > 128] = light_yellow
                        if j < len(frames_list):
                            orig_frame = frames_list[j]
                            H, W = orig_frame.shape[:2]
                            resized_mask = cv2.resize(colored_mask, (W, H))
                            overlay = cv2.addWeighted(orig_frame, 0.5, resized_mask, 0.5, 0)
                        else:
                            overlay = colored_mask
                        out_mask_path = os.path.join(subfolder, f"{j}.png")
                        cv2.imwrite(out_mask_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                    except Exception as e:
                        print(f"保存 prompt_masks_112 overlay 第 {idx} 个 mask, 第 {j} 帧时出错: {e}")
        else:
            print("data_dict 中没有 prompt_masks_112，或格式不正确。")

        convs = data_dict.get("vanilla_conversation", [])
        text_path = os.path.join(text_folder, "qa.txt")
        try:
            with open(text_path, "w", encoding="utf-8") as f:
                for turn in convs:
                    f.write(f"{turn.get('from', '')}: {turn.get('value', '')}\n")
                    f.write("-" * 50 + "\n")
        except Exception as e:
            print("保存问答文本时出错:", e)

        print(f"全部可视化结果已保存至: {video_folder}")

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
        ret_masks = ret_masks.flatten(0, 1)
        return ret_masks

    def dataset_map_fn(self, data_dict):
        images = []
        file_name = data_dict['file_name']
        if 'sav_train' in file_name:
            video_path = os.path.join(self.image_folder, file_name)
            video_frames = get_video_frames(video_path)
            video_frames = video_frames[::4]
            len_frames = len(video_frames)
            height = video_frames[0].data.shape[0]
            width = video_frames[0].data.shape[1]
        else:
            len_frames = len(data_dict['frames'])
            height = data_dict['height']
            width = data_dict['width']
            video_frames = data_dict['frames']
        conversations = data_dict['conversations']
        masks_mapping = data_dict["annotation"]
        _ret = {}
        mask_info = dict()
        _ret['image'] = file_name

        sparse_indices = get_sparse_indices_uniform(len_frames, max_frames=self.max_frames)
        self.sparse_len = len(sparse_indices)
        dense_indices = get_dense_indices(self.sparse_len, self.dense_len)

        selected_sparse = [sparse_indices[i] for i in dense_indices]
        fast_video_frames = None
        sampled_fast_frame_idxs = None

        if data_dict['frames'] is not None:
            for selected_frame_index in sparse_indices:
                frame_id = data_dict['frames'][selected_frame_index]
                images.append(os.path.join(data_dict['file_name'], frame_id + '.jpg'))
        elif video_frames is not None:
            for selected_frame_index in sparse_indices:
                images.append(video_frames[selected_frame_index])
        else:
            print(f"[INFO] Wrong at processing: {file_name}")
            new_index = random.randint(0, self.real_len() - 1)
            return self.__getitem__(new_index)

        if height is None or width is None:
            frame_path = os.path.join(self.image_folder, images[0])
            frame_image = Image.open(frame_path).convert('RGB')
            width, height = frame_image.size

        _ret['height'] = height
        _ret['width'] = width
        _ret['images'] = images
        _ret['dense_indices'] = dense_indices
        _ret['prompt_masks'] = []
        _ret['prompt_masks_112'] = []
        if 'mevis' in file_name:
            mask_dict = self.mask_dicts.get('mevis', {})
        elif 'lvvis' in file_name:
            mask_dict = self.mask_dicts.get('lvvis', {})
        elif 'ref_youtube_vos' in file_name:
            mask_dict = self.mask_dicts.get('ref_youtube', {})
        elif 'VidSTG' in file_name:
            mask_dict = self.mask_dicts.get('vidstg', {})
        elif 'sav' in file_name:
            mask_dict = self.mask_dicts.get('sav', {})
        for item in masks_mapping:
            if item == -1:
                continue
            obj_list = []
            obj_list_all = []
            frames_masks_ = []
            frames_masks_all = []
            mask_annotation = mask_dict[str(item)]
            # for frame_idx in sparse_indices:
            #     frames_masks_all.append(copy.deepcopy(mask_annotation[frame_idx]))
            # obj_list_all.append(frames_masks_all)

            for frame_idx in selected_sparse:
                frames_masks_.append(copy.deepcopy(mask_annotation[frame_idx]))
            obj_list.append(frames_masks_)

            obj_masks = self.decode_mask([obj_list], image_size=(height, width))
            # obj_masks_all = self.decode_mask([obj_list_all], image_size=(height, width))
            if str(item) not in mask_info:
                temp_item = str(item)
                mask_info[temp_item] = dict()
                processed_mask, processed_mask_112 = self._get_region_infos(obj_masks)
                mask_info[temp_item]['mask'] = obj_masks
                mask_info[temp_item]['processed_mask'] = processed_mask
                mask_info[temp_item]['prompt_masks_112'] = processed_mask_112
                mask_info[temp_item]['height'] = height
                mask_info[temp_item]['width'] = width
                _ret['prompt_masks'].append(processed_mask)
                _ret['prompt_masks_112'].append(processed_mask_112)
            else:
                print("mask exist...")
                exit(0)
        prompt_masks = []
        prompt_masks_112 = []
        mask_count = []
        mask_token_list = []
        obj_list = data_dict['obj_masks']
        gt_list = data_dict['groundtruth']
        for item in obj_list:
            mask_count.append(len(item))
            for index, sub_item in enumerate(item):
                prompt_masks.append(mask_info[str(sub_item)]['processed_mask'])
                prompt_masks_112.append(mask_info[str(sub_item)]['prompt_masks_112'])
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
        fast_video_masks = None
        qa_list, text_prompts = self.prepare_text(n_frames=self.dense_len, conversations=conversations, mask_token_list=mask_token_list,
                                      num_image_tokens=self.patch_token)
        ret = {'images': images, 'prompt_masks': prompt_masks, 'prompt_masks_112': prompt_masks_112, 'conversation': qa_list, 'prompts':text_prompts,
               'gt_masks': gt_final,'mask_count':mask_count,'fast_images': fast_video_frames, 'file_name': file_name,
               'vanilla_conversation': data_dict['conversations'],
               'fast_video_masks': fast_video_masks, 'dense_indices': dense_indices}
        return ret

    def _get_region_infos(self, masks):
        n_obj, h, w = masks.shape
        box_masks = []

        for i_obj in range(n_obj):
            cur_mask = masks[i_obj]  # shape: (H, W)
            nonzero_rows, nonzero_cols = torch.where(cur_mask > 0)
            if len(nonzero_rows) == 0:
                box_masks.append(cur_mask)  # 或者 torch.zeros_like(cur_mask)
            else:
                min_row, max_row = nonzero_rows.min(), nonzero_rows.max()
                min_col, max_col = nonzero_cols.min(), nonzero_cols.max()
                new_mask = torch.zeros_like(cur_mask)
                new_mask[min_row:max_row + 1, min_col:max_col + 1] = 1
                box_masks.append(new_mask)

        box_masks = torch.stack(box_masks, dim=0)

        masks_112_112 = F.interpolate(
            box_masks.unsqueeze(0),
            size=(112, 112),
            mode='nearest'
        ).squeeze(0)  # -> (n_obj, 64, 64)

        target_h = int(self.image_size // self.patch_size * self.downsample_ratio)
        target_w = int(self.image_size // self.patch_size * self.downsample_ratio)
        masks_small = F.interpolate(
            box_masks.unsqueeze(0),
            size=(target_h, target_w),
            mode='nearest'
        ).squeeze(0)  # -> (n_obj, target_h, target_w)

        return masks_small, masks_112_112

    def fill_masks(self, question, replacement_tokens):
        token_iterator = iter(replacement_tokens)

        def replace_func(match):
            return next(token_iterator, match.group(0))

        filled_question = re.sub(r'<mask>', replace_func, question)
        return filled_question

    def prepare_text(self, n_frames, conversations, mask_token_list=None, num_image_tokens=256, n_fast_images=50):
        text_prompts = []
        num_context_token = self.sparse_len
        context_str = self.IMG_CONTEXT_TOKEN * num_context_token

        frame_token_str = f'{self.IMG_START_TOKEN}' \
                          f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                          f'{self.IMG_END_TOKEN}'

        if self.use_fast and not self.fast_token_after_question:
            fast_frame_token_str = f'{self.FAST_IMG_START_TOKEN}' \
                          f'{self.FAST_IMG_CONTEXT_TOKEN * n_fast_images * self.fast_pool_size * self.fast_pool_size}' \
                          f'{self.FAST_IMG_END_TOKEN}' + '\n'
        else:
            fast_frame_token_str = ''
        if self.fast_token_after_question:
            assert self.use_fast
            after_question_str = f'{self.FAST_IMG_START_TOKEN}' \
                          f'{self.FAST_IMG_CONTEXT_TOKEN * n_fast_images * self.fast_pool_size * self.fast_pool_size}' \
                          f'{self.FAST_IMG_END_TOKEN}'
        else:
            after_question_str = ''
        qa_list = []
        input = ''
        mask_index = 0
        question = None
        # conv_manager = ConversationManager(max_length=self.max_conv_len)  as
        for i, item in enumerate(conversations):
            if item['from'] == 'human':
                item['value'] = item['value'].replace('<image>\n', '').replace('<video>\n', '').replace('\n<video>', '').replace('\n<image>', '')
                question = item['value'].strip()
                if i == 0:
                    frame_tokens = frame_token_str + '\n'
                    frame_tokens = frame_tokens * n_frames
                    frame_tokens = frame_tokens.strip()
                    frame_tokens = fast_frame_token_str + frame_tokens
                    input = input + frame_tokens + context_str + item['value']
                    text_prompts.append(question)
                else:
                    input = input + context_str + item['value']
                    text_prompts.append(question)
                mask_count = input.count("<mask>")
                fill_token = []
                for i_idx in range(mask_index, mask_index + mask_count):
                    fill_token.append(self.VP_START_TOKEN + self.IMG_CONTEXT_TOKEN * 2 + self.VP_END_TOKEN)
                    # fill_token.append(self.IMG_CONTEXT_TOKEN * (self.sparse_len * 2))
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

    def __getitem__(self, index):
        index = index % self.real_len()
        data_dict = copy.deepcopy(self.text_data[index])
        data_dict = self.dataset_map_fn(data_dict)

        assert 'images' in data_dict.keys()
        pixel_values = []
        extra_pixel_values = []
        num_video_tokens = None
        num_frame_tokens = None
        dense_indices = data_dict['dense_indices']
        frames_files = data_dict['images']
        if isinstance(frames_files[0], str):
            frames_files = [os.path.join(self.image_folder, frame_file) for frame_file in frames_files]
            for index, frame_path in enumerate(frames_files):
                frame_image = Image.open(frame_path).convert('RGB')
                ori_width, ori_height = frame_image.size
                if self.extra_image_processor is not None and index in dense_indices:
                    g_image = np.array(frame_image)  # for grounding
                    g_image = self.extra_image_processor.apply_image(g_image)
                    g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
                    extra_pixel_values.append(g_pixel_values)

                if self.preprocessor is not None:
                    pass
                else:
                    frame_image = self.transformer(frame_image)
                pixel_values.append(frame_image)
        else:
            for index, frame in enumerate(frames_files):
                frame = frame[:, :, ::-1]
                frame_image = Image.fromarray(frame).convert('RGB')
                ori_width, ori_height = frame_image.size
                if self.extra_image_processor is not None and index in dense_indices:
                    g_image = np.array(frame_image)  # for grounding
                    g_image = self.extra_image_processor.apply_image(g_image)
                    g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
                    extra_pixel_values.append(g_pixel_values)

                frame_image = self.transformer(frame_image)
                pixel_values.append(frame_image)

        if self.preprocessor is not None:
            if self.arch_type == 'qwen':
                _data_dict = self.preprocessor(pixel_values, do_resize=True, size=(self.image_size, self.image_size))
                _data_dict['pixel_values'] = torch.tensor(_data_dict['pixel_values'], dtype=torch.float)
                _data_dict['image_grid_thw'] = torch.tensor(_data_dict['image_grid_thw'], dtype=torch.int)
                num_frame_tokens = int(_data_dict['image_grid_thw'][0].prod() * (self.downsample_ratio ** 2))
                num_frames = _data_dict['image_grid_thw'].shape[0]
                num_video_tokens = num_frame_tokens * num_frames
            elif self.arch_type == 'llava':
                _data_dict = self.preprocessor(pixel_values, do_resize=True, size=(self.image_size, self.image_size))
                _data_dict['pixel_values'] = np.stack(_data_dict['pixel_values'], axis=0)
                _data_dict['pixel_values'] = torch.tensor(_data_dict['pixel_values'], dtype=torch.float)
            else:
                raise NotImplementedError
            data_dict.update(_data_dict)
        else:
            pixel_values = torch.stack(pixel_values, dim=0) # (n_f, 3, h, w)
            data_dict['pixel_values'] = pixel_values
        if self.extra_image_processor is not None:
            data_dict['g_pixel_values'] = extra
        gt_masks = data_dict.get('gt_masks', [])
        if not gt_masks or any(mask is None for mask in gt_masks):
            print(f"Empty or invalid gt_masks for index {index}. Fetching a new sample.")
            new_index = random.randint(0, self.real_len() - 1)
            return self.__getitem__(new_index)

        try:
            masks = torch.stack(gt_masks, dim=0).flatten(0, 1)
        except TypeError as e:
            print(f"Error stacking gt_masks for index {index}: {e}. Fetching a new sample.")
            # 随机获取一个新的索引
            new_index = random.randint(0, self.real_len() - 1)
            return self.__getitem__(new_index)

        data_dict['masks'] = masks

        if num_video_tokens is not None:
            assert self.patch_token == 1
            input_str = data_dict['conversation'][0]['input']
            input_str = input_str.replace(self.IMG_CONTEXT_TOKEN, self.IMG_CONTEXT_TOKEN * num_frame_tokens)
            assert input_str.count(self.IMG_CONTEXT_TOKEN) == num_video_tokens
            data_dict['conversation'][0]['input'] = input_str

        result = self.template_map_fn(data_dict)
        data_dict.update(result)
        result = video_lisa_encode_fn(data_dict, tokenizer=self.tokenizer, max_length=self.max_length)
        data_dict.update(result)

        # for fast branch
        if self.use_fast:
            fast_pixel_values = []
            frames_files = data_dict['fast_images']
            frames_files = [os.path.join(self.image_folder, frame_file) for frame_file in frames_files]
            for frame_path in frames_files:
                frame_image = Image.open(frame_path).convert('RGB')
                ori_width, ori_height = frame_image.size

                frame_image = self.transformer(frame_image)
                fast_pixel_values.append(frame_image)

            fast_pixel_values = torch.stack(fast_pixel_values, dim=0)  # (n_f, 3, h, w)
            data_dict['fast_pixel_values'] = fast_pixel_values

            # process and get masks
            masks = self.decode_mask(data_dict['fast_video_masks'], image_size=(ori_height, ori_width))

            if masks is None:
                print("[INFO] Mask is none...")
                return self.__getitem__(random.randint(0, self.real_len()))

            data_dict['fast_exists'] = masks.to(dtype=torch.int).sum(dim=(-2, -1)).ge(self.exist_thr).unsqueeze(-1)

            del data_dict['fast_video_masks']
        # self.save_visualization_results(data_dict)
        return data_dict

    def visualization_debug(self, data_dict):
        save_folder = os.path.join(self.save_folder, 'sample_{}'.format(self.cur_number))
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        self.cur_number += 1

        # images

        show_images = []

        pixel_values = data_dict['pixel_values']
        save_folder_image = os.path.join(save_folder, 'image')
        if not os.path.exists(save_folder_image):
            os.mkdir(save_folder_image)
        for i_image, image_pixel_value in enumerate(pixel_values):
            # print(image_pixel_value.shape)
            image_pixel_value[0] = image_pixel_value[0] * 0.2686
            image_pixel_value[1] = image_pixel_value[1] * 0.2613
            image_pixel_value[2] = image_pixel_value[2] * 0.2757
            image_pixel_value[0] = image_pixel_value[0] + 0.4814
            image_pixel_value[1] = image_pixel_value[1] + 0.4578
            image_pixel_value[2] = image_pixel_value[2] + 0.4082
            image_pixel_value = image_pixel_value * 255
            image_pixel_value = image_pixel_value.permute(1, 2, 0)
            image_pixel_value = image_pixel_value.to(torch.uint8).numpy()
            # print(os.path.join(save_folder_image, '{}.jpg'.format(i_image)))
            # print(image_pixel_value.shape)
            show_images.append(image_pixel_value)
            cv2.imwrite(os.path.join(save_folder_image, '{}.jpg'.format(i_image)), image_pixel_value)

        # text
        input_text = self.tokenizer.decode(data_dict['input_ids'], skip_special_tokens=False)
        with open(os.path.join(save_folder, 'text.json'), 'w') as f:
            json.dump([input_text], f)

        # masks
        save_folder_mask = os.path.join(save_folder, 'mask')
        if not os.path.exists(save_folder_mask):
            os.mkdir(save_folder_mask)
        n_frames = len(pixel_values)
        masks = data_dict['masks']
        _, h, w = masks.shape
        masks = masks.reshape(-1, n_frames, h, w)
        for i_obj, obj_masks in enumerate(masks):
            save_folder_mask_obj_folder = os.path.join(save_folder_mask, 'obj_{}'.format(i_obj))
            if not os.path.exists(save_folder_mask_obj_folder):
                os.mkdir(save_folder_mask_obj_folder)
            for i_frame, f_mask in enumerate(obj_masks):
                f_mask = f_mask.numpy()
                f_mask = f_mask * 255
                f_mask = np.stack([f_mask * 1, f_mask * 0, f_mask * 0], axis=2)
                f_mask = show_images[i_frame] * 0.3 + 0.7 * f_mask
                f_mask = f_mask.astype(np.uint8)
                cv2.imwrite(os.path.join(save_folder_mask_obj_folder, '{}.png'.format(i_frame)), f_mask)
        return
