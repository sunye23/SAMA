import json
import os

from tqdm import tqdm
import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from PIL import Image
from torch.utils.data import Dataset
from pycocotools import mask as maskUtils
import numpy as np
import copy
import re
import cv2
import pickle
from xtuner.registry import BUILDER
from xtuner.dataset.huggingface import process_hf_dataset, build_origin_dataset
import torchvision.transforms as T
from xtuner.utils import DEFAULT_IMAGE_TOKEN
from torchvision.transforms.functional import InterpolationMode
from .encode_fn import video_lisa_encode_fn
from .utils import dynamic_preprocess
from projects.llava_sam2.datasets.utils import get_sparse_indices, get_dense_indices, get_sparse_indices_uniform

import random
import math
import torch.nn.functional as F


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


class VideoRegionDescriptionDataset(Dataset):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    LIMIT = ''

    VP_START_TOKEN = '<vp>'
    VP_END_TOKEN = '</vp>'

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    def __init__(self,
                 image_folder,
                 data_path=None,
                 mevis_json=None,
                 lvvis_json=None,
                 ref_youtube_pkl=None,
                 sav_json=None,
                 vidstg_json=None,
                 tokenizer=None,
                 max_length=8196,
                 special_tokens=None,
                 template_map_fn=None,
                 extra_image_processor=None,
                 lazy=True,
                 frame_contiguous_sample=False,
                 repeats=1,
                 sampled_frames=5,
                 dense_len=None,
                 sparse_len=None,
                 num_temporal_token=None,
                 num_spatial_token=None,
                 window_stride=None,
                 window_size=None,
                 description_per_item=3,
                 single_image_mode=False,
                 # 仅在 use_fast 为 True 时有效
                 use_fast=False,
                 n_fast_images=50,
                 fast_pool_size=4,
                 fast_token_after_question=False,
                 max_frames=32,
                 selected_datasets=None,
    ):
        super().__init__()
        assert lazy
        self.lazy = lazy
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
        self.selected_datasets = selected_datasets
        self.max_length = max_length
        self.max_frames = max_frames
        self.description_per_item = description_per_item
        json_data = self.json_file_preprocess(data_path)
        self.text_data = json_data

        self.dense_len = dense_len
        self.sparse_len = sparse_len
        self.num_temporal_token = num_temporal_token
        self.num_spatial_token = num_spatial_token
        self.window_stride = window_stride
        self.window_size = window_size

        self.sampled_frames = sampled_frames
        self.image_folder = image_folder

        self.tokenizer = BUILDER.build(tokenizer)

        self.template_map_fn = template_map_fn
        if isinstance(self.template_map_fn, dict) and self.lazy:
            _type = self.template_map_fn['type']
            del self.template_map_fn['type']
            self.template_map_fn = _type(**self.template_map_fn)

        if extra_image_processor is not None:
            self.extra_image_processor = BUILDER.build(extra_image_processor)

        self.repeats = repeats

        self._system = ''

        self.use_fast = use_fast
        self.n_fast_images = n_fast_images
        self.fast_pool_size = fast_pool_size
        self.frame_contiguous_sample = frame_contiguous_sample
        self.min_dynamic_patch = 1
        self.max_dynamic_patch = 12
        self.downsample_ratio = 0.5
        self.image_size = 448
        self.use_thumbnail = True
        patch_size = 14
        self.patch_size = patch_size
        self.patch_token = int((self.image_size // patch_size) ** 2 * (self.downsample_ratio ** 2))

        self.transformer = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])

        if special_tokens is not None:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        self.single_image_mode = single_image_mode

    def json_file_preprocess(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as fp:
            data = json.load(fp)

        dataset_dicts = []
        description_per_data = self.description_per_item
        for video in tqdm(data, desc='Registering VideoRegionCaptioning videos'):
            video_info = {
                'video': video.get('video', None),
                'video_id': video.get('video_id', None),
                'height': video.get('height', None),
                'width': video.get('width', None),
                'frames': video.get('frames', None),
                'annotation': video.get('anno_map', None),
            }
            if self.selected_datasets is not None:
                flag = False
                for item in self.selected_datasets:
                    if item in video_info['video']:
                        flag = True
                if flag is True:
                    continue

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
                if index + 1 == len(descriptions) or (index + 1) % description_per_data == 0 or len(descriptions) == 1:
                    record = {
                        'file_name': video_info['video'],
                        'id': video_info['video_id'],
                        'height': video_info['height'],
                        'width': video_info['width'],
                        'frames': video_info['frames'],
                        'annotation': video_info['annotation'],
                        'description': new_dict,
                    }
                    dataset_dicts.append(record)
                    new_dict = dict()
        return dataset_dicts

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

    def __len__(self):
        return len(self.text_data) * self.repeats

    def real_len(self):
        return len(self.text_data)

    def annToMask(self, mask_ann, h, w):
        if isinstance(mask_ann, list):
            rles = maskUtils.frPyObjects(mask_ann, h, w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, h, w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

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

    def _process_conversation(self, descriptions, mask_info):
        conversation = []
        question_list = []
        frame_token_str = f'{self.IMG_START_TOKEN}' \
                          f'{self.IMG_CONTEXT_TOKEN * self.patch_token}' \
                          f'{self.IMG_END_TOKEN}'

        num_context_token = self.sparse_len
        context_str = self.IMG_CONTEXT_TOKEN * num_context_token + '\n'
        for index, (key, value) in enumerate(descriptions.items()):
            question = random.choice(DETAILED_QUESTIONS)
            question_list.append(question.strip())
            match = re.search(r"obj(\d+)", key)
            obj_num = int(match.group(1))
            if index == 0:
                frame_tokens = frame_token_str + '\n'
                frame_tokens = frame_tokens * self.dense_len
                frame_tokens = frame_tokens.strip()
                question =  frame_tokens + context_str + question
            else:
                question = context_str + question
            region_str = self.VP_START_TOKEN + self.IMG_CONTEXT_TOKEN  * 2 + self.VP_END_TOKEN
            question = question.replace("<mask>", region_str) + self.LIMIT
            question = question.strip()
            answer = value.strip()
            conversation.append({'input': question, 'output': answer})
        return conversation, question_list
    #

    def _get_region_infos(self, masks):
        n_obj, h, w = masks.shape
        box_masks = []

        for i_obj in range(n_obj):
            cur_mask = masks[i_obj]  # shape: (H, W)
            nonzero_rows, nonzero_cols = torch.where(cur_mask > 0)
            if len(nonzero_rows) == 0:
                box_masks.append(cur_mask)
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

    def dataset_map_fn(self, data_dict, select_k=5):
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
        images = []
        descriptions = data_dict['description']
        masks_mapping = data_dict["annotation"]
        video_length = len(video_frames)
        _ret = {}
        mask_info = dict()
        sparse_indices = get_sparse_indices_uniform(len_frames, max_frames=self.max_frames)
        self.sparse_len = len(sparse_indices)
        dense_indices = get_dense_indices(self.sparse_len, self.dense_len)

        selected_sparse = [sparse_indices[idx] for idx in dense_indices]
        if self.use_fast:
            fast_interval = len_frames / (self.n_fast_images + 1e-4)
            sampled_fast_frame_idxs = [
                min(int(i * fast_interval), len_frames - 1)
                for i in range(self.n_fast_images)
            ]
            fast_video_frames = []
            for selected_frame_index in sampled_fast_frame_idxs:
                frame_id = data_dict['frames'][selected_frame_index]
                fast_video_frames.append(os.path.join(data_dict['video'], frame_id + '.jpg'))
        else:
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

        for obj_key, value in descriptions.items():
            # obj_list = []
            obj_list_all = []
            match = re.search(r"obj(\d+)", obj_key)
            obj_num = int(match.group(1))
            mask_annotation = mask_dict[str(masks_mapping[obj_num])]
            frames_masks_ = []
            frames_masks_all = []
            for frame_idx in selected_sparse:
                frames_masks_all.append(copy.deepcopy(mask_annotation[frame_idx]))
            obj_list_all.append(frames_masks_all)

            obj_masks_all = self.decode_mask([obj_list_all], image_size=(height, width))

            if obj_masks_all is None:
                zero_mask = np.zeros(
                    (len(sparse_indices), height, width), dtype=np.uint8)
                obj_masks_all = torch.from_numpy(zero_mask).float()
            if obj_num not in mask_info:
                temp_obj_num = str(obj_num)
                mask_info[temp_obj_num] = dict()
                processed_mask, processed_mask_112 = self._get_region_infos(obj_masks_all)
                mask_info[temp_obj_num]['mask'] = obj_masks_all
                mask_info[temp_obj_num]['processed_mask'] = processed_mask
                mask_info[temp_obj_num]['processed_mask_112'] = processed_mask_112
                mask_info[temp_obj_num]['height'] = height
                mask_info[temp_obj_num]['width'] = width
                _ret['prompt_masks'].append(processed_mask)
                _ret['prompt_masks_112'].append(processed_mask_112)
            else:
                print("mask exist...")
                return None

        if mask_info is None:
            return None

        # 这里用 self._process_conversation 来构造对话
        conversations, question_list = self._process_conversation(descriptions, mask_info)
        _ret['conversation'] = conversations
        _ret['mask_count'] = [1 for i in range(len(question_list))]
        _ret['prompts'] = question_list
        _ret['dense_indices'] = dense_indices

        return _ret

    def replace_image_str(self, data_dict, image_str):
        data_dict['conversation'][0]['input'] = \
            data_dict['conversation'][0]['input'].replace(DEFAULT_IMAGE_TOKEN, image_str)
        return data_dict

    def __getitem__(self, index):
        index = index % self.real_len()
        data_dict = copy.deepcopy(self.text_data[index])
        # parse datasets

        result = self.dataset_map_fn(data_dict, select_k=self.dense_len) # {'image', 'height', 'width', 'conversation', 'masks'}
        if result is None or result['prompt_masks'] is None:
            return self.__getitem__(random.randint(0, self.real_len()))

        data_dict = result

        pixel_values = []

        frames_files = data_dict['images']
        if isinstance(frames_files[0], str):
            frames_files = [os.path.join(self.image_folder, frame_file) for frame_file in frames_files]
            for frame_path in frames_files:
                frame_image = Image.open(frame_path).convert('RGB')
                ori_width, ori_height = frame_image.size

                frame_image = self.transformer(frame_image)
                pixel_values.append(frame_image)
        else:
            for index, frame in enumerate(frames_files):
                frame = frame[:, :, ::-1]
                frame_image = Image.fromarray(frame).convert('RGB')
                ori_width, ori_height = frame_image.size

                frame_image = self.transformer(frame_image)
                pixel_values.append(frame_image)

        pixel_values = torch.stack(pixel_values, dim=0) # (n_f, 3, h, w)

        data_dict['pixel_values'] = pixel_values

        result = self.template_map_fn(data_dict)
        data_dict.update(result)
        result = video_lisa_encode_fn(data_dict, tokenizer=self.tokenizer, max_length=self.max_length,
                                      with_image_token=True)
        data_dict.update(result)
        data_dict['prompt_masks'] = data_dict['prompt_masks']

        if data_dict['prompt_masks'] is None:
            return self.__getitem__(random.randint(0, self.real_len()))
        return data_dict


DETAILED_QUESTIONS =  [
    'Can you provide me with a detailed description of the region <mask> in the video?',
    "I'm curious about the region <mask> in the video. Could you describe it in detail?",
    'What can you tell me about the region <mask> in the video?',
    "I'd like to know more about the area <mask> in the video. Can you give me a detailed description?",
    'Could you describe the region <mask> in the video in great detail?',
    'What details can you give me about the region <mask> in the video?',
    'Please provide me with a comprehensive description of the region <mask> in the video.',
    'Can you give me a detailed account of the region <mask> in the video?',
    "I'm interested in learning more about the region <mask> in the video. Can you describe it in detail?",
    'What is the region <mask> in the video like? Could you give me a detailed description?',
    'Please describe the region <mask> in the video in detail.',
    'Can you offer a thorough analysis of the region <mask> in the video?',
    'Could you elaborate on the region <mask> in the video provided?',
    'Please share more information about the zone <mask> in the video.',
    'What insights can you give ablout the area <mask> in the video presented?',
    'Can you share a comprehensive rundown of the region <mask> in the presented video?',
    "I'd like to know more about the region <mask> in the video provided.",
    'Work through the important details of the area <mask> in the video.',
    'Illustrate the area <mask> through a descriptive explanation.',
    'Examine the region <mask> closely and share its details.'
]