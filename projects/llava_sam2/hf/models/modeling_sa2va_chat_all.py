# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import random
import warnings
from typing import Any, List, Optional, Tuple, Union

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

import torch.utils.checkpoint
import transformers

from .modeling_internlm2 import InternLM2ForCausalLM
from .modeling_phi3 import Phi3ForCausalLM
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, Qwen2ForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging
from transformers import StoppingCriteriaList, StoppingCriteria

from .configuration_sa2va_chat import Sa2VAChatConfig
from .modeling_intern_vit import InternVisionModel, has_flash_attn

from .sam2 import SAM2
from .templates import PROMPT_TEMPLATE

import numpy as np
from torchvision.transforms.functional import resize, to_pil_image

from types import MethodType
import torch.nn.functional as F
from projects.llava_sam2.models.qformer import BertConfig, BertLMHeadModel

from transformers import BertTokenizer
import math
from projects.llava_sam2.models.utils import MaskPooling, MLP
try:
    from .flash_attention import FlashAttention
    has_flash_attn = True
except:
    print('FlashAttention is not installed.')
    has_flash_attn = False

logger = logging.get_logger(__name__)

def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))

class StopWordStoppingCriteria(StoppingCriteria):
    """StopWord stopping criteria."""

    def __init__(self, tokenizer, stop_word):
        self.tokenizer = tokenizer
        self.stop_word = stop_word
        self.length = len(self.stop_word)

    def __call__(self, input_ids, *args, **kwargs) -> bool:
        cur_text = self.tokenizer.decode(input_ids[0])
        cur_text = cur_text.replace('\r', '').replace('\n', '')
        return cur_text[-self.length:] == self.stop_word

def get_stop_criteria(
    tokenizer,
    stop_words=[],
):
    stop_criteria = StoppingCriteriaList()
    for word in stop_words:
        stop_criteria.append(StopWordStoppingCriteria(tokenizer, word))
    return stop_criteria

class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        img = to_pil_image(image, mode='RGB')
        return np.array(img.resize((self.target_length, self.target_length)))

class Sa2VAChatModel(PreTrainedModel):
    config_class = Sa2VAChatConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'InternLM2DecoderLayer',
                         'Phi3DecoderLayer', 'Qwen2DecoderLayer', 'SAM2']
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True

    def __init__(self, config: Sa2VAChatConfig, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config)
        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.template = self.template.replace('-', '_')
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.llm_arch_name = config.llm_config.architectures[0]

        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config._attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Phi3ForCausalLM':
                self.language_model = Phi3ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        self.img_context_token_id = None
        self.conv_template = PROMPT_TEMPLATE[self.template]
        self.template = self.conv_template
        if hasattr(config, 'system_message'):
            self.system_message = config.system_message
        self.num_samples = 0

        if config.use_backbone_lora:
            self.wrap_backbone_lora(r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora)

        if config.use_llm_lora:
            self.wrap_llm_lora(r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)

        self.grounding_encoder = SAM2()
        out_dim = self.grounding_encoder.hidden_dim
        in_dim = llm_hidden_size
        self.text_hidden_fcs = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim), nn.Dropout(0.0)
        )

        self.init_prediction_config = False

        ###########################################
        #### Init Spatial and Temporal Qformer ####
        ###########################################
        self.mask_pooling = MaskPooling()
        self.Qformer_tokenizer = self.init_tokenizer()
        self.Qformer_temporal_tokenizer = self.init_temporal_tokenizer()

        self.spatial_Qformer_ln = torch.nn.LayerNorm(1408)

        self.spatial_Qformer, self.spatial_Qformer_query_tokens = self.init_spatial_Qformer(num_query_token=32, vision_width=1408)
        self.spatial_Qformer.resize_token_embeddings(len(self.Qformer_tokenizer))
        self.spatial_Qformer.cls = None

        self.temporal_Qformer, self.temporal_Qformer_query_tokens = self.init_temporal_Qformer(num_query_token=32, vision_width=self.spatial_Qformer.config.hidden_size, num_hidden_layers=2)
        self.temporal_Qformer.resize_token_embeddings(len(self.Qformer_temporal_tokenizer))
        self.temporal_Qformer.cls = None

        self.Qformer_mask_pooling_proj = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
        self.Qformer_mask_proj = MLP(112 * 112, 1024, config.hidden_size, 3)
        self.config_hidden_size = config.hidden_size
        self.Qformer_mask_pooling_proj_st = nn.Linear(in_features=self.config_hidden_size, out_features=self.temporal_Qformer.config.hidden_size)
        self.Qformer_mask_proj_st = MLP(112 * 112, 1024, self.temporal_Qformer.config.hidden_size, 3)

        self.spatial_Qformer.bert.embeddings.word_embeddings = None
        self.spatial_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.spatial_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.vlm2spatial_Qformer_proj = nn.Linear(config.hidden_size, 1408)

        self.Qformer_temp_attn_q = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.Qformer_temp_attn_k = torch.nn.Linear(self.spatial_Qformer.config.hidden_size, config.hidden_size)
        self.Qformer_temp_attn_v = torch.nn.Linear(self.spatial_Qformer.config.hidden_size, config.hidden_size)
        self.Qformer_temp_proj = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.Qformer_final_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.window_size = 512
        self.stride = 512

    @classmethod
    def init_temporal_Qformer(self, num_query_token, vision_width, num_hidden_layers=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = vision_width
        # 设置交叉注意力
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token

        Qformer = BertLMHeadModel(config=encoder_config)

        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    @classmethod
    def init_tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side='left')
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})        # "<p>", "</p>", "[SEG]", "<vp>", "</vp>"
        return tokenizer

    @classmethod
    def init_temporal_tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side='left')

        special_tokens_dict = {
            "bos_token": "[DEC]",
            "additional_special_tokens": [
                "<mask>",
                "<pos>",
                "[SEG]",
                "<p>",
                "</p>",
                "<vp>",
                "</vp>"
            ]
        }
        tokenizer.add_special_tokens(special_tokens_dict)
        return tokenizer

    @classmethod
    def init_spatial_Qformer(self, num_query_token=32, vision_width=1408, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def spatial_temporal_token_generation(self, image_features, prompts=None, image_counts=None, prompt_masks=None,
                                          prompt_masks_112=None, dense_indices=None, random_idx_list=None, mask_count=None):
        spatial_token_list = []
        temporal_token_list = []
        spatial_temporal_token_list = []
        assert len(prompts) == len(image_counts), f"Size mismatch! prompts: {len(prompts)}, image_counts: {len(image_counts)}"
        image_atts = torch.ones(image_features.size()[:-1], dtype=torch.long).to(image_features.device)
        total_count = 0
        # calculate each image feat according to the prompt
        for _idx in range(len(prompts)):
            assert isinstance(prompts[_idx], list), f"Prompt should be a list, but got {type(prompts[_idx])}"
            prompts[_idx] = [item.replace('<mask>', '<vp><mask><pos></vp>') for item in prompts[_idx]]
            input_token =  self.Qformer_temporal_tokenizer(
                prompts[_idx],
                padding='longest',
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(image_features.device)
            input_ids = input_token.input_ids
            attention_masks = input_token.attention_mask
            ori_input_ids = input_ids
            ori_attention_masks = attention_masks
            # shape: [prompt_num*frame_num, image_shape, feat_dim]
            img_feat_prompt = image_features[total_count:total_count + image_counts[_idx]]
            img_feat_prompt_expand = img_feat_prompt[None].expand(len(prompts[_idx]), -1, -1, -1).flatten(0, 1)
            raw_dtype = image_features.dtype
            if dense_indices is not None and dense_indices[_idx] is not None:
                dense_indices_item = dense_indices[_idx]
                dense_img_feat_prompt = img_feat_prompt[dense_indices_item, :, :]
            if prompt_masks is not None and prompt_masks_112 is not None and len(prompt_masks) > 0:
                current_obj_mask = prompt_masks[_idx]
                current_obj_mask_112 = prompt_masks_112[_idx]
                item_mask_count = mask_count[_idx]
            else:
                current_obj_mask = []
                current_obj_mask_112 = []
                item_mask_count = None
            visual_feat_list = []
            mask_feat_list = []
            dummy_visual_feat = None
            dummy_mask_feat = None
            if current_obj_mask is not None and item_mask_count is not None:
                mask_index = 0
                for tp_idx in range(len(item_mask_count)):
                    temp_mask_count = item_mask_count[tp_idx]

                    item_mask_feats = []
                    item_visual_feats = []
                    if temp_mask_count == 0:
                        mask_feat_list.append([])
                        visual_feat_list.append([])
                        continue
                    cur_mask = current_obj_mask[mask_index: mask_index + temp_mask_count]
                    cur_mask_112 = current_obj_mask_112[mask_index:mask_index + temp_mask_count]
                    for tj in range(temp_mask_count):
                        all_masks = cur_mask[tj].to(image_features.device).to(torch.bfloat16)  # [T, H, W]
                        all_masks_112 = cur_mask_112[tj].to(self.Qformer_mask_proj_st.layers[0].weight.device).to(
                            self.Qformer_mask_proj_st.layers[0].weight.dtype)  # [T, 112, 112]
                        non_empty_indices = (all_masks.view(all_masks.size(0), -1).sum(dim=1) > 0).nonzero(as_tuple=True)[0]

                        if len(non_empty_indices) > 0:
                            selected_idx = random.choice(non_empty_indices.tolist())
                        else:
                            selected_idx = 0
                        temp_image_feat = dense_img_feat_prompt[selected_idx].unsqueeze(0)

                        num_img, hw, C = temp_image_feat.shape
                        spatial_dim = int(hw ** 0.5)
                        temp_image_feat_reshaped = temp_image_feat.view(1, spatial_dim, spatial_dim, C).permute(0, 3, 1, 2)

                        temp_obj_mask = all_masks[selected_idx].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                        temp_obj_mask_112 = all_masks_112[selected_idx].unsqueeze(0).flatten(1, 2)

                        pooled_feature = self.mask_pooling(temp_image_feat_reshaped, temp_obj_mask)
                        pooled_feature = pooled_feature.to(self.Qformer_mask_pooling_proj_st.weight.dtype).to(self.Qformer_mask_pooling_proj_st.weight.device)
                        pooled_feature = self.Qformer_mask_pooling_proj_st(pooled_feature)
                        pooled_feature = pooled_feature.reshape(-1, pooled_feature.shape[-1])
                        mask_feature = self.Qformer_mask_proj_st(temp_obj_mask_112)
                        item_visual_feats.append(pooled_feature)
                        item_mask_feats.append(mask_feature)
                    visual_feat_list.append(torch.cat(item_visual_feats, dim=0))
                    mask_feat_list.append(torch.cat(item_mask_feats, dim=0))
                    mask_index = mask_index + temp_mask_count
            else:
                dummy_visual_feat = torch.zeros(
                    1, self.config_hidden_size,
                    device=self.Qformer_mask_pooling_proj_st.weight.device,
                    dtype=self.Qformer_mask_pooling_proj_st.weight.dtype
                )
                dummy_mask_feat = torch.zeros(
                    1,  112 * 112,
                    device=self.Qformer_mask_pooling_proj_st.weight.device,
                    dtype=self.Qformer_mask_pooling_proj_st.weight.dtype
                )
                dummy_visual_feat = self.Qformer_mask_pooling_proj_st(dummy_visual_feat)
                dummy_mask_feat = self.Qformer_mask_proj_st(dummy_mask_feat)
                visual_feat_list = None
                mask_feat_list = None

            img_att_prompt = image_atts[total_count:total_count + image_counts[_idx]]
            # img_att_prompt = img_att_prompt[None].expand(len(prompts[_idx]), -1, -1).flatten(0, 1)

            input_ids = input_ids[:, None].expand(-1, image_counts[_idx], -1).flatten(0, 1)
            attention_masks = attention_masks[:, None].expand(-1, image_counts[_idx], -1).flatten(0, 1)
            total_count += image_counts[_idx]

            bert_feat = self.vlm2spatial_Qformer_proj(img_feat_prompt)

            query_tokens = self.spatial_Qformer_query_tokens.expand(bert_feat.shape[0], -1, -1)
            # query_atts = torch.cat([torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(bert_feat.device),
            #                         attention_masks], dim=1)

            mm_img_in = self.spatial_Qformer_ln(bert_feat)
            spatial_mm_output = self.spatial_Qformer.bert(
                # input_ids,
                query_embeds=query_tokens,
                # attention_mask=query_atts,
                encoder_hidden_states=mm_img_in,
                encoder_attention_mask=img_att_prompt,
                return_dict=True,
            )
            spatial_mm_output = spatial_mm_output.last_hidden_state[:, :query_tokens.shape[1]]
            # text_q = self.spatial_Qformer_query_proj(spatial_mm_output)
            # final_token = self.token_generation(spatial_mm_output, img_feat_prompt)
            temporal_feat = spatial_mm_output.flatten(0, 1)
            temporal_feat = temporal_feat.unsqueeze(0).expand(ori_input_ids.shape[0], -1, -1)

            if self.window_size <= 0:
                temporal_att_masks = torch.ones(temporal_feat.size()[:-1], dtype=torch.long).to(temporal_feat.device)
                temporal_Qformer_query_tokens = self.temporal_Qformer_query_tokens.expand(temporal_feat.shape[0], -1, -1)
                temporal_query_atts = torch.cat(
                    [torch.ones(temporal_Qformer_query_tokens.size()[:-1], dtype=torch.long).to(temporal_feat.device), ori_attention_masks], dim=1)
                temporal_mm_output = self.temporal_Qformer.bert(
                    ori_input_ids,
                    query_embeds=temporal_Qformer_query_tokens,
                    visual_feats=visual_feat_list,
                    mask_feats=mask_feat_list,
                    dummy_visual_feat=dummy_visual_feat,
                    dummy_mask_feat=dummy_mask_feat,
                    attention_mask=temporal_query_atts,
                    encoder_hidden_states=temporal_feat,
                    encoder_attention_mask=temporal_att_masks,
                    return_dict=True,
                )
                temporal_mm_output = temporal_mm_output.last_hidden_state[:, :temporal_Qformer_query_tokens.shape[1]]
                temporal_token_list.append(temporal_mm_output)
            else:
                temporal_outputs = []
                window_size = self.window_size
                stride = self.stride
                time_length = temporal_feat.shape[1]
                for start_t in range(0, time_length, stride):
                    end_t = min(start_t + window_size, time_length)
                    temp_feat = temporal_feat[:, start_t:end_t, :]
                    temp_att_masks = torch.ones(temp_feat.size()[:-1], dtype=torch.long).to(temp_feat.device)
                    temporal_Qformer_query_tokens = self.temporal_Qformer_query_tokens.expand(temp_feat.shape[0], -1, -1)
                    temporal_query_atts = torch.cat([torch.ones(temporal_Qformer_query_tokens.size()[:-1], dtype=torch.long).to(temp_feat.device), ori_attention_masks], dim=1)

                    temporal_mm_output = self.temporal_Qformer.bert(
                        ori_input_ids,
                        query_embeds=temporal_Qformer_query_tokens,
                        visual_feats=visual_feat_list,
                        mask_feats=mask_feat_list,
                        dummy_visual_feat=dummy_visual_feat,
                        dummy_mask_feat=dummy_mask_feat,
                        attention_mask=temporal_query_atts,
                        encoder_hidden_states=temp_feat,
                        encoder_attention_mask=temp_att_masks,
                        return_dict=True,
                    )
                    temporal_outputs.append(temporal_mm_output.last_hidden_state[:, :temporal_Qformer_query_tokens.shape[1]])

                temporal_outputs = torch.cat(temporal_outputs, dim=1)
                temporal_inject = self.temporal_context_inject(img_feat_prompt_expand, temporal_outputs, image_counts[_idx])
                mm_output = torch.mean(temporal_inject, dim=1, keepdim=True)
                mm_output = self.Qformer_final_proj(mm_output)
                if image_counts is not None:
                    mm_output = mm_output.reshape(len(prompts[_idx]), image_counts[_idx],
                                                                          *mm_output.shape[-2:])
                    mm_output = mm_output.flatten(1, 2)
                spatial_temporal_token_list.append(mm_output)
        return spatial_temporal_token_list

    def temporal_context_inject(self, vis_embed, temp_embed, image_count=None):
        num_prompts, num_token, channels = temp_embed.shape
        temp_embed = temp_embed.unsqueeze(1).expand(-1, image_count, -1, -1).reshape(num_prompts * image_count, num_token, channels)
        query = self.Qformer_temp_attn_q(vis_embed)
        key = self.Qformer_temp_attn_k(temp_embed)
        value = self.Qformer_temp_attn_v(temp_embed)
        ctx_embed = query @ key.transpose(-1, -2)
        ctx_embed = ctx_embed / (key.shape[-1] ** 0.5)
        ctx_embed = ctx_embed.softmax(-1) @ value
        ctx_embed = self.Qformer_temp_proj(ctx_embed) + vis_embed
        return ctx_embed

    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        self.vision_model.print_trainable_parameters()

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        # Determine the target modules based on the architecture of the language model
        if self.llm_arch_name == 'InternLM2ForCausalLM':
            target_modules = ['attention.wqkv', 'attention.wo', 'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']
        elif self.llm_arch_name == 'Phi3ForCausalLM':
            target_modules = ['mlp.down_proj', 'mlp.gate_up_proj', 'self_attn.o_proj', 'self_attn.qkv_proj']
        elif self.llm_arch_name in ['Qwen2ForCausalLM', 'LlamaForCausalLM']:
            target_modules = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                              'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']
        else:
            raise NotImplemented
        lora_config = LoraConfig(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type='CAUSAL_LM'
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def forward(self, data, data_samples=None, mode='loss'):
        pixel_values = data['pixel_values']

        if type(pixel_values) is list or pixel_values.ndim == 5:
            if type(pixel_values) is list:
                pixel_values = [
                    x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values
                ]
            # b*n, c, h, w
            concat_images = torch.cat(
                [image.to(self.vision_model.dtype) for image in pixel_values], dim=0)
        else:
            raise NotImplementedError()

        input_ids = data['input_ids']
        position_ids = data['position_ids']
        attention_mask = data['attention_mask']
        # sum is 0 are text
        image_flags = torch.sum(concat_images, dim=(1, 2, 3)) != 0
        image_flags = image_flags.long()

        labels = data['labels']
        use_cache = False

        if 'vp_overall_mask' not in data.keys():
            vp_overall_mask = None
        else:
            vp_overall_mask = data['vp_overall_mask']

        if 'prompt_masks' in data.keys():
            prompt_masks = data['prompt_masks']
        else:
            prompt_masks = None

        outputs = self._llm_forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            pixel_values=concat_images,
            labels=labels,
            use_cache=use_cache,
            output_hidden_states=True,
            vp_overall_mask=vp_overall_mask,
            prompt_masks=prompt_masks,
        )

        return outputs

    def _llm_forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            vp_overall_mask=None,
            prompt_masks=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None \
            else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        # We only added the clone code here to avoid the error.
        input_embeds = self.language_model.get_input_embeddings()(
            input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds.to(input_embeds.dtype)  # FIXME: why vit_embeds is float16?
        fast_vit_embeds = None

        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        self._count += 1

        if vp_overall_mask is not None and prompt_masks is not None:
            vp_embeds = []
            vp_overall_mask = vp_overall_mask.to(vit_embeds.device).bool()
            prompt_masks = [item.to(vit_embeds.device).bool() for item in prompt_masks]

            vp_overall_mask = vp_overall_mask[image_flags == 1]
            overall_tile_vit_embeds = vit_embeds[vp_overall_mask]  # (n_img, hw, c)

            i_vp_img = 0
            for i_img in range(len(vit_embeds)):
                vp_embeds.append(vit_embeds[i_img].reshape(-1, C))
                if vp_overall_mask[i_img]:
                    tile_vit_embeds = overall_tile_vit_embeds[i_vp_img].reshape(-1, C)  # (hw, C)
                    objects_prompt_masks = prompt_masks[i_vp_img]
                    n_obj = len(objects_prompt_masks)
                    tile_vit_embeds = tile_vit_embeds.unsqueeze(0).repeat(n_obj, 1, 1)
                    objects_prompt_masks = objects_prompt_masks.reshape(n_obj, -1)
                    vp_embeds.append(tile_vit_embeds[objects_prompt_masks])
                    i_vp_img += 1
            vp_embeds = torch.cat(vp_embeds, dim=0)
        else:
            vp_embeds = None

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)

        if vp_embeds is None:
            try:
                input_embeds[selected] = vit_embeds.reshape(-1, C)
            except Exception as e:
                vit_embeds = vit_embeds.reshape(-1, C)
                print(f'warning: {e}, input_embeds[selected].shape='
                      f'{input_embeds[selected].shape}, '
                      f'vit_embeds.shape={vit_embeds.shape}')
                n_token = selected.sum()
                if n_token > len(vit_embeds):
                    print(f"Wrong !!! {n_token} image tokens in text but only {len(vit_embeds)} vit embeds !!!")
                    expand_ratio = n_token // len(vit_embeds) + 1
                    vit_embeds = torch.cat([vit_embeds] * expand_ratio, dim=0)

                input_embeds[selected] = vit_embeds[:n_token]
                raise
        else:
            try:
                input_embeds[selected] = vp_embeds.reshape(-1, C)
            except Exception as e:
                vp_embeds = vp_embeds.reshape(-1, C)
                print(f'warning: {e}, input_embeds[selected].shape='
                      f'{input_embeds[selected].shape}, '
                      f'vp_embeds.shape={vp_embeds.shape}')
                n_token = selected.sum()
                if n_token > len(vp_embeds):
                    print(f"Wrong !!! {n_token} image tokens in text but only {len(vp_embeds)} vit embeds !!!")
                    expand_ratio = n_token // len(vp_embeds) + 1
                    vp_embeds = torch.cat([vp_embeds] * expand_ratio, dim=0)

                input_embeds[selected] = vp_embeds[:n_token]
                raise
        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(
                -1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            prompt_masks=None,
            prompts=None,
            vp_overall_mask=None,
            sparse_indices=None,
            dense_indices=None,
            mask_count=None,
            video_prompt_masks=None,
            video_prompt_masks_64=None,
            **generate_kwargs,
    ) -> torch.LongTensor:
        device = self.device
        assert self.img_context_token_id is not None

        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                if type(pixel_values) is list or pixel_values.ndim == 5:
                    if type(pixel_values) is list:
                        pixel_values = [
                            x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values
                        ]
                    # b*n, c, h, w
                    pixel_values = torch.cat(
                        [image.to(self.vision_model.dtype) for image in pixel_values], dim=0)

                vit_embeds = self.extract_feature(pixel_values.to(device))
            if video_prompt_masks is not None and video_prompt_masks_64 is not None:
                st_video_prompt_masks = [item[sparse_indices, :, :] for item in video_prompt_masks]
                st_video_prompt_masks_64 = [item[sparse_indices, :, :] for item in video_prompt_masks_64]
                st_video_prompt_masks = [item[dense_indices, :, :] for item in st_video_prompt_masks]
                st_video_prompt_masks_64 = [item[dense_indices, :, :] for item in st_video_prompt_masks_64]
                spatial_temporal_token = self.spatial_temporal_token_generation(vit_embeds, [[prompts]], [vit_embeds.shape[0]],
                                                                                [st_video_prompt_masks], [st_video_prompt_masks_64],
                                                                                [dense_indices], None,  [mask_count])
            else:
                spatial_temporal_token = self.spatial_temporal_token_generation(vit_embeds, [[prompts]], [vit_embeds.shape[0]],
                                                                                None, None,
                                                                                None, None, None)
            st_dim = spatial_temporal_token[-1].shape[-1]
            original_vit_embeds = vit_embeds.clone()
            if dense_indices is not None:
                vit_embeds = vit_embeds[dense_indices, :, :]

            if dense_indices is not None:
                pixel_values = pixel_values[dense_indices, :, :, :]

            image_flags = torch.sum(pixel_values, dim=(1, 2, 3)) != 0
            image_flags = image_flags.long()
            vit_embeds = vit_embeds[image_flags == 1]

            input_embeds = self.language_model.get_input_embeddings()(input_ids.to(device))
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            if vp_overall_mask is not None and prompt_masks is not None:
                vp_embeds = []
                vp_overall_mask = vp_overall_mask.to(vit_embeds.device).bool()
                prompt_masks = [item.to(vit_embeds.device).bool() for item in prompt_masks]

                vp_overall_mask = vp_overall_mask[image_flags == 1]
                overall_tile_vit_embeds = vit_embeds[vp_overall_mask]  # (n_img, hw, c)

                i_vp_img = 0

                # vp_embeds.append(spatial_temporal_token[0].reshape(-1, st_dim))
                for i_img in range(len(vit_embeds)):
                    vp_embeds.append(vit_embeds[i_img].reshape(-1, C))
                    if vp_overall_mask[i_img]:
                        tile_vit_embeds = overall_tile_vit_embeds[i_vp_img].reshape(-1, C)  # (hw, C)
                        objects_prompt_masks = prompt_masks[i_vp_img]
                        n_obj = len(objects_prompt_masks)
                        tile_vit_embeds = tile_vit_embeds.unsqueeze(0).repeat(n_obj, 1, 1)
                        objects_prompt_masks = objects_prompt_masks.reshape(n_obj, -1)
                        vp_embeds.append(tile_vit_embeds[objects_prompt_masks])
                        i_vp_img += 1
                vp_embeds = torch.cat(vp_embeds, dim=0)
            elif video_prompt_masks is not None and video_prompt_masks_64 is not None:
                vp_embeds = []
                prompt_masks_item = video_prompt_masks
                prompt_masks_item_112 = video_prompt_masks_64
                mask_count_item = mask_count
                sample_vit_embeds_item = vit_embeds
                num_imgs_item, hw, C = original_vit_embeds.shape

                spatial_dim = int(hw ** 0.5)
                sample_vit_embeds_item_reshaped = sample_vit_embeds_item.view(sample_vit_embeds_item.shape[0], spatial_dim, spatial_dim, C).permute(0, 3, 1, 2)
                original_vit_embeds = original_vit_embeds.reshape(-1, spatial_dim, spatial_dim, C).permute(0, 3, 1, 2)

                raw_dtype = original_vit_embeds.dtype
                vp_embeds.append(sample_vit_embeds_item.reshape(-1, C))
                mask_index = 0
                if len(prompt_masks_item) != 0 and sum(mask_count_item) != 0:
                    obj_mask = [item[sparse_indices, :, :].to(sample_vit_embeds_item.device).bool().view(num_imgs_item, -1) for item in
                                prompt_masks_item]
                    obj_mask_112 = [item[sparse_indices, :, :].to(sample_vit_embeds_item.device).bool().view(num_imgs_item, -1) for item in
                                prompt_masks_item_112]
                    for index, item_count in enumerate(mask_count_item):
                        vp_embeds.append(spatial_temporal_token[index].reshape(-1, C))
                        if item_count != 0:
                            current_obj_masks = obj_mask[mask_index:(mask_index + item_count)]
                            current_obj_masks_112 = obj_mask_112[mask_index:(mask_index + item_count)]
                            for idx, (single_mask, single_mask_112) in enumerate(zip(current_obj_masks, current_obj_masks_112)):
                                single_mask = single_mask.view(num_imgs_item, spatial_dim, spatial_dim).unsqueeze(1)  # [num_img, 1, h, w]
                                single_mask = single_mask[dense_indices]
                                single_mask = single_mask.to(torch.bfloat16)
                                sum_single_mask = single_mask.view(single_mask.size(0), -1).sum(dim=1)
                                nonzero_indices = (sum_single_mask > 0).nonzero(as_tuple=True)[0]
                                if len(nonzero_indices) == 0:
                                    selected_idx = 0
                                else:
                                    selected_idx = random.choice(nonzero_indices.tolist())

                                single_mask = single_mask[selected_idx].unsqueeze(0)

                                single_mask_112 = single_mask_112.view(num_imgs_item, -1).to(self.Qformer_mask_proj.layers[0].weight.device).to(self.Qformer_mask_proj.layers[0].weight.dtype)
                                single_mask_112 = single_mask_112[dense_indices]
                                single_mask_112 = single_mask_112[selected_idx].unsqueeze(0)
                                sample_vit_embeds_item_reshaped_item = sample_vit_embeds_item_reshaped[selected_idx].unsqueeze(0)
                                pooled_feature = self.mask_pooling(sample_vit_embeds_item_reshaped_item.to(single_mask.dtype), single_mask)  # [1, num_imgs_item, C]
                                pooled_feature = pooled_feature.to(self.Qformer_mask_pooling_proj.weight.dtype).to(self.Qformer_mask_pooling_proj.weight.device)
                                pooled_feature = self.Qformer_mask_pooling_proj(pooled_feature)
                                pooled_feature = pooled_feature.reshape(-1, pooled_feature.shape[-1])
                                pooled_feature = pooled_feature.to(raw_dtype)
                                mask_feature = self.Qformer_mask_proj(single_mask_112)
                                vp_embeds.append(pooled_feature)  #
                                vp_embeds.append(mask_feature)
                            mask_index = mask_index + item_count
                else:
                    vp_embeds.append(spatial_temporal_token[0].reshape(-1, C))
            elif spatial_temporal_token[0] is not None:
                vp_embeds = []
                for i_img in range(len(vit_embeds)):
                    vp_embeds.append(vit_embeds[i_img].reshape(-1, C).to(input_embeds.device))
                vp_embeds.append(spatial_temporal_token[0].reshape(-1, st_dim).to(input_embeds.device))
            else:
                vp_embeds = None
            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            if vp_embeds is None:
                input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
            else:
                ori_vp_embeds = vp_embeds
                vp_embeds = torch.cat(vp_embeds, dim=0)
                if len(input_embeds[selected]) != len(vp_embeds.reshape(-1, C)):
                    for item in ori_vp_embeds:
                        print(f"item shape in ori_vp_embeds: {item.shape}")
                    print("prompts is: {}".format(prompts))
                    print("Shape mismatch, selected is {}, vp embeds is {} !!!" \
                          .format(len(input_embeds[selected]), len(vp_embeds.reshape(-1, C))))
                    min_tokens = min(len(input_embeds[selected]), len(vp_embeds.reshape(-1, C)))
                    input_embeds[selected][:min_tokens] = vp_embeds.reshape(-1, C)[:min_tokens].to(input_embeds.device)
                else:
                    input_embeds[selected] = vp_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask.to(device),
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs

    def preparing_for_generation(self, tokenizer, max_new_tokens=2048, torch_dtype=torch.bfloat16):
        # set stop criteria and generation configs for model
        if not hasattr(self, 'tokenizer'):
            self.tokenizer = tokenizer
        self.bot_name = 'BOT'
        stop_words = []
        stop_words += self.template.get('STOP_WORDS', [])
        stop_criteria = get_stop_criteria(
            tokenizer=self.tokenizer, stop_words=stop_words)
        self.stop_criteria = stop_criteria

        default_generation_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=(
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            ),
        )

        self.gen_config = GenerationConfig(**default_generation_kwargs)
        self.init_prediction_config = True
        self.torch_dtype = torch_dtype
        self.to(torch_dtype)
        self.extra_image_processor = DirectResize(target_length=1024, )
        # for multi image process
        self.min_dynamic_patch = 1
        self.max_dynamic_patch = 12
        self.downsample_ratio = 0.5
        self.image_size = 448
        self.use_thumbnail = True
        patch_size = 14
        self.patch_size = patch_size

        self.patch_token = int((self.image_size // patch_size) ** 2 * (self.downsample_ratio ** 2))
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)
        self.IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        self.IMG_START_TOKEN = '<img>'
        self.IMG_END_TOKEN = '</img>'

        self.transformer = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])
        self.VP_START_TOKEN = '<vp>'
        self.VP_END_TOKEN = '</vp>'

        # change phi3 prepare for generation fuction
        if self.config.llm_config.architectures[0] == 'Phi3ForCausalLM':
            self.language_model.prepare_inputs_for_generation = MethodType(prepare_inputs_for_generation_phi3, self.language_model)

        img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        self.img_context_token_id = img_context_token_id
        self.seg_token_idx = tokenizer.convert_tokens_to_ids('[SEG]')
        return

    def uniform_sample(self, total_len, sample_num):
        intervals = np.linspace(start=0, stop=total_len, num=sample_num + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]

        return frame_idxs

    # def get_sparse_indices(self, total_frame_num, num_frames_sparse):
    #     if total_frame_num > num_frames_sparse:  # video is long, uniformly sample frames
    #         frame_idxs = self.uniform_sample(total_frame_num, num_frames_sparse)
    #         return sorted(frame_idxs)
    #     else:
    #         num_repeat = num_frames_sparse // total_frame_num
    #         num_sample = num_frames_sparse % total_frame_num
    #         frame_idxs = list(range(total_frame_num)) * num_repeat + self.uniform_sample(total_frame_num, num_sample)
    #         return sorted(frame_idxs)

    def get_sparse_indices_uniform(self, vid_len, interval=3, max_frames=32, min_frames=8):
        if vid_len <= 0:
            return []

        sample_indices = list(range(0, vid_len, interval))

        if len(sample_indices) > max_frames:
            step = len(sample_indices) / max_frames
            sample_indices = [sample_indices[int(i * step)] for i in range(max_frames)]

        if len(sample_indices) < min_frames:
            additional_needed = min_frames - len(sample_indices)
            extra_step = vid_len / (additional_needed + 1)
            extra_frames = [int((i + 1) * extra_step) for i in range(additional_needed)]
            sample_indices.extend(extra_frames)
            sample_indices = sorted(set(sample_indices))

        return sorted(sample_indices)

    def get_dense_indices(self, num_frames_temporal, num_frames_dense):
        intervals = np.linspace(start=0, stop=num_frames_temporal - 1, num=num_frames_dense + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]

        return frame_idxs

    def predict_forward(
            self,
            image=None,
            video=None,
            text=None,
            past_text='',
            mask_prompts=None,
            tokenizer=None,
            prompt_masks=None,
            prompt_masks_112=None,
            text_prompts=None,
            mask_count=None,
            prediction_only=True,
            max_frames=32,
    ):
        if not self.init_prediction_config:
            assert tokenizer
            self.preparing_for_generation(tokenizer=tokenizer)
        num_context_token = None
        context_str = None
        dense_indices = None
        if image is None and video is None and '<image>' not in past_text:
            text = text.replace('<image>', "")
            num_context_token = 1
            context_str = self.IMG_CONTEXT_TOKEN * num_context_token + '\n'
            input_text = ''
            input_text += self.template['INSTRUCTION'].format(
                input=text, round=1, bot_name=self.bot_name)
            input_text = past_text + input_text
            ids = self.tokenizer.encode(input_text)
            ids = torch.tensor(ids).cuda().unsqueeze(0)

            attention_mask = torch.ones_like(ids, dtype=torch.bool)

            mm_inputs = {
                'pixel_values': torch.zeros(1, 3, self.image_size, self.image_size),
                'input_ids': ids,
                'attention_mask': attention_mask,
                'position_ids': None,
                'past_key_values': None,
                'labels': None,
                'prompt_masks': None,
                'prompts': text,
                'vp_overall_mask': None,
                'dense_indices': dense_indices,
            }
            ret_masks = []
        else:
            input_dict = {}
            sparse_indices = None
            dense_indices = None
            if video is not None:
                pixel_values = []
                extra_pixel_values = []

                ori_image_size = video[0].size
                # sparse_indices = list(range(0, len(video)))
                sparse_indices = self.get_sparse_indices_uniform(len(video), max_frames=max_frames)
                dense_indices = [0,1,2,3,4]

                desired_dense_num = 5
                num_sparse = len(sparse_indices)

                if num_sparse < desired_dense_num:
                    dense_indices = list(range(num_sparse))

                for frame_idx, frame_image in enumerate(video):
                    assert ori_image_size == frame_image.size
                    g_image = np.array(frame_image)  # for grounding
                    g_image = self.extra_image_processor.apply_image(g_image)
                    g_image = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
                    extra_pixel_values.append(g_image)

                for selected_frame_index in sparse_indices:
                    frame_image = video[selected_frame_index]
                    img = self.transformer(frame_image)
                    pixel_values.append(img)

                pixel_values = torch.stack(pixel_values, dim=0).to(self.torch_dtype)  # (n_f, 3, h, w)
                g_pixel_values = torch.stack([
                    self.grounding_encoder.preprocess_image(pixel) for pixel in extra_pixel_values
                ]).to(self.torch_dtype)
                num_image_tokens = self.patch_token
                num_frames = len(dense_indices)

                num_context_token = len(sparse_indices)
                context_str = self.IMG_CONTEXT_TOKEN * num_context_token + '\n'
                input_dict['vp_overall_mask'] = None
            else:
                ori_image_size = image.size

                # prepare grounding images
                g_image = np.array(image)  # for grounding
                g_image = self.extra_image_processor.apply_image(g_image)
                g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous().to(self.torch_dtype)
                extra_pixel_values = [g_pixel_values]
                g_pixel_values = torch.stack([
                    self.grounding_encoder.preprocess_image(pixel) for pixel in extra_pixel_values
                ]).to(self.torch_dtype)

                images = dynamic_preprocess(image, self.min_dynamic_patch,
                                            self.max_dynamic_patch,
                                            self.image_size, self.use_thumbnail)

                if mask_prompts is not None:
                    vp_overall_mask = torch.Tensor([False] * (len(images) - 1) + [True])
                    input_dict['vp_overall_mask'] = vp_overall_mask
                else:
                    input_dict['vp_overall_mask'] = None

                pixel_values = [self.transformer(image) for image in images]
                pixel_values = torch.stack(pixel_values).to(self.torch_dtype)
                num_image_tokens = pixel_values.shape[0] * self.patch_token
                num_frames = 1

                num_context_token = pixel_values.shape[0]
                context_str = self.IMG_CONTEXT_TOKEN * num_context_token + '\n'
            input_dict['g_pixel_values'] = g_pixel_values
            input_dict['pixel_values'] = pixel_values

            if mask_prompts is not None:
                # reshape mask prompts to feature size
                mask_prompts = [torch.Tensor(item).to(pixel_values.device) for item in mask_prompts]
                mask_prompts = [F.interpolate(
                    item.unsqueeze(0),
                    size=(int(self.image_size // self.patch_size * self.downsample_ratio),
                          int(self.image_size // self.patch_size * self.downsample_ratio)),
                    mode='nearest').squeeze(0) for item in mask_prompts]
                region_pixels = []
                for mask_prompt in mask_prompts[0]:
                    region_pixels.append(mask_prompt.bool().to(torch.int64).sum())

                vp_token_str = '\nThere are {} part regions in the picture: '.format(len(mask_prompts[0]))
                for i in range(len(mask_prompts[0])):
                    vp_token_str = vp_token_str + \
                                   f"region{i + 1}" + self.VP_START_TOKEN + \
                                   self.IMG_CONTEXT_TOKEN * region_pixels[i] + \
                                   self.VP_END_TOKEN
                    if i == len(mask_prompts[0]) - 1:
                        vp_token_str = vp_token_str + '.\n'
                    else:
                        vp_token_str = vp_token_str + ', '
            else:
                vp_token_str = ''

            image_token_str = f'{self.IMG_START_TOKEN}' \
                              f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                              f'{self.IMG_END_TOKEN}'
            image_token_str = image_token_str + '\n'
            image_token_str = image_token_str * num_frames
            image_token_str = image_token_str.strip()

            ret_masks = []

            if '<image>' in text or mask_prompts is not None:
                assert past_text is None or len(past_text) == 0
            if text_prompts is None or self.IMG_CONTEXT_TOKEN not in text:
                text_prompts = text.replace('<image>', '').replace('<video>', '').replace('\n', '').strip()
                text = text.replace('<image>', '').replace('<video>', '').replace('\n', '').strip()
                text = image_token_str + context_str + text
            input_text = ''
            input_text += self.template['INSTRUCTION'].format(
                input=text, round=1, bot_name=self.bot_name)
            input_text = past_text + input_text
            ids = self.tokenizer.encode(input_text)
            ids = torch.tensor(ids).cuda().unsqueeze(0)

            attention_mask = torch.ones_like(ids, dtype=torch.bool)

            mm_inputs = {
                'pixel_values': input_dict['pixel_values'],
                'input_ids': ids,
                'attention_mask': attention_mask,
                'position_ids': None,
                'past_key_values': None,
                'labels': None,
                'prompt_masks': mask_prompts,
                'sparse_indices': sparse_indices,
                'dense_indices': dense_indices,
                'prompts': text_prompts,
                'mask_count': mask_count,
                'video_prompt_masks': prompt_masks,
                'video_prompt_masks_64': prompt_masks_112,
                'vp_overall_mask': input_dict['vp_overall_mask'],
            }

        generate_output = self.generate(
            **mm_inputs,
            generation_config=self.gen_config,
            streamer=None,
            bos_token_id=self.tokenizer.bos_token_id,
            stopping_criteria=self.stop_criteria,
            output_hidden_states=True,
            return_dict_in_generate=True
        )
        predict = self.tokenizer.decode(
            generate_output.sequences[0], skip_special_tokens=False).strip()

        if image is None and video is None and '<image>' not in past_text:
            return {'prediction': predict, 'prediction_masks': ret_masks, }
        if prediction_only is True:
            return {'prediction': predict, 'prediction_masks': None, }
        # if have seg result, find the seg hidden states
        hidden_states = generate_output.hidden_states
        last_hidden_states = [item[-1][0] for item in hidden_states]
        last_hidden_states = torch.cat(last_hidden_states, dim=0)
        seg_hidden_states = get_seg_hidden_states(
            last_hidden_states, generate_output.sequences[0][:-1],
            seg_id=self.seg_token_idx
        )
        all_seg_hidden_states = self.text_hidden_fcs(seg_hidden_states)

        for seg_hidden_states in all_seg_hidden_states:
            seg_hidden_states = seg_hidden_states.unsqueeze(0)
            g_pixel_values = input_dict['g_pixel_values']
            sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values)
            pred_masks = self.grounding_encoder.language_embd_inference(sam_states, [seg_hidden_states] * num_frames)
            w, h = ori_image_size
            masks = F.interpolate(pred_masks, size=(h, w), mode='bilinear', align_corners=False)
            masks = masks[:, 0]
            masks = masks.sigmoid() > 0.5
            masks = masks.cpu().numpy()
            ret_masks.append(masks)

        return {'prediction': predict, 'prediction_masks': ret_masks,}

def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    if n_out == 0:
        return hidden_states[0:0]
    return hidden_states[-n_out:][seg_mask]

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height,
                              image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image,
                       min_num=1,
                       max_num=6,
                       image_size=448,
                       use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = {(i, j)
                     for n in range(min_num, max_num + 1)
                     for i in range(1, n + 1) for j in range(1, n + 1)
                     if i * j <= max_num and i * j >= min_num}
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio,
                                                    target_ratios, orig_width,
                                                    orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size,
               (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size,
               ((i // (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


from transformers.cache_utils import Cache, DynamicCache

def prepare_inputs_for_generation_phi3(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens
            max_cache_length = past_key_values.get_max_length()
        else:
            cache_length = past_length = past_key_values[0][0].shape[2]
            max_cache_length = None

        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get('position_ids', None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1]:]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and (past_key_values is None or len(past_key_values)==0):
        model_inputs = {'inputs_embeds': inputs_embeds}
    else:
        model_inputs = {'input_ids': input_ids}

    model_inputs.update(
        {
            'position_ids': position_ids,
            'past_key_values': past_key_values,
            'use_cache': kwargs.get('use_cache'),
            'attention_mask': attention_mask,
        }
    )
    return model_inputs

