import torch
from xtuner.model import InternVL_V1_5
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast
import logging
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer)
import torch.nn as nn

from mmengine import print_log
from torch.nn import CrossEntropyLoss
from transformers import (AutoConfig, AutoModel, AutoTokenizer,
                          BitsAndBytesConfig)
from xtuner.model.utils import (find_all_linear_names, get_peft_model_state_dict,
                                guess_load_checkpoint, make_inputs_require_grad)
import os
from projects.llava_sam2.models.qformer import BertConfig, BertLMHeadModel
from transformers import BertTokenizer
import torch.nn.functional as F
# from projects.llava_sam2.models.utils import MaskPooling, MLP
import random
def get_rank_and_world_size():
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    return rank, world_size


class MLP(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MaskPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask):

        if not x.shape[-2:] == mask.shape[-2:]:
            # reshape mask to x
            mask = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=False)

        b, c, h ,w = x.shape
        b, q, h, w = mask.shape
        mask = (mask > 0).to(mask.dtype)
        denorm = mask.sum(dim=(-1, -2), keepdim=True) + 1e-8

        mask_pooled_x = torch.einsum(
            "bchw,bqhw->bqc",
            x,
            mask / denorm,
        )
        return mask_pooled_x


# This function is used to split large model
def split_model(model_name):
    import math
    device_map = {}
    num_gpus = torch.cuda.device_count()
    rank, world_size = get_rank_and_world_size()
    num_gpus = num_gpus // world_size

    num_layers = {'InternVL2-8B': 32, 'InternVL2-26B': 48,
                  'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as 0.8 GPU.
    num_layers_per_gpu = math.ceil(num_layers / (num_gpus - 0.2))
    num_layers_per_gpu = [num_layers_per_gpu] * num_gpus
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.8)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = rank + world_size * i
            layer_cnt += 1
    device_map['vision_model'] = rank
    device_map['mlp1'] = rank
    device_map['language_model.model.tok_embeddings'] = rank
    device_map['language_model.model.embed_tokens'] = rank
    device_map['language_model.output'] = rank
    device_map['language_model.model.norm'] = rank
    device_map['language_model.lm_head'] = rank
    device_map[f'language_model.model.layers.{num_layers - 1}'] = rank
    return device_map


class InternVL_Slowfast(InternVL_V1_5):

    def __init__(self,
                 model_path,
                 freeze_llm=False,
                 freeze_visual_encoder=False,
                 llm_lora=None,
                 num_temporal_token=32,
                 visual_encoder_lora=None,
                 quantization_vit=False,
                 quantization_llm=False,
                 pretrained_pth=None,
                 special_tokens=None,
                 model_split=False,
                 window_size=512,
                 stride=512,
                 ):
        print_log('Start to load InternVL_V1_5 model.', logger='current')
        super(InternVL_V1_5, self).__init__()
        if "Sa2VA-1B" in model_path:
            out_channels = 896
        elif "Sa2VA-4B" in model_path:
            out_channels = 2048
        elif "Sa2VA-8B" in model_path or 'InternVL2_5-8B' in model_path:
            out_channels = 4096
        elif "Sa2VA-26B" in model_path:
            out_channels = 6144

        self.freeze_llm = freeze_llm
        self.freeze_visual_encoder = freeze_visual_encoder
        self.use_llm_lora = llm_lora is not None
        self.use_visual_encoder_lora = visual_encoder_lora is not None
        self.quantization_vit = quantization_vit
        self.quantization_llm = quantization_llm
        if quantization_vit:
            assert visual_encoder_lora is not None
        if quantization_llm:
            assert quantization_llm and llm_lora is not None

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if config.llm_config.model_type == 'internlm2':
            config.llm_config.attn_implementation = 'flash_attention_2'
        else:
            config.llm_config._attn_implementation = 'flash_attention_2'

        if quantization_vit is False and quantization_llm is False:
            quantization = None
        else:
            llm_int8_skip_modules = ['mlp1']
            if quantization_llm and not quantization_vit:
                llm_int8_skip_modules.append('vision_model')

            if quantization_vit and not quantization_llm:
                llm_int8_skip_modules.append('language_model')

            quantization_config = dict(
                type=BitsAndBytesConfig,
                llm_int8_skip_modules=llm_int8_skip_modules,
                load_in_4bit=True,
                load_in_8bit=False,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4')
            quantization_clazz = quantization_config.pop('type')
            quantization = quantization_clazz(**quantization_config)

        if model_split:
            # print("\n\nDone Model Split !!!!!!!!!!!\n\n")
            device_map = split_model("InternVL2-26B")
            # print(device_map)
            self.device = 'cuda'
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map=device_map).eval()

        else:
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                quantization_config=quantization,
                config=config,
                trust_remote_code=True)

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
        self.tokenizer = tokenizer

        if special_tokens is not None:
            self._add_special_tokens(special_tokens)

        img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        self.model.img_context_token_id = img_context_token_id

        if self.freeze_llm:
            self.model.language_model.requires_grad_(False)
        if self.freeze_visual_encoder:
            self.model.vision_model.requires_grad_(False)

        if hasattr(self.model.language_model, 'enable_input_require_grads'):
            self.model.language_model.enable_input_require_grads()
        else:
            self.model.language_model.get_input_embeddings(
            ).register_forward_hook(make_inputs_require_grad)

        self.gradient_checkpointing_enable()

        if self.use_llm_lora:
            self._prepare_llm_for_lora(llm_lora)

        if self.use_visual_encoder_lora:
            self._prepare_visual_encoder_for_lora(visual_encoder_lora)

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)

            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f'Load pretrained weight from {pretrained_pth}')

        self._count = 0
        print_log(self, logger='current')
        print_log('InternVL_V1_5 construction is complete', logger='current')

        self.transfer_to_hf = False

        ###########################################
        #### Init Spatial and Temporal Qformer ####
        ###########################################
        self.mask_pooling = MaskPooling()
        def get_w(weights, keyword):
            return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
        self.Qformer_tokenizer = self.init_tokenizer()
        self.Qformer_temporal_tokenizer = self.init_temporal_tokenizer()
        self.spatial_Qformer, self.spatial_Qformer_query_tokens = self.init_spatial_Qformer(num_query_token=32, vision_width=1408)
        self.spatial_Qformer.resize_token_embeddings(len(self.Qformer_tokenizer))
        self.spatial_Qformer.cls = None

        self.temporal_Qformer, self.temporal_Qformer_query_tokens = self.init_temporal_Qformer(num_query_token=num_temporal_token, vision_width=self.spatial_Qformer.config.hidden_size, num_hidden_layers=2)
        self.temporal_Qformer.resize_token_embeddings(len(self.Qformer_temporal_tokenizer))
        self.temporal_Qformer.cls = None

        self.Qformer_mask_pooling_proj = nn.Linear(in_features=out_channels, out_features=out_channels)
        self.Qformer_mask_proj = MLP(112*112, 1024, out_channels, 3)

        self.Qformer_mask_pooling_proj_st = nn.Linear(in_features=out_channels, out_features=self.temporal_Qformer.config.hidden_size)
        self.Qformer_mask_proj_st = MLP(112*112, 1024, self.temporal_Qformer.config.hidden_size, 3)

        self.spatial_Qformer_ln = torch.nn.LayerNorm(1408)

        print("Loading pretrained qformer weights...")
        pretrain_qformer = '/datasets/datasets_sysy/model_zoo/LAVIS/instruct_blip_vicuna7b_trimmed.pth'
        qformer_weight = torch.load(pretrain_qformer, map_location='cpu')['model']
        bert_weight = {_key: qformer_weight[_key] for _key in qformer_weight if 'bert' in _key}

        self.spatial_Qformer.load_state_dict(get_w(bert_weight, 'Qformer'))
        self.spatial_Qformer_ln.load_state_dict(get_w(qformer_weight, 'ln_vision'))
        self.spatial_Qformer_query_tokens.data = qformer_weight['query_tokens']

        self.spatial_Qformer.bert.embeddings.word_embeddings = None
        self.spatial_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.spatial_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.vlm2spatial_Qformer_proj = nn.Linear(out_channels, 1408)

        self.Qformer_temp_attn_q = torch.nn.Linear(out_channels, out_channels)
        self.Qformer_temp_attn_k = torch.nn.Linear(self.spatial_Qformer.config.hidden_size, out_channels)
        self.Qformer_temp_attn_v = torch.nn.Linear(self.spatial_Qformer.config.hidden_size, out_channels)
        self.Qformer_temp_proj = torch.nn.Linear(out_channels, out_channels)
        self.Qformer_final_proj = nn.Linear(out_channels, out_channels)

        self.window_size = window_size
        self.stride = stride

    def _add_special_tokens(self, special_tokens):
        num_new_tokens = self.tokenizer.add_tokens(
            special_tokens, special_tokens=True)

        if num_new_tokens > 0:
            self.model.language_model.resize_token_embeddings(len(self.tokenizer))

    def _post_init(self, fast_pool_size=4, fast_pool=True):
        if fast_pool:
            self.fast_pool = nn.AdaptiveAvgPool2d((fast_pool_size, fast_pool_size))
        return

    @classmethod
    def init_tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side='left')
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})        # "<p>", "</p>", "[SEG]", "<vp>", "</vp>"
        return tokenizer

    @classmethod
    def init_temporal_tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side='left')

        special_tokens_dict = {
            "bos_token": "[DEC]",  # 仍保留原先作为 bos_token
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
        # 为 tokenizer 注册新的特殊词表
        tokenizer.add_special_tokens(special_tokens_dict)
        return tokenizer

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
            random_idx_item = random_idx_list[_idx]
            if dense_indices is not None and dense_indices[_idx] is not None:
                dense_indices_item = dense_indices[_idx]
                dense_img_feat_prompt = img_feat_prompt[dense_indices_item]
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
                    random_idx = random_idx_item[tp_idx]
                    temp_mask_count = item_mask_count[tp_idx]
                    item_mask_feats = []
                    item_visual_feats = []
                    if temp_mask_count == 0:
                        z_vis = torch.zeros(1, self.Qformer_mask_pooling_proj_st.in_features,
                                            device=image_features.device, dtype=image_features.dtype)
                        z_mask = torch.zeros(1, 112 * 112,
                                             device=image_features.device, dtype=image_features.dtype)
                        z_vis = self.Qformer_mask_pooling_proj_st(z_vis) * 1e-6  # dummy call 
                        z_mask = self.Qformer_mask_proj_st(z_mask) * 1e-6 # dummy call
                        if dummy_visual_feat is None:
                            dummy_visual_feat = z_vis
                        else:
                            dummy_visual_feat = dummy_visual_feat + z_vis
                        if dummy_mask_feat is None:
                            dummy_mask_feat = z_mask
                        else:
                            dummy_mask_feat = dummy_mask_feat + z_mask
                        mask_feat_list.append([])
                        visual_feat_list.append([])
                        continue
                    cur_mask = current_obj_mask[mask_index: mask_index + temp_mask_count]
                    cur_mask_112 = current_obj_mask_112[mask_index:mask_index + temp_mask_count]
                    temp_image_feat = dense_img_feat_prompt[random_idx].unsqueeze(0)
                    num_img, hw, C = temp_image_feat.shape
                    spatial_dim = int(hw ** 0.5)
                    temp_image_feat_reshaped = temp_image_feat.view(1, spatial_dim, spatial_dim, C).permute(0, 3, 1, 2)
                    for tj in range(temp_mask_count):
                        temp_obj_mask = cur_mask[tj].to(image_features.device).to(torch.bfloat16)
                        temp_obj_mask_112 = cur_mask_112[tj].to(self.Qformer_mask_proj_st.layers[0].weight.device).to(self.Qformer_mask_proj_st.layers[0].weight.dtype)
                        temp_obj_mask = temp_obj_mask[random_idx].unsqueeze(0).unsqueeze(0)
                        temp_obj_mask_112 = temp_obj_mask_112[random_idx].unsqueeze(0).flatten(1, 2)

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
                if dummy_visual_feat is None:
                    dummy_visual_feat = torch.zeros(
                        1,
                        self.Qformer_mask_pooling_proj_st.in_features,
                        device=image_features.device,
                        dtype=image_features.dtype
                    )
                    dummy_visual_feat = self.Qformer_mask_pooling_proj_st(dummy_visual_feat) * 1e-6
                else:
                    dummy_visual_featV2 = torch.zeros(
                        1,
                        self.Qformer_mask_pooling_proj_st.in_features,
                        device=image_features.device,
                        dtype=image_features.dtype
                    )
                    dummy_visual_feat = dummy_visual_feat + self.Qformer_mask_pooling_proj_st(dummy_visual_featV2) * 1e-6
                if dummy_mask_feat is None:
                    dummy_mask_feat = torch.zeros(
                        1,
                        112 * 112,
                        device=image_features.device,
                        dtype=image_features.dtype
                    )
                    dummy_mask_feat = self.Qformer_mask_proj_st(dummy_mask_feat) * 1e-6
                else:
                    dummy_mask_featV2 = torch.zeros(
                        1,
                        112 * 112,
                        device=image_features.device,
                        dtype=image_features.dtype
                    )
                    dummy_mask_feat = dummy_mask_feat + self.Qformer_mask_proj_st(dummy_mask_featV2) * 1e-6

                visual_feat_list = None
                mask_feat_list = None

            img_att_prompt = image_atts[total_count:total_count + image_counts[_idx]]
            # img_att_prompt = img_att_prompt[None].expand(len(prompts[_idx]), -1, -1).flatten(0, 1)

            input_ids = input_ids[:, None].expand(-1, image_counts[_idx], -1).flatten(0, 1)
            attention_masks = attention_masks[:, None].expand(-1, image_counts[_idx], -1).flatten(0, 1)
            total_count += image_counts[_idx]

            bert_feat = self.vlm2spatial_Qformer_proj(img_feat_prompt)

            query_tokens = self.spatial_Qformer_query_tokens.expand(bert_feat.shape[0], -1, -1)

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

                    temporal_mm_output = self.temporal_Qformer.bert(        # todo: 改进的话只能是参考MA_LMM的方式引入memory bank
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
                    # temporal_mm_output = temporal_mm_output.last_hidden_state[:, :temporal_Qformer_query_tokens.shape[1]]
                    temporal_outputs.append(temporal_mm_output.last_hidden_state[:, :temporal_Qformer_query_tokens.shape[1]])

                temporal_outputs = torch.cat(temporal_outputs, dim=1)
                mm_output = self.temporal_context_inject(img_feat_prompt_expand, temporal_outputs, image_counts[_idx])
                if image_counts is not None:
                    mm_output = mm_output.reshape(len(prompts[_idx]), image_counts[_idx],
                                                                          *mm_output.shape[-2:])
                    mm_output = mm_output.flatten(1, 2)
                spatial_temporal_token_list.append(mm_output)
        return spatial_temporal_token_list

    def temporal_context_inject(self, vis_embed, temp_embed, image_count=None, chunk=16):
        num_prompts, num_token, channels = temp_embed.shape
        if num_prompts > chunk:
            ctx_embed_list = []
            for start in range(0, num_prompts, chunk):
                end = min(start + chunk, num_prompts)
                ft_s = start * image_count
                ft_e = end * image_count
                pt_s = start
                pt_e = end
                sub_temp_embed = temp_embed[pt_s:pt_e, : , :]
                sub_vis_embed = vis_embed[ft_s:ft_e, : , :]
                sub_temp_embed = sub_temp_embed.unsqueeze(1).expand(-1, image_count, -1, -1).reshape((end - start) * image_count, num_token, channels)
                query = self.Qformer_temp_attn_q(sub_vis_embed)
                key = self.Qformer_temp_attn_k(sub_temp_embed)
                value = self.Qformer_temp_attn_v(sub_temp_embed)
                # Key part 1: calculate context-related embedding
                ctx_embed_item = query @ key.transpose(-1, -2)
                ctx_embed_item = ctx_embed_item / (key.shape[-1] ** 0.5)
                ctx_embed_item = ctx_embed_item.softmax(-1) @ value
                ctx_embed_item  = self.Qformer_temp_proj(ctx_embed_item) + sub_vis_embed
                ctx_embed_item = torch.mean(ctx_embed_item, dim=1, keepdim=True)
                ctx_embed_item = self.Qformer_final_proj(ctx_embed_item)
                ctx_embed_list.append(ctx_embed_item)
            ctx_embed = torch.cat(ctx_embed_list, dim=0)
            return ctx_embed
        else:
            temp_embed = temp_embed.unsqueeze(1).expand(-1, image_count, -1, -1).reshape(num_prompts * image_count, num_token, channels)
            # query = vis_embed
            query = self.Qformer_temp_attn_q(vis_embed)
            key = self.Qformer_temp_attn_k(temp_embed)
            value = self.Qformer_temp_attn_v(temp_embed)
            # Key part 1: calculate context-related embedding
            ctx_embed = query @ key.transpose(-1, -2)
            ctx_embed = ctx_embed / (key.shape[-1] ** 0.5)
            ctx_embed = ctx_embed.softmax(-1) @ value
            ctx_embed = self.Qformer_temp_proj(ctx_embed)
            ctx_embed = ctx_embed + vis_embed
            ctx_embed = torch.mean(ctx_embed, dim=1, keepdim=True)
            ctx_embed = self.Qformer_final_proj(ctx_embed)

            return ctx_embed

    def forward(self, data, data_samples=None, mode='loss', fast_token_idx=None):
        if 'fast_pixel_values' in data.keys():
            assert fast_token_idx is not None
            fast_pixel_values = data['fast_pixel_values']
            if type(fast_pixel_values) is list or fast_pixel_values.ndim == 5:
                if type(fast_pixel_values) is list:
                    fast_pixel_values = [
                        x.unsqueeze(0) if x.ndim == 3 else x for x in fast_pixel_values
                    ]
                # b*n, c, h, w
                fast_concat_images = torch.cat(
                    [image.to(self.model.vision_model.dtype) for image in fast_pixel_values], dim=0)
            else:
                raise NotImplementedError()
        else:
            fast_pixel_values = None
            fast_concat_images = None
        image_counts = None
        pixel_values = data['pixel_values']
        if type(pixel_values) is list or pixel_values.ndim == 5:
            if type(pixel_values) is list:
                pixel_values = [
                    x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values
                ]
            image_counts = [image.shape[0] for image in pixel_values]
            # b*n, c, h, w
            concat_images = torch.cat(
                [image.to(self.model.vision_model.dtype) for image in pixel_values], dim=0)
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

        if 'mask_count' in data.keys():
            mask_count = data['mask_count']
        else:
            mask_count = None

        if 'prompt_masks_112' in data.keys():
            prompt_masks_112 = data['prompt_masks_112']
        else:
            prompt_masks_112 = None

        if 'prompts' in data.keys():
            prompts = data['prompts']
        else:
            prompts = None

        if 'dense_indices' in data.keys():
            dense_indices = data['dense_indices']
        else:
            dense_indices = None

        outputs = self._llm_forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            pixel_values=concat_images,
            image_counts=image_counts,
            prompts=prompts,
            labels=labels,
            use_cache=use_cache,
            output_hidden_states=True,
            fast_pixel_values=fast_concat_images,
            fast_token_idx=fast_token_idx,
            vp_overall_mask=vp_overall_mask,
            prompt_masks=prompt_masks,
            prompt_masks_112=prompt_masks_112,
            mask_count=mask_count,
            dense_indices=dense_indices,
        )

        return outputs

    def _llm_forward(
            self,
            pixel_values: torch.FloatTensor,
            image_counts=None,
            prompts=None,
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
            fast_pixel_values=None,
            fast_token_idx=None,
            vp_overall_mask=None,
            prompt_masks=None,
            prompt_masks_112=None,
            mask_count=None,
            dense_indices=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # internvl_stV11_alldata.py  (建议放在 _llm_forward 开头，保证最早执行)
        return_dict = return_dict if return_dict is not None \
            else self.model.config.use_return_dict
        image_flags = image_flags.squeeze(-1)
        # We only added the clone code here to avoid the error.
        input_embeds = self.model.language_model.get_input_embeddings()(
            input_ids).clone()
        spatial_temporal_token = None
        st_dim = None
        if fast_pixel_values is not None:
            n_fast_images = fast_pixel_values.shape[0]
            whole_pixel_values = torch.cat([fast_pixel_values, pixel_values], dim=0)
            vit_embeds = self.model.extract_feature(whole_pixel_values)
            vit_embeds = vit_embeds.to(input_embeds.dtype)  # FIXME: why vit_embeds is float16?
            fast_vit_embeds = vit_embeds[:n_fast_images]  # (n_fast_images, hw, c)
            _size = int(fast_vit_embeds.shape[1] ** 0.5)
            fast_vit_embeds = fast_vit_embeds.reshape(fast_vit_embeds.shape[0], _size, _size, fast_vit_embeds.shape[-1])
            # pooling
            fast_vit_embeds = fast_vit_embeds.permute(0, 3, 1, 2)  # (n_fast_images, c, h, w)
            fast_vit_embeds = self.fast_pool(fast_vit_embeds).flatten(2)  # (n_fast_images, c, hw)
            fast_vit_embeds = fast_vit_embeds.permute(0, 2, 1)
            vit_embeds = vit_embeds[n_fast_images:]
        else:
            vit_embeds = self.model.extract_feature(pixel_values)   #pixel_values: torch.Size([6, 3, 448, 448]), vit_embeds:  torch.Size([6, 256, 2048])
            vit_embeds = vit_embeds.to(input_embeds.dtype)  # FIXME: why vit_embeds is float16?
            fast_vit_embeds = None

        # vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        selected_ori = (input_ids == self.model.img_context_token_id).bool()
        selected_ori = selected_ori.view(B, -1).sum(-1)
        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.model.img_context_token_id)
        n_token = selected.sum().item()
        random_idx_list = []
        for i in range(B):
            random_idx_list.append([random.randint(0, 4) for j in range(len(prompts[i]))])

        spatial_temporal_token = self.spatial_temporal_token_generation(vit_embeds, prompts, image_counts, prompt_masks, prompt_masks_112, dense_indices, random_idx_list, mask_count)
        st_dim = spatial_temporal_token[-1].shape[-1]

        self._count += 1
        vp_embeds = []
        image_index = 0

        if image_flags.dim() == 0:
            image_flags = image_flags.unsqueeze(0)

        for batch_idx, num_imgs in enumerate(image_counts):
            random_idx_item = random_idx_list[batch_idx]
            if batch_idx + 1 != len(image_counts):
                vit_embeds_item = vit_embeds[image_index:image_index + num_imgs]
                local_flags = image_flags[image_index: image_index + num_imgs]
            else:
                vit_embeds_item = vit_embeds[image_index:]
                local_flags = image_flags[image_index:]
            spatial_temporal_token_item = spatial_temporal_token[batch_idx]
            if dense_indices is not None:
                dense_indices_item = dense_indices[batch_idx]
            else:
                dense_indices_item = None
            image_index = image_index + num_imgs

            if vp_overall_mask is not None and prompt_masks is not None and vp_overall_mask[batch_idx] is not None and prompt_masks[batch_idx] is not None:
                dummy_input = torch.zeros(
                    1,
                    self.Qformer_mask_pooling_proj.in_features,
                    device=self.Qformer_mask_pooling_proj.weight.device,
                    dtype=self.Qformer_mask_pooling_proj.weight.dtype
                )
                dummy_input = self.Qformer_mask_pooling_proj(dummy_input) * 1e-6
                dummy_mask_input_device = self.Qformer_mask_proj.layers[0].weight.device
                dummy_mask_input_dtype = self.Qformer_mask_proj.layers[0].weight.dtype
                dummy_mask_input = torch.zeros(
                    1,
                    112 * 112,
                    device=dummy_mask_input_device,
                    dtype=dummy_mask_input_dtype
                )
                dummy_mask_input = self.Qformer_mask_proj(dummy_mask_input) * 1e-6

                vp_overall_mask_item = vp_overall_mask[batch_idx]
                vp_overall_mask_item = vp_overall_mask_item.to(vit_embeds_item.device).bool()
                vp_overall_mask_item = vp_overall_mask_item[local_flags == 1]
                prompt_masks_item = prompt_masks[batch_idx]
                prompt_masks_item = [item.to(vit_embeds_item.device).bool() for item in prompt_masks_item]

                vp_embeds.append(vit_embeds_item.reshape(-1, C) + dummy_mask_input + dummy_input)
                if vp_overall_mask_item[-1]:
                    tile_vit_embeds = vit_embeds_item[-1].reshape(-1, C)
                    objects_prompt_masks = torch.stack(prompt_masks_item, dim=0)
                    n_obj = len(objects_prompt_masks)
                    tile_vit_embeds = tile_vit_embeds.unsqueeze(0).repeat(n_obj, 1, 1)
                    objects_prompt_masks = objects_prompt_masks.reshape(n_obj, -1)
                    vp_embeds.append(tile_vit_embeds[objects_prompt_masks])
                vp_embeds.append(spatial_temporal_token_item.reshape(-1, st_dim))

            elif prompt_masks is not None and prompt_masks[batch_idx] is not None:
                prompt_masks_item = prompt_masks[batch_idx]
                prompt_masks_item_112 = prompt_masks_112[batch_idx]
                mask_count_item = mask_count[batch_idx]
                if dense_indices_item is not None:
                    sample_vit_embeds_item = vit_embeds_item[dense_indices_item, :, :]
                else:
                    sample_vit_embeds_item = vit_embeds_item
                num_imgs_item, hw, C = vit_embeds_item.shape
                spatial_dim = int(hw ** 0.5)
                vit_embeds_item_reshaped = vit_embeds_item.reshape(-1, spatial_dim,
                                                                                 spatial_dim, C).permute(0, 3, 1, 2)
                sample_vit_embeds_item_reshaped = sample_vit_embeds_item.reshape(-1, spatial_dim,
                                                                                 spatial_dim, C).permute(0, 3, 1, 2)

                raw_dtype = vit_embeds_item_reshaped.dtype
                vp_embeds.append(sample_vit_embeds_item.reshape(-1, C))
                mask_index = 0
                if len(prompt_masks_item) != 0 and sum(mask_count_item) != 0:
                    obj_mask = [item.to(sample_vit_embeds_item.device).bool() for item in prompt_masks_item]
                    obj_mask_112 = [item.to(sample_vit_embeds_item.device).bool() for item in prompt_masks_item_112]
                    for index, item_count in enumerate(mask_count_item):
                        random_idx = random_idx_item[index]
                        vp_embeds.append(spatial_temporal_token_item[index])
                        if item_count != 0:
                            current_obj_masks = obj_mask[mask_index:(mask_index + item_count)]
                            current_obj_masks_112 = obj_mask_112[mask_index:(mask_index + item_count)]
                            for idx, (single_mask, single_mask_112) in enumerate(zip(current_obj_masks, current_obj_masks_112)):
                                single_mask = single_mask.view(single_mask.shape[0], spatial_dim, spatial_dim).unsqueeze(1)  # [1, num_imgs_item, h, w]
                                single_mask = single_mask.to(torch.bfloat16)
                                random_single_mask = single_mask[random_idx].unsqueeze(0)
                                random_sample_vit_embeds_item_reshaped = sample_vit_embeds_item_reshaped[random_idx].unsqueeze(0)

                                single_mask_112 = single_mask_112.view(single_mask_112.shape[0], -1).to(self.Qformer_mask_proj.layers[0].weight.device).to(self.Qformer_mask_proj.layers[0].weight.dtype)
                                single_mask_112 = single_mask_112[random_idx].unsqueeze(0)
                                pooled_feature = self.mask_pooling(random_sample_vit_embeds_item_reshaped.to(random_single_mask.dtype), random_single_mask)  # [1, num_imgs_item, C]
                                pooled_feature = pooled_feature.to(self.Qformer_mask_pooling_proj.weight.dtype).to(self.Qformer_mask_pooling_proj.weight.device)
                                pooled_feature = self.Qformer_mask_pooling_proj(pooled_feature)
                                pooled_feature = pooled_feature.reshape(-1, pooled_feature.shape[-1])
                                pooled_feature = pooled_feature.to(raw_dtype)
                                mask_feature = self.Qformer_mask_proj(single_mask_112)
                                vp_embeds.append(pooled_feature)  #
                                vp_embeds.append(mask_feature)  #
                            mask_index = mask_index + item_count
                else:
                    vp_embeds.append(spatial_temporal_token_item.reshape(-1, C))
            else:
                dummy_input = torch.zeros(
                    1,
                    self.Qformer_mask_pooling_proj.in_features,
                    device=self.Qformer_mask_pooling_proj.weight.device,
                    dtype=self.Qformer_mask_pooling_proj.weight.dtype
                )
                dummy_input = self.Qformer_mask_pooling_proj(dummy_input) * 1e-6
                dummy_mask_input_device = self.Qformer_mask_proj.layers[0].weight.device
                dummy_mask_input_dtype = self.Qformer_mask_proj.layers[0].weight.dtype
                dummy_mask_input = torch.zeros(
                    1,
                    112 * 112,
                    device=dummy_mask_input_device,
                    dtype=dummy_mask_input_dtype
                )
                dummy_mask_input = self.Qformer_mask_proj(dummy_mask_input) * 1e-6
                if local_flags.sum() != 0:

                    if dense_indices_item is not None:
                        sample_vit_embeds_item = vit_embeds_item[dense_indices_item, :, :]
                    else:
                        sample_vit_embeds_item = vit_embeds_item
                    vp_embeds.append(sample_vit_embeds_item.reshape(-1, C))
                vp_embeds.append(spatial_temporal_token_item.reshape(-1, st_dim) + dummy_mask_input + dummy_input)

        if len(vp_embeds) == 0:
            try:
                dummy_input = torch.zeros(
                    1,
                    self.Qformer_mask_pooling_proj.in_features,
                    device=self.Qformer_mask_pooling_proj.weight.device,
                    dtype=self.Qformer_mask_pooling_proj.weight.dtype
                )
                dummy_input = self.Qformer_mask_pooling_proj(dummy_input) * 1e-6
                dummy_mask_input_device = self.Qformer_mask_proj.layers[0].weight.device
                dummy_mask_input_dtype = self.Qformer_mask_proj.layers[0].weight.dtype
                dummy_mask_input = torch.zeros(
                    1,
                    112 * 112,
                    device=dummy_mask_input_device,
                    dtype=dummy_mask_input_dtype
                )
                dummy_mask_input = self.Qformer_mask_proj(dummy_mask_input) * 1e-6
                n_token = selected.sum()
                if n_token != 0:
                    input_embeds[selected] = vit_embeds.reshape(-1, C) + dummy_mask_input + dummy_input
                else:
                    vit_embeds = vit_embeds.reshape(-1, C) + dummy_mask_input + dummy_input
                    input_embeds[selected] = vit_embeds[:n_token]
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
        else:
            try:
                input_embeds[selected] = torch.cat(vp_embeds, dim=0).reshape(-1, C)
            except Exception as e:
                vp_embeds = torch.cat(vp_embeds, dim=0).reshape(-1, C)
                print(f'warning: {e}, input_embeds[selected].shape='
                      f'{input_embeds[selected].shape}, '
                      f'vp_embeds.shape={vp_embeds.shape}')
                n_token = selected.sum()
                if n_token > len(vp_embeds):
                    print(f"Wrong !!! {n_token} image tokens in text but only {len(vp_embeds)} vit embeds !!!")
                    expand_ratio = n_token // len(vp_embeds) + 1
                    vp_embeds = torch.cat([vp_embeds] * expand_ratio, dim=0)

                input_embeds[selected] = vp_embeds[:n_token]
        if fast_vit_embeds is not None:
            selected = (input_ids == fast_token_idx)
            selected_tot = selected.sum().item()
            if selected_tot > fast_vit_embeds.shape[0] * fast_vit_embeds.shape[1]:
                assert selected_tot % (fast_vit_embeds.shape[0] * fast_vit_embeds.shape[1]) == 0
                repeat_times = selected_tot / (fast_vit_embeds.shape[0] * fast_vit_embeds.shape[1])
                fast_vit_embeds = fast_vit_embeds.repeat(int(repeat_times), 1, 1)
            try:
                input_embeds[selected] = fast_vit_embeds.reshape(-1, C)
            except Exception as e:
                fast_vit_embeds = fast_vit_embeds.reshape(-1, C)
                print(f'warning: {e}, input_embeds[fast_selected].shape='
                      f'{input_embeds[selected].shape}, '
                      f'fast_vit_embeds.shape={fast_vit_embeds.shape}')
                n_token = selected.sum()
                input_embeds[selected] = fast_vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.model.language_model(
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
                -1, self.model.language_model.config.vocab_size)
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
            fast_token_idx=None,
            fast_pixel_values=None,
            prompt_masks=None,
            vp_overall_mask=None,
            **generate_kwargs,
    ) -> torch.LongTensor:
        device = self.model.device
        assert self.model.img_context_token_id is not None

        if fast_pixel_values is not None:
            assert fast_token_idx is not None
            if type(fast_pixel_values) is list or fast_pixel_values.ndim == 5:
                if type(fast_pixel_values) is list:
                    fast_pixel_values = [
                        x.unsqueeze(0) if x.ndim == 3 else x for x in fast_pixel_values
                    ]
                # b*n, c, h, w
                fast_pixel_values = torch.cat(
                    [image.to(self.model.vision_model.dtype) for image in fast_pixel_values], dim=0)

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
                        [image.to(self.model.vision_model.dtype) for image in pixel_values], dim=0)

                if fast_pixel_values is not None:
                    n_fast_images = fast_pixel_values.shape[0]
                    whole_pixel_values = torch.cat([fast_pixel_values, pixel_values], dim=0)
                    vit_embeds = self.model.extract_feature(whole_pixel_values.to(device))
                    # vit_embeds = vit_embeds.to(input_embeds.dtype)  # FIXME: why vit_embeds is float16?
                    fast_vit_embeds = vit_embeds[:n_fast_images]  # (n_fast_images, hw, c)
                    _size = int(fast_vit_embeds.shape[1] ** 0.5)
                    fast_vit_embeds = fast_vit_embeds.reshape(fast_vit_embeds.shape[0], _size, _size,
                                                              fast_vit_embeds.shape[-1])
                    # pooling
                    fast_vit_embeds = fast_vit_embeds.permute(0, 3, 1, 2)  # (n_fast_images, c, h, w)
                    fast_vit_embeds = self.fast_pool(fast_vit_embeds).flatten(2)  # (n_fast_images, c, hw)
                    fast_vit_embeds = fast_vit_embeds.permute(0, 2, 1)
                    vit_embeds = vit_embeds[n_fast_images:]
                else:
                    fast_vit_embeds = None
                    vit_embeds = self.model.extract_feature(pixel_values.to(device))
            image_flags = torch.sum(pixel_values, dim=(1, 2, 3)) != 0
            image_flags = image_flags.long()
            vit_embeds = vit_embeds[image_flags == 1]

            input_embeds = self.model.language_model.get_input_embeddings()(input_ids.to(device))
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

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
            selected = (input_ids == self.model.img_context_token_id)
            assert selected.sum() != 0
            if vp_embeds is None:
                input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
            else:
                if len(input_embeds[selected]) != len(vp_embeds.reshape(-1, C)):
                    print("Shape mismatch, selected is {}, vp embeds is {} !!!" \
                          .format(len(input_embeds[selected]), len(vp_embeds.reshape(-1, C))))
                    min_tokens = min(len(input_embeds[selected]), len(vp_embeds.reshape(-1, C)))
                    input_embeds[selected][:min_tokens] = vp_embeds.reshape(-1, C)[:min_tokens].to(input_embeds.device)
                else:
                    input_embeds[selected] = vp_embeds.reshape(-1, C).to(input_embeds.device)

            if fast_vit_embeds is not None:
                selected = (input_ids == fast_token_idx)
                # FIXME, add repeat.
                assert selected.sum() != 0
                input_embeds[selected] = fast_vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.model.language_model.get_input_embeddings()(input_ids)

        outputs = self.model.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask.to(device),
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs

    def state_dict(self, *args, **kwargs):
        if self.transfer_to_hf:
            state_dict = super(InternVL_V1_5, self).state_dict(*args, **kwargs)
            return state_dict
        else:
            return super().state_dict(*args, **kwargs)


