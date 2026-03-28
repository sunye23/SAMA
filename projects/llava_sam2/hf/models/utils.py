from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from third_parts.mmdet.models.losses import CrossEntropyLoss


class CrossAttention(nn.Module):
    def __init__(self, q_dim=896, kv_dim=896, qkv_bias=False):
        super().__init__()
        self.scale = q_dim ** -0.5

        self.q_proj = nn.Linear(q_dim, q_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(kv_dim, q_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(kv_dim, q_dim, bias=qkv_bias)
        self.proj = nn.Linear(q_dim, q_dim)

    def forward(self, x_q, x_kv):          # todo: x_q shape: (32 * 5 * 256)   x_kv shape: (32 * 256 * 896)
        B_q, N_q, C_q = x_q.shape
        query = self.q_proj(x_q)
        key = self.k_proj(x_kv)
        value = self.v_proj(x_kv)
        attn = (query @ key.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)

        x = (attn @ value)

        x = self.proj(x) + x_q

        return x

class CustomizedSelfAttention(nn.Module):
    def __init__(self, q_dim=896, out_dim=896, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.scale = q_dim ** -0.5

        self.q_proj = nn.Linear(q_dim, q_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(q_dim, q_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(q_dim, q_dim, bias=qkv_bias)
        self.proj = nn.Linear(q_dim, q_dim)
        self.final_proj = nn.Sequential(
            nn.Linear(q_dim, q_dim), nn.ReLU(inplace=True),
            nn.Linear(q_dim, out_dim), nn.Dropout(0.0)
        )

    def forward(self, x_):          # todo: x_ shape: (5 * 32 * 256)
        B, N, C = x_.shape
        query = self.q_proj(x_)
        key = self.k_proj(x_)
        value = self.v_proj(x_)
        attn = (query @ key.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)

        x = (attn @ value)

        x = self.proj(x) + x_

        x = torch.mean(x, dim=1, keepdim=True)

        x = self.final_proj(x).squeeze(1)

        return x
