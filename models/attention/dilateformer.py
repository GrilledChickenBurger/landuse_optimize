# DilateFormer: Multi-Scale Dilated Attention for Vision Transformers
# 论文: DilateFormer (Jiao et al., ICCV 2023, arXiv:2302.01791)
# 来源: 多尺度空洞注意力 (MSDA) 的机制。
# 差异: 官方为 ViT 骨干的多尺度设计; 本项目只抽取"逐 head 不同 dilation 的滑动窗口
#       空洞注意力"这一核心, 以纯 PyTorch 邻域 mask 实现, 无 CUDA 依赖。

import torch
import torch.nn as nn

from .neighborhood import NeighborhoodAttention


class DilateFormerAttention(nn.Module):
    """DilateFormer 的多尺度空洞注意力 (MSDA)。

    不同 head 使用不同 dilation 的滑动窗口空洞注意力, 使单层内同时捕捉局部细节与多尺度
    远距离依赖 (而非逐 head 统一感受野)。dilations 缺省按 [1,2,3,4,...] 循环分配给各 head。
    """

    def __init__(self, dim, num_heads, grid_size=0, attn_dropout=0.0, vis=False,
                 kernel_size=3, dilations=None, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.grid_size = grid_size
        self.kernel_size = kernel_size
        self.vis = vis

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(attn_dropout)

        if dilations is None:
            dilations = [(i % 4) + 1 for i in range(num_heads)]
        if len(dilations) != num_heads:
            raise ValueError(f"dilations 长度应等于 num_heads={num_heads}, 当前为 {len(dilations)}")

        if grid_size > 0:
            masks = torch.stack([
                NeighborhoodAttention._build_mask(grid_size, kernel_size, d)
                for d in dilations
            ])  # [h, N, N]
            self.register_buffer("attn_masks", masks, persistent=False)
        else:
            self.attn_masks = None

    def forward(self, hidden_states):
        B, N, C = hidden_states.shape
        qkv = self.qkv(hidden_states).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.attn_masks is not None:
            attn = attn.masked_fill(~self.attn_masks.unsqueeze(0), float("-inf"))
        weights = attn.softmax(dim=-1)
        attn = self.attn_drop(weights)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj_drop(self.proj(out))
        return out, (weights if self.vis else None)
