# Swin Transformer V2, 2021 (CVPR 2022), arXiv:2111.09883
# 来源: microsoft/Swin-Transformer 官方实现的 WindowAttention 核心机制。
# 相对标准点积注意力的两点改动:
#   1) 缩放余弦注意力: 相似度改为 cos(q, k) * logit_scale, logit_scale 逐头可学习,
#      并截断上界 log(1/0.01), 对应论文中把注意力权重下界控制在 0.01;
#   2) Log-CPB: 相对位置偏置由相对坐标 (dh, dw) 经两层的 MLP 连续生成,
#      而非 V1 的离散可学习位置偏置表。

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledCosineAttention(nn.Module):
    """Swin Transformer V2 的缩放余弦注意力 + 连续相对位置偏置 (Log-CPB)。

    要求 token 数恰好为 grid_size^2, 相对位置索引按 grid_size x grid_size 的网格预计算。
    """

    def __init__(self, dim, num_heads, grid_size=0, attn_dropout=0.0, vis=False,
                 cpb_hidden=512, **kwargs):
        super().__init__()
        if grid_size < 1:
            raise ValueError(f"swinv2 注意力需要 grid_size >= 1, 当前为 {grid_size}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.grid_size = grid_size
        self.vis = vis

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        # 逐头可学习的对数尺度, 截断上界 log(1/0.01)
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        # 由相对坐标生成逐头位置偏置的 MLP
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, cpb_hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(cpb_hidden, num_heads, bias=False),
        )
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(attn_dropout)

        # 常量表, 不进 state_dict
        self.register_buffer("relative_position_index",
                             self._build_relative_position_index(), persistent=False)
        self.register_buffer("relative_coords_table",
                             self._build_relative_coords_table(), persistent=False)

    def _build_relative_position_index(self):
        """[N, N] 的相对位置索引, N = grid_size^2, 取值落在 [0, (2g-1)^2)。"""
        g = self.grid_size
        coords = torch.stack(torch.meshgrid(torch.arange(g), torch.arange(g),
                                            indexing="ij"))            # [2, g, g]
        coords_flatten = torch.flatten(coords, 1)                      # [2, N]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, N, N]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()            # [N, N, 2]
        relative_coords[:, :, 0] += g - 1
        relative_coords[:, :, 1] += g - 1
        relative_coords[:, :, 0] *= 2 * g - 1
        return relative_coords.sum(-1)                                 # [N, N]

    def _build_relative_coords_table(self):
        """[(2g-1)^2, 2] 的归一化相对坐标表 (Log-CPB 的输入)。"""
        g = self.grid_size
        coords_h = torch.arange(-(g - 1), g, dtype=torch.float32)
        coords_w = torch.arange(-(g - 1), g, dtype=torch.float32)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))    # [2, 2g-1, 2g-1]
        table = coords.permute(1, 2, 0).contiguous().unsqueeze(0)                  # [1, 2g-1, 2g-1, 2]
        table[:, :, :, 0] /= (g - 1)
        table[:, :, :, 1] /= (g - 1)
        table *= 8  # 归一化到 [-8, 8]
        table = torch.sign(table) * torch.log2(torch.abs(table) + 1.0) / np.log2(8.0)
        return table.reshape(-1, 2)                                    # [(2g-1)^2, 2]

    def forward(self, hidden_states):
        B, N, C = hidden_states.shape
        if N != self.grid_size * self.grid_size:
            raise ValueError(
                f"swinv2 注意力要求 token 数为 grid_size^2 = {self.grid_size ** 2}, 当前为 {N}"
            )

        qkv = self.qkv(hidden_states).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)            # [B, h, N, d]

        # Log-CPB: 相对位置偏置 [1, h, N, N]
        bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        bias = bias_table[self.relative_position_index.reshape(-1)].view(N, N, self.num_heads)
        bias = (16 * torch.sigmoid(bias)).permute(2, 0, 1).contiguous().unsqueeze(0)

        # 缩放余弦注意力
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()
        attn = (q @ k.transpose(-2, -1)) * logit_scale + bias

        weights = attn.softmax(dim=-1)
        attn = self.attn_dropout(weights)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj_dropout(self.proj(out))
        return out, (weights if self.vis else None)
