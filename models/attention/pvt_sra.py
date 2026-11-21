# Pyramid Vision Transformer (PVT)
# 论文: Pyramid Vision Transformer (Wang et al., ICCV 2021, arXiv:2102.12122)
# 来源: whai362/PVT 官方实现中的 Spatial-Reduction Attention (SRA) 机制。
# 差异: 本项目 TransUNet 的 ViT 是单尺度, 故只抽取 SRA 模块本身 (Q 全分辨率、K/V 空间缩减),
#       不引入金字塔多尺度结构。

import torch
import torch.nn as nn


class SpatialReductionAttention(nn.Module):
    """空间缩减注意力 (SRA, PVT)。

    Q 保持全分辨率, K/V 先经 sr_ratio×sr_ratio 的卷积下采样缩减 token 数, 再算注意力,
    从而把注意力复杂度从 O(N^2) 降到 O(N^2 / sr_ratio^2)。sr_ratio=1 (或 grid 太小无法缩减)
    时退化为标准全局注意力。
    """

    def __init__(self, dim, num_heads, grid_size=0, attn_dropout=0.0, vis=False,
                 sr_ratio=2, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.grid_size = grid_size
        self.vis = vis

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(attn_dropout)

        # 只有真正能缩减时才启用空间缩减
        if sr_ratio > 1 and grid_size > 0 and sr_ratio < grid_size:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None

    def forward(self, hidden_states):
        B, N, C = hidden_states.shape
        q = self.q(hidden_states).reshape(B, N, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)

        if self.sr is not None:
            g = self.grid_size
            x = hidden_states.transpose(1, 2).reshape(B, C, g, g)
            x = self.sr(x).flatten(2).transpose(1, 2)  # [B, N', C]
            x = self.norm(x)
        else:
            x = hidden_states

        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        weights = attn.softmax(dim=-1)
        attn = self.attn_drop(weights)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj_drop(self.proj(out))
        return out, (weights if self.vis else None)
