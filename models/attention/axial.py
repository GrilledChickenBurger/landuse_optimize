# Axial Attention in Multidimensional Transformers
# 论文: Axial Attention in Multidimensional Transformers (Ho et al., ECCV 2020, arXiv:1912.12180)
# 来源: 官方 axial attention 机制 (lucidrains 等实现)。
# 差异: 本项目只需 2D 图像上的轴向分解; 对 H×W token 网格先逐行 (宽轴) 再逐列 (高轴)
#       各做一次注意力, 两轴独立 QKV, 把 O((HW)^2) 降到 O(HW·(H+W))。

import torch
import torch.nn as nn


class AxialAttention(nn.Module):
    """轴向注意力 (Axial Attention)。

    把 2D 自注意力分解为沿宽轴 (逐行) 与沿高轴 (逐列) 两段, 两轴各用一套独立 QKV,
    避免全图注意力。要求 token 数恰为 grid_size^2, 便于还原成二维网格。
    """

    def __init__(self, dim, num_heads, grid_size=0, attn_dropout=0.0, vis=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.grid_size = grid_size
        self.vis = vis

        self.qkv_w = nn.Linear(dim, dim * 3)  # 沿宽轴 (逐行)
        self.qkv_h = nn.Linear(dim, dim * 3)  # 沿高轴 (逐列)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(attn_dropout)

    def _axis_attn(self, qkv_layer, x):
        # x: [B, g, g, C], 沿最后一维展开, 对每条长为 g 的序列独立做注意力
        B, g, _, C = x.shape
        x = x.reshape(B * g, g, C)
        qkv = qkv_layer(x).reshape(B * g, g, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B * g, g, C)
        return out.reshape(B, g, g, C)

    def forward(self, hidden_states):
        B, N, C = hidden_states.shape
        g = self.grid_size
        if g <= 0 or g * g != N:
            raise ValueError(f"axial 注意力要求 token 数为 grid_size^2, grid_size={g}, 但 N={N}")

        x = hidden_states.reshape(B, g, g, C)
        x = self._axis_attn(self.qkv_w, x)                        # 逐行
        x = self._axis_attn(self.qkv_h, x.permute(0, 2, 1, 3))    # 逐列 (转置后逐行)
        x = x.permute(0, 2, 1, 3).reshape(B, N, C)

        out = self.proj_drop(self.proj(x))
        # 轴向注意力的权重分两轴, 形状不统一, vis=True 时不返回权重
        return out, None
