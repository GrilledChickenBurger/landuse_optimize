# Neighborhood Attention Transformer (NAT)
# 论文: Neighborhood Attention Transformer (Hassani et al., CVPR 2023, arXiv:2204.07143)
# 来源: SHI-Labs/Neighborhood-Attention-Transformer 的机制。
# 差异: 官方用 natten 库 (CUDA kernel), 本项目为避免编译依赖, 改用纯 PyTorch 邻域 mask
#       实现 (注意力分数先算全图 O(N^2), 再用邻域 mask 屏蔽非邻域 token)。
#       token 数仅 grid_size^2 (本项目 64), O(N^2) 代价可忽略。

import torch
import torch.nn as nn


class NeighborhoodAttention(nn.Module):
    """逐像素滑动窗口邻域注意力。

    每个 token 只对以自身为中心的 kernel_size×kernel_size 邻域做注意力 (而非 Swin 的
    固定非重叠窗口), 兼顾局部归纳偏置与跨窗口的信息连续。dilation=1 时为标准邻域注意力,
    dilation>1 时退化为空洞邻域 (供 DiNAT 复用)。
    """

    def __init__(self, dim, num_heads, grid_size=0, attn_dropout=0.0, vis=False,
                 kernel_size=3, dilation=1, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.grid_size = grid_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.vis = vis

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(attn_dropout)

        if grid_size > 0:
            mask = self._build_mask(grid_size, kernel_size, dilation)
            self.register_buffer("attn_mask", mask, persistent=False)
        else:
            self.attn_mask = None

    @staticmethod
    def _build_mask(grid_size, kernel_size, dilation):
        """[N, N] 布尔 mask, N = grid_size^2; mask[i, j]=True 表示 j 在 i 的邻域内。"""
        half = kernel_size // 2
        offsets = torch.arange(-half * dilation, half * dilation + 1, dilation)
        g = grid_size
        mask = torch.zeros(g * g, g * g, dtype=torch.bool)
        for i in range(g * g):
            ri, ci = divmod(i, g)
            for dr in offsets.tolist():
                rr = ri + dr
                if rr < 0 or rr >= g:
                    continue
                for dc in offsets.tolist():
                    cc = ci + dc
                    if cc < 0 or cc >= g:
                        continue
                    mask[i, rr * g + cc] = True
        return mask

    def forward(self, hidden_states):
        B, N, C = hidden_states.shape
        qkv = self.qkv(hidden_states).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.attn_mask is not None:
            attn = attn.masked_fill(~self.attn_mask, float("-inf"))
        weights = attn.softmax(dim=-1)
        attn = self.attn_drop(weights)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj_drop(self.proj(out))
        return out, (weights if self.vis else None)
