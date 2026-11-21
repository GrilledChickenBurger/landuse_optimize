"""跨尺度融合 (新增方案 3.3, 方向一正解)。

落点: DecoderCup.forward 中送入 decoder 之前的编码器 skip 特征列表。
各级特征经 1×1 投影到统一 dim 并展平为 token; 以各级自身 token 为 Q、
其余各级 token 拼接为 K/V 做多头跨尺度注意力, 再按原分辨率还原并残差回写。

门控 gamma 零初始化 -> 训练初期为恒等映射 (严格等价于 concat 默认路径)。

接口: forward(list[[B, C_i, H_i, W_i]]) -> list[[B, C_i, H_i, W_i]]
"""

import torch
import torch.nn as nn


class CrossScaleFusion(nn.Module):
    def __init__(self, dim=256, num_heads=4, levels=4, channels=None, dropout=0.0, **kwargs):
        super().__init__()
        self.dim = dim
        self.levels = levels
        channels = list(channels) if channels else []
        self.proj_in = nn.ModuleList([nn.Conv2d(c, dim, 1) for c in channels])
        self.proj_out = nn.ModuleList([nn.Conv2d(dim, c, 1) for c in channels])
        self.attn = nn.ModuleList([
            nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
            for _ in channels
        ])
        self.norm = nn.ModuleList([nn.GroupNorm(1, dim) for _ in channels])
        # 零初始化门控: 恒等启动
        self.gamma = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in channels])

    def forward(self, features):
        if len(self.proj_in) == 0 or len(features) != len(self.proj_in):
            return features

        dim = self.dim
        # 1) 统一投影到 dim 并展平为 token
        tokens = []
        for proj, feat in zip(self.proj_in, features):
            tokens.append(proj(feat).flatten(2).transpose(1, 2))   # [B, N_i, dim]
        kv = torch.cat(tokens, dim=1)                              # [B, sum N, dim]

        # 2) 各级以自身为 Q、跨级 token 为 K/V 做注意力
        outs = []
        for i, q in enumerate(tokens):
            a, _ = self.attn[i](q, kv, kv)                         # [B, N_i, dim]
            a = self.norm[i](a.transpose(1, 2)).transpose(1, 2)
            h, w = features[i].shape[-2:]
            a = a.transpose(1, 2).reshape(q.shape[0], dim, h, w)
            outs.append(features[i] + self.gamma[i] * self.proj_out[i](a))
        return outs
