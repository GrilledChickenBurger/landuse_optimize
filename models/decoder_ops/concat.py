"""默认融合算子: 保持现有逐级 skip 融合, 不引入跨级交互。

现状 (官方 TransUNet) 的 decoder 只在 DecoderBlock 内部做同尺度 skip 拼接 (torch.cat),
各级之间无交互; 本类即该默认路径的占位实现, 原样返回特征列表, 保证默认行为与现状完全一致。
跨尺度交互由后续新增 3.3 的 cross_scale 提供。

接口: forward(features: list[[B, C, H, W]]) -> list[[B, C, H, W]]
"""

import torch.nn as nn


class ConcatFusion(nn.Module):
    def __init__(self, dim=0, num_heads=0, levels=4, channels=None, **kwargs):
        super().__init__()
        self.levels = levels

    def forward(self, features):
        return features
