"""双线性上采样算子 (默认)。

等价官方 TransUNet DecoderBlock.up 的 nn.UpsamplingBilinear2d(scale_factor=2), 无参数,
不改变默认模型的 state_dict。
"""

import torch.nn as nn


class BilinearUpsample(nn.Module):
    """接口: forward(x[B, C, H, W]) -> [B, C, scale*H, scale*W]"""

    def __init__(self, in_ch=None, scale=2, **kwargs):
        super().__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale)

    def forward(self, x):
        return self.up(x)
