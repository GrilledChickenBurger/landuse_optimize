"""并联局部算子块 (新增方案 3.4)。

落点: DecoderBlock.conv1 / conv2, 在既有 3×3 卷积旁并联一条大核局部支路, 输出相加
(不削减原卷积)。大核支路复用替换 3.2 的 LargeKernelBlock, 核尺寸分档一致 (13/17/5/7)。
与替换 3.2 属同一条路线: 先做替换 3.2 (替换既有卷积), 本项为后续扩展。

接口: forward(x[B, in_channels, H, W]) -> [B, out_channels, H, W]
"""

import torch.nn as nn

from .conv2drelu import Conv2dReLU
from .large_kernel import LargeKernelBlock


class ParallelLocalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=None, level=None,
                 use_batchnorm=True, **kwargs):
        super().__init__()
        # 原 3×3 卷积 (保留, 不替换)
        self.conv = Conv2dReLU(in_channels, out_channels, kernel_size=3, padding=1,
                               use_batchnorm=use_batchnorm)
        # 并联的大核局部支路
        self.local = LargeKernelBlock(in_channels, out_channels, kernel=kernel,
                                      level=level, use_batchnorm=use_batchnorm)

    def forward(self, x):
        return self.conv(x) + self.local(x)
