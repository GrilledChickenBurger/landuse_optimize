"""decoder 默认卷积块 (官方 TransUNet 的 Conv2dReLU)。

自 models/transunet.py 原样迁移至此, 模块结构与参数命名不变, 保证默认模型的 state_dict
与历史 checkpoint 完全兼容。
"""

import torch.nn as nn


class Conv2dReLU(nn.Sequential):
    """Conv2d -> BatchNorm2d -> ReLU, 接口 forward(x[B, C, H, W]) -> [B, out_channels, H, W]

    **kwargs 用于吞掉 factory 透传的其他算子专属参数 (如 large_kernel 的 level), 不影响本块结构。
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1,
                 use_batchnorm=True, **kwargs):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)
        super(Conv2dReLU, self).__init__(conv, bn, relu)
