"""ConvNeXt block (替换方案 3.3), 替换 ResNetV2.body 的 PreActBottleneck。

结构: [可选 stem 下采样 / 通道对齐] -> 7×7 深度卷积 -> LayerNorm
      -> 1×1 升维 (×4) -> GELU -> 1×1 还原 -> layer-scale 缩放, 与 stem 输出残差相加。
当 in_dim != dim 或 stride != 1 时, stem 为一条 Conv2d 支路 (3×3/stride=2/pad=1 或 1×1),
对应原 bottleneck 的 downsample 与 stage 起始单元的 stride=2, 从而保持 ResNetV2.forward
的 features 空间尺寸与 skip 通道布局不变。

接口: forward(x[B, in_dim, H, W]) -> [B, dim, H//stride, W//stride]
"""

import torch
import torch.nn as nn


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, in_dim=None, stride=1, layer_scale_init=1e-6, **kwargs):
        super().__init__()
        in_dim = in_dim or dim

        # stage 下采样 / 通道对齐 (对应原 bottleneck 的 stride=2 与 downsample)
        if stride != 1 or in_dim != dim:
            if stride == 2:
                # 3×3 stride=2 padding=1: 与原 conv3x3(s2) 主路径的空间尺寸完全一致
                # (注意不能用 2×2 stride=2, 奇数输入如 31 会得到 15 而非 16)
                self.stem = nn.Sequential(
                    nn.Conv2d(in_dim, dim, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(1, dim),
                )
            else:
                # 仅通道对齐
                self.stem = nn.Sequential(
                    nn.Conv2d(in_dim, dim, kernel_size=1, bias=False),
                    nn.GroupNorm(1, dim),
                )
        else:
            self.stem = None

        self.dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pw1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(dim * 4, dim)
        # layer-scale (默认 1e-6), 与官方 ConvNeXt 一致
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim)) if layer_scale_init else None

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        residual = x

        y = self.dw(x)                              # 7×7 深度卷积
        y = y.permute(0, 2, 3, 1)                   # [B, H, W, C]
        y = self.norm(y)                            # LayerNorm (channels-last)
        y = self.pw2(self.act(self.pw1(y)))         # 1×1 ↑×4 -> GELU -> 1×1 ↓
        y = y.permute(0, 3, 1, 2)                   # [B, C, H, W]
        if self.gamma is not None:
            y = y * self.gamma[None, :, None, None]
        return residual + y
