"""
TransUNet 解码器的可插拔算子库。

与编码器注意力族 (models/attention/, token 接口 [B, N, C]) 并列, 本族接口为 2D 特征图 [B, C, H, W]。
通过 config 开关选择替换 decoder / 主干中的既有算子; 默认项与官方 TransUNet 完全等价,
保持默认时不影响既有模型, 历史 checkpoint 可继续加载 (默认项 state_dict 不变)。

三类开关:
  up_type     - 上采样算子:     bilinear(默认) | dysample | carafe(仅末级并联)
  conv_type   - decoder 卷积块: conv2drelu(默认) | large_kernel | parallel_local
  fusion_type - 多分辨率融合:   concat(默认)   | cross_scale

统一接口:
  上采样   forward(x[B, C, H, W]) -> [B, C, scale*H, scale*W]
  卷积块   forward(x[B, C, H, W]) -> [B, C, H, W]
  融合     forward(list[[B, C, H, W]]) -> list[[B, C, H, W]]
"""

import torch.nn as nn

from .bilinear import BilinearUpsample
from .dysample import DySampleUpsample
from .carafe import CarafeParallelUpsample
from .conv2drelu import Conv2dReLU
from .large_kernel import LargeKernelBlock
from .parallel_local import ParallelLocalBlock
from .concat import ConcatFusion
from .cross_scale import CrossScaleFusion
from .high_res_skip import HighResSkipProj

# up_type -> 实现类
UPSAMPLE_REGISTRY = {
    "bilinear": BilinearUpsample,          # 默认, 等价 nn.UpsamplingBilinear2d(scale_factor=2)
    "dysample": DySampleUpsample,          # 替换 3.1
    "carafe": CarafeParallelUpsample,      # 新增 3.2 (bilinear ⊕ CARAFE; 仅末级并联, 落点由 DecoderCup 限定)
}

# conv_type -> 实现类
CONV_REGISTRY = {
    "conv2drelu": Conv2dReLU,              # 默认, 现状不变
    "large_kernel": LargeKernelBlock,      # 替换 3.2 (核尺寸按 decoder 级分档)
    "parallel_local": ParallelLocalBlock,  # 新增 3.4 (原 3×3 卷积 ⊕ 大核局部支路)
}

# fusion_type -> 实现类
FUSION_REGISTRY = {
    "concat": ConcatFusion,                # 默认, 保持逐级 skip 融合, 不引入跨级交互
    "cross_scale": CrossScaleFusion,       # 新增 3.3 (跨级多头注意力, 零门控恒等启动)
}


def build_upsample(up_type: str, in_ch: int, scale: int = 2, **kwargs) -> nn.Module:
    """按 up_type 构造上采样算子。

    接口: forward(x[B, C, H, W]) -> [B, C, scale*H, scale*W]
    """
    if up_type not in UPSAMPLE_REGISTRY:
        raise ValueError(
            f"不支持的 up_type: {up_type!r}, 可选值为 {sorted(UPSAMPLE_REGISTRY)}"
        )
    return UPSAMPLE_REGISTRY[up_type](in_ch, scale=scale, **kwargs)


def build_conv(conv_type: str, in_ch: int, out_ch: int, **kwargs) -> nn.Module:
    """按 conv_type 构造 decoder 卷积块。

    接口: forward(x[B, in_ch, H, W]) -> [B, out_ch, H, W]
    kwargs 原样透传给实现类; level (decoder 级索引) 供 large_kernel 选取核尺寸,
    其余实现类通过 **kwargs 忽略未知参数。
    """
    if conv_type not in CONV_REGISTRY:
        raise ValueError(
            f"不支持的 conv_type: {conv_type!r}, 可选值为 {sorted(CONV_REGISTRY)}"
        )
    return CONV_REGISTRY[conv_type](in_ch, out_ch, **kwargs)


def build_fusion(fusion_type: str, dim: int = 256, num_heads: int = 4,
                 levels: int = 4, channels=None, **kwargs) -> nn.Module:
    """按 fusion_type 构造多分辨率融合算子。

    接口: forward(list[[B, C, H, W]]) -> list[[B, C, H, W]]
    channels 为各级通道数 (cross_scale 据此建立 1×1 投影), concat 忽略之。
    """
    if fusion_type not in FUSION_REGISTRY:
        raise ValueError(
            f"不支持的 fusion_type: {fusion_type!r}, 可选值为 {sorted(FUSION_REGISTRY)}"
        )
    return FUSION_REGISTRY[fusion_type](dim=dim, num_heads=num_heads, levels=levels,
                                        channels=channels, **kwargs)
