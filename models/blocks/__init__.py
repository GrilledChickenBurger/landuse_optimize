"""R50 主干 block 族 (族三)。

经 resblock_type 开关替换 ResNetV2.body 的阶段单元 (block1–block3)。
默认项 preact 与现状完全等价, 保持默认时不改变默认模型的 state_dict, 历史 checkpoint 可继续加载。
接口: forward(x[B, cin, H, W]) -> [B, cout, H//stride, W//stride]
"""

import torch.nn as nn

from .preact import PreActBottleneck, StdConv2d, conv3x3, conv1x1
from .convnext_block import ConvNeXtBlock

# resblock_type -> 实现类
BLOCK_REGISTRY = {
    "preact": PreActBottleneck,      # 默认, 现状不变
    "convnext": ConvNeXtBlock,       # 替换 3.3
}


def build_block(resblock_type: str, cin: int, cout: int, cmid=None, stride: int = 1,
                **kwargs) -> nn.Module:
    """按 resblock_type 构造主干阶段单元。

    preact   : PreActBottleneck(cin, cout, cmid, stride) —— cmid 为其瓶颈通道
    convnext : ConvNeXtBlock(dim=cout, in_dim=cin, stride=stride) —— cmid 不适用
    """
    if resblock_type not in BLOCK_REGISTRY:
        raise ValueError(
            f"不支持的 resblock_type: {resblock_type!r}, 可选值为 {sorted(BLOCK_REGISTRY)}"
        )
    cls = BLOCK_REGISTRY[resblock_type]
    if resblock_type == "convnext":
        return cls(dim=cout, in_dim=cin, stride=stride, **kwargs)
    return cls(cin, cout, cmid=cmid, stride=stride, **kwargs)
