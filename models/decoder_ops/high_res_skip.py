"""高分辨率 skip 注入 (新增方案 3.1 / B1 轻量备选)。

在 ResNetV2.root 卷积之前导出一路 128×128 浅层特征, 经 1×1 对齐 16 通道后
作为 decoder 末级 skip; 配合 config.n_skip 3 -> 4 启用 skip_channels[3] = 16。

注意 (方案 5.5 风险): 该浅层特征来自原始 factors + 6 类 LU one-hot 的线性组合,
存在诱导模型直接复现 srcLU、退化为全判无变化的风险, 需以消融验证。

接口: forward(x[B, in_ch, H, W]) -> [B, out_ch, H, W]
"""

import torch.nn as nn


class HighResSkipProj(nn.Module):
    def __init__(self, in_ch, out_ch=16, use_batchnorm=True):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch) if use_batchnorm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.proj(x)))
