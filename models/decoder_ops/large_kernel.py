"""反瓶颈大核卷积块 (替换方案 3.2)。

落点: DecoderBlock.conv1 / conv2 (现为 Conv2dReLU)。
统一结构: 1×1 降通道 -> 大核深度卷积 (DW) + BN + ReLU -> 1×1 升通道。
conv1 为降通道卷积 (in+skip -> out), 深度卷积要求同通道, 故反瓶颈为强制前提;
conv2 为同通道 (out -> out), 首尾 1×1 退化为冗余同通道映射, 故采用简化形式
(直接 大核 DW + BN + ReLU + 1×1)。
核尺寸按 decoder 级分档 (见 LARGE_KERNEL_SCHEDULE), 控制 FLOPs 随分辨率平方增长。

接口: forward(x[B, in_channels, H, W]) -> [B, out_channels, H, W]
"""

import torch.nn as nn

# 按 decoder 级 (0..3, 上采样后空间 16 / 32 / 64 / 128) 的核尺寸分档:
# 低分辨率级 (16 / 32, block0 / block1) 用大核 13 / 17;
# 高分辨率级 (64 / 128, block2 / block3) 保持 5 / 7, 控制 FLOPs 随分辨率平方增长。
LARGE_KERNEL_SCHEDULE = (13, 17, 5, 7)


def resolve_large_kernel(level, default=7):
    """按 decoder 级索引取核尺寸; level 缺失或越界时回退 default。"""
    if level is None or not (0 <= level < len(LARGE_KERNEL_SCHEDULE)):
        return default
    return LARGE_KERNEL_SCHEDULE[level]


class LargeKernelBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=None, level=None,
                 use_batchnorm=True, **kwargs):
        super().__init__()
        if kernel is None:
            kernel = resolve_large_kernel(level)
        self.kernel = kernel
        self.same = (in_channels == out_channels)
        self.relu = nn.ReLU(inplace=True)
        # 大核深度卷积 (groups = out_channels), padding 保持同分辨率
        self.dw = nn.Conv2d(out_channels, out_channels, kernel,
                            padding=kernel // 2, groups=out_channels, bias=False)
        self.bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        # 1×1 升通道 / 投影
        self.proj = nn.Conv2d(out_channels, out_channels, 1, bias=not use_batchnorm)
        # 反瓶颈的 1×1 降通道; 同通道时省略 (简化形式)
        self.down = None if self.same else nn.Conv2d(
            in_channels, out_channels, 1, bias=not use_batchnorm)

    def forward(self, x):
        if self.down is not None:
            x = self.down(x)                    # 1×1 降通道 (满足 DW 同通道要求)
        x = self.relu(self.bn(self.dw(x)))      # 大核 DW + BN + ReLU
        x = self.proj(x)                        # 1×1 升通道 / 投影
        return x
