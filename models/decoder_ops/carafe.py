"""CARAFE 内容感知上采样 (ICCV 2019, 新增方案 3.2)。

论文: CARAFE: Content-Aware ReAssembly of FEatures (arXiv:1905.02188)。
本实现为纯 PyTorch 版本 (不依赖 mmcv 的 CUDA 扩展), 重排用 unfold + gather 向量化完成。

落点: 末级 DecoderBlock 的 self.up, 与双线性支路并联相加 (不替换既有算子);
重排核的预测网络末端零初始化, 使初始重排核为均匀分布 (稳定启动)。

接口: forward(x[B, C, H, W]) -> [B, C, scale*H, scale*W]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CarafeUpsample(nn.Module):
    def __init__(self, in_ch, scale=2, kernel_size=3, up_kernel=3, c_mid=None, **kwargs):
        super().__init__()
        self.scale = scale
        self.k_up = up_kernel
        c_mid = c_mid or max(in_ch // 4, 16)
        self.comp = nn.Conv2d(in_ch, c_mid, 1)
        self.enc = nn.Conv2d(c_mid, (scale ** 2) * (up_kernel ** 2), kernel_size,
                             padding=kernel_size // 2)
        # 零初始化 -> softmax 后为均匀重排核 (等价局部均值上采样), 保证训练稳定
        nn.init.zeros_(self.enc.weight)
        nn.init.zeros_(self.enc.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        s, k = self.scale, self.k_up

        # 1) 预测重排核: [B, s^2*k^2, H, W] -> pixel_shuffle -> [B, k^2, sH, sW] -> softmax
        logits = F.pixel_shuffle(self.enc(self.comp(x)), s)
        kernel = F.softmax(logits, dim=1)

        # 2) 取源特征每个位置的 k×k 邻域
        patches = F.unfold(x, k, padding=k // 2)          # [B, C*k^2, H*W]
        patches = patches.view(B, C, k * k, H * W)

        # 3) 输出位置 (i, j) 的邻域中心对应源位置 (i // s, j // s)
        sH, sW = H * s, W * s
        rows = torch.arange(sH, device=x.device).view(sH, 1) // s
        cols = torch.arange(sW, device=x.device).view(1, sW) // s
        src = (rows * W + cols).reshape(-1)               # [sH*sW]

        patches = patches.gather(3, src.view(1, 1, 1, -1).expand(B, C, k * k, -1))
        kernel = kernel.view(B, k * k, -1).gather(2, src.view(1, 1, -1).expand(B, k * k, -1))

        # 4) 按预测核加权重组
        out = (patches * kernel.unsqueeze(1)).sum(dim=2)  # [B, C, sH*sW]
        return out.view(B, C, sH, sW)


class CarafeParallelUpsample(nn.Module):
    """新增 3.2 的落点形式: 双线性支路 ⊕ CARAFE 支路 (仅末级并联, 不削减原算子)。"""

    def __init__(self, in_ch, scale=2, **kwargs):
        super().__init__()
        self.bilinear = nn.UpsamplingBilinear2d(scale_factor=scale)
        self.carafe = CarafeUpsample(in_ch, scale=scale, **kwargs)

    def forward(self, x):
        return self.bilinear(x) + self.carafe(x)
