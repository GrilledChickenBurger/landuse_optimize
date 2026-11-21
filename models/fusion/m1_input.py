"""M1 变体: 输入端驱动的 因子 x LU 交叉注意力 (新增 3.5.1 消融).

与 models/fusion/m1.py (瓶颈同源版) 的差别有两点, 用于两个小消融:

1. 分支来源 src
   - "same"  : F、L 均由同一路融合输入得到 (复刻"同源"语义, 用于隔离位置效应);
   - "hetero": F 只吃因子通道 x[:, :factor_ch], L 只吃 LU 通道 [factor_ch:],
               两路各自独立编码 -- 真正的异构双流交互.
2. 接入位置 pos (由 VisionTransformer 解析为目标通道/分辨率)
   - "tok" : 1/16 瓶颈 token (ViT 输出, 与现有 m1 同位置);
   - "p8"  : 1/8  特征图 (features[0]);
   - "p4"  : 1/4  特征图 (features[1]);
   - "p2"  : 1/2  特征图 (features[2]).

实现: 从原始输入拆流 -> 各自卷积下采样到目标分辨率 -> 展平为 token ->
交叉注意力 (F 为 query, L 为 key/value) -> 门控残差叠加.
门控 gate 初始化为 0, 训练初期严格为恒等映射(等价原模型), 可从既有起点稳定学习.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _pick_groups(ch, max_groups=32):
    """给通道数挑一个可整除的 GroupNorm 组数."""
    for g in range(min(max_groups, ch), 0, -1):
        if ch % g == 0:
            return g
    return 1


class InputFactorLUCrossAttn(nn.Module):
    """输入端 因子 x LU 交叉注意力 (支持 同源/异构 x 多位置)."""

    def __init__(self, factor_ch, lu_ch, out_ch, target_hw, num_heads,
                 src="hetero", dropout=0.0):
        super().__init__()
        if src not in ("same", "hetero"):
            raise ValueError("src 仅支持 'same' / 'hetero', 收到 %r" % (src,))
        if out_ch % num_heads != 0:
            raise ValueError("out_ch (%d) 必须能被 num_heads (%d) 整除" % (out_ch, num_heads))
        self.src = src
        self.factor_ch = int(factor_ch)
        self.lu_ch = int(lu_ch)
        self.out_ch = int(out_ch)
        self.target_hw = tuple(target_hw)
        groups = _pick_groups(out_ch)

        def _stem(in_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(groups, out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(groups, out_ch), nn.ReLU(inplace=True),
            )

        if src == "same":
            self.stem_f = _stem(self.factor_ch + self.lu_ch)
            self.stem_l = None
        else:
            self.stem_f = _stem(self.factor_ch)
            self.stem_l = _stem(self.lu_ch)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=out_ch, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        # 可学习门控, 初始化为 0 -> 训练初期恒等映射
        self.gate = nn.Parameter(torch.zeros(1, out_ch, 1, 1))

    def forward(self, x_in):
        """x_in: [B, factor_ch + lu_ch, H, W] -> 门控残差 [B, out_ch, th, tw]."""
        if self.src == "same":
            f = self.stem_f(x_in)
            l = f
        else:
            f = self.stem_f(x_in[:, :self.factor_ch])
            l = self.stem_l(x_in[:, self.factor_ch:self.factor_ch + self.lu_ch])
        if f.shape[-2:] != self.target_hw:
            f = F.interpolate(f, size=self.target_hw, mode="bilinear", align_corners=False)
            l = F.interpolate(l, size=self.target_hw, mode="bilinear", align_corners=False)
        b, c, h, w = f.shape
        f_t = f.flatten(2).transpose(1, 2)
        l_t = l.flatten(2).transpose(1, 2)
        attn_out, _ = self.cross_attn(f_t, l_t, l_t)
        attn_out = attn_out.transpose(1, 2).reshape(b, c, h, w)
        return self.gate * attn_out
