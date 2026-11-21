"""DyScaleC-TransUNet: 论文最终模型（固定结构，无可插拔开关）。

把三处改进固化成一个可直接阅读的实现:

  1. DySample 动态上采样   —— models/decoder_ops/dysample.py
     （反向使用确定性 grid_sample, 见 models/decoder_ops/det_grid_sample.py）
  2. 跨尺度融合 cross_scale —— models/decoder_ops/cross_scale.py
  3. 新 M1 交叉注意力       —— models/fusion/m1_input.py, 固定 src=hetero, pos=tok

为什么另起一个文件: models/transunet.py 为做消融保留了 up_type / fusion_type /
m1_src / m1_pos 等开关, 分支较多; 本文件把这些开关全部去掉, 只保留最终组合,
便于论文读者按一个文件读懂完整网络, 也避免误配置。

固定超参
--------
  输入        [B, 10, 128, 128]   前 4 通道 = 驱动因子, 后 6 通道 = 6 类 LU one-hot
  主干        R50-ViT-B/16 (preact, vanilla MHSA), 共 16 倍下采样
  新 M1       hetero / tok:
                - 因子 4 通道、LU 6 通道各自过独立 stem (真异构双流)
                - 各自下采样 4 倍后插值到 1/16, 展平为 token
                - F 作 query, L 作 key/value 做多头交叉注意力 (8 头)
                - 零初始化门控残差, 加回 ViT 的 1/16 token
  跨尺度融合  3 级 skip (1/8: 512ch, 1/4: 256ch, 1/2: 64ch)
                - 各级先 1×1 投到 256 维并展平
                - 以本级 token 为 Q、三级拼接为 K/V 做 4 头注意力
                - 零门控残差回写, 训练初期等价于原始 skip
  解码器      4 级: DySample(×2) -> 拼接同尺度 skip -> 2×Conv2dReLU
  输出        [B, 1, 128, 128] logits

前向数据流
----------
  x ─┬─ R50-ViT ─────────────────→ token [B, 64, 768] ─┐
     │                                                  ├─ (+) → 跨尺度融合 → Decoder×4 → Head → logits
     └─ M1(hetero, tok) ─────────→ delta [B, 768, 8, 8]─┘

  注意: M1 吃的是 x 的原始通道（未经主干），这正是"输入端驱动"的含义;
        而旧 M1 (models/fusion/m1.py) 吃的是编码器输出 token, 两者不同。

与可插拔版的等价性
------------------
子模块类型、参数形状与初始化方式和 models/transunet.py 在
  up_type="dysample", fusion_type="cross_scale", m1_src="hetero", m1_pos="tok"
下完全一致。子模块命名也刻意保持一致
  (vit.transformer / vit.m1_input / vit.decoder / vit.segmentation_head),
因此该配置训练得到的 checkpoint 可直接 load_state_dict 到本模型。
"""

import torch
import torch.nn as nn

from .decoder_ops import Conv2dReLU, DySampleUpsample
from .decoder_ops.cross_scale import CrossScaleFusion
from .fusion import InputFactorLUCrossAttn
from .transunet import SegmentationHead, Transformer, get_r50_b16_config

# ---- 固定结构常量（不是开关, 只是为了让下面代码可读） ----

_FACTOR_CH = 4                                   # 驱动因子通道数
_HIDDEN_SIZE = 768                               # ViT-B/16 的 token 维度
_M1_HEADS = 8                                    # 768 / 8 = 96, 每个头 96 维
_HEAD_CH = 512                                   # conv_more 输出通道
_DECODER_CHANNELS = (256, 128, 64, 16)           # 4 级 decoder 输出通道
_SKIP_CHANNELS = (512, 256, 64)                  # 编码器 1/8, 1/4, 1/2 三级 skip
_FUSION_DIM = 256                                # 跨尺度注意力的统一维度
_FUSION_HEADS = 4                                # 跨尺度注意力头数


class _DySampleDecoderBlock(nn.Module):
    """解码器单级: DySample 上采样 -> 拼接同尺度 skip -> 两次 3×3 卷积。

    参数
    ----
    in_ch   : 上一级输出通道数（第一级为 conv_more 的 512）
    out_ch  : 本级输出通道数
    skip_ch : 对应编码器 skip 的通道数；最后一级无 skip, 传 0

    形状变化（以第一级为例, 输入为 1/16 的 8×8）
    -------------------------------------------
    x [B, 512, 8, 8] --DySample×2--> [B, 512, 16, 16]
      --cat skip(1/8, 512ch)--> [B, 1024, 16, 16]
      --conv1--> [B, 256, 16, 16] --conv2--> [B, 256, 16, 16]
    """

    def __init__(self, in_ch, out_ch, skip_ch):
        super().__init__()
        # DySample: 从输入特征预测每个输出点的采样偏移, 再做双线性采样.
        # style="lp" 为官方默认; groups=4 要求 in_ch 能被 4 整除
        # (512 / 256 / 128 / 64 均满足).
        self.up = DySampleUpsample(in_channels=in_ch, scale=2, style="lp", groups=4)
        # 官方 TransUNet 的 Conv2d -> BatchNorm -> ReLU, 结构保持不变.
        self.conv1 = Conv2dReLU(
            in_ch + skip_ch, out_ch, kernel_size=3, padding=1, use_batchnorm=True
        )
        self.conv2 = Conv2dReLU(
            out_ch, out_ch, kernel_size=3, padding=1, use_batchnorm=True
        )

    def forward(self, x, skip=None):
        x = self.up(x)                              # 空间尺寸 ×2
        if skip is not None:
            x = torch.cat([x, skip], dim=1)         # 通道维拼接同尺度 skip
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class _FixedDecoder(nn.Module):
    """固定解码器: 先做跨尺度融合, 再逐级 DySample 上采样。

    输入
    ----
    hidden_states : [B, 64, 768]   ViT 输出的 1/16 token 序列（128/16=8, 8×8=64）
    features      : list[3], 依次为 1/8(512ch), 1/4(256ch), 1/2(64ch) 的编码器特征

    输出
    ----
    [B, 16, 128, 128]  最后一级 decoder 特征（再由分割头映射到类别数）

    说明: cross_scale 在送入各级 decoder 之前统一处理 skip 列表, 因此
          三级 skip 之间会先发生一次跨尺度信息交换, 之后仍按原来的
          同尺度 concat 方式进入各级 DecoderBlock.
    """

    def __init__(self):
        super().__init__()
        # 1/16 token 先 1×1 风格的 3×3 卷积压到 512 通道
        self.conv_more = Conv2dReLU(
            _HIDDEN_SIZE, _HEAD_CH, kernel_size=3, padding=1, use_batchnorm=True
        )

        # 4 级 decoder 的输入通道: [512, 256, 128, 64]
        # 第 i 级读取第 i 个 skip; 第 4 级 (i=3) 的 skip_ch=0, 不拼接.
        in_channels = (_HEAD_CH,) + _DECODER_CHANNELS[:-1]
        self.blocks = nn.ModuleList([
            _DySampleDecoderBlock(in_ch, out_ch, skip_ch)
            for in_ch, out_ch, skip_ch in zip(
                in_channels, _DECODER_CHANNELS, _SKIP_CHANNELS + (0,)
            )
        ])

        # 跨尺度融合: 3 级 skip -> 3 级 skip（原地改写列表）
        self.fusion = CrossScaleFusion(
            dim=_FUSION_DIM,
            num_heads=_FUSION_HEADS,
            levels=len(_DECODER_CHANNELS),
            channels=list(_SKIP_CHANNELS),
        )

    def forward(self, hidden_states, features):
        # token 序列还原成 2D 特征图: [B, 64, 768] -> [B, 768, 8, 8]
        b, n, c = hidden_states.shape
        h = w = int(n ** 0.5)
        x = hidden_states.permute(0, 2, 1).reshape(b, c, h, w).contiguous()
        x = self.conv_more(x)                       # [B, 512, 8, 8]

        # 三级 skip 先做跨尺度注意力（零门控, 训练初期为恒等映射）
        if features is not None:
            features = self.fusion(features)

        # 逐级上采样: 8→16→32→64→128
        for i, block in enumerate(self.blocks):
            skip = features[i] if (features is not None and i < len(features)) else None
            x = block(x, skip)
        return x


class _FixedVisionTransformer(nn.Module):
    """固定主干: R50-ViT 编码器 + 输入端 M1 + 固定解码器 + 分割头。

    子模块命名与 models/transunet.py 的 VisionTransformer 一致,
    以便直接加载 up_type=dysample + fusion_type=cross_scale +
    m1_src=hetero + m1_pos=tok 训练出的权重。
    """

    def __init__(self, in_channels=10, class_num=1, img_size=128):
        super().__init__()
        if in_channels <= _FACTOR_CH:
            raise ValueError(
                "in_channels 必须大于驱动因子通道数 %d（收到 %d）"
                % (_FACTOR_CH, in_channels)
            )
        if img_size % 16 != 0:
            raise ValueError("img_size 必须是 16 的整数倍（收到 %d）" % img_size)

        # ---- 编码器配置: 固定为官方 R50-ViT-B/16 ----
        config = get_r50_b16_config()
        config.n_classes = class_num
        # 16 倍下采样 -> token 网格边长 = img_size / 16
        config.patches.grid = (img_size // 16, img_size // 16)
        # 编码器注意力固定为官方 vanilla MHSA（不做 agent / swinv2 消融）
        config.attn_type = "vanilla"
        config.attn_kwargs = {}
        config.attn_grid_size = img_size // 16

        # R50 主干 + ViT: 返回 (1/16 token, 注意力权重, 三级 skip)
        # skip 顺序固定为 [1/8(512), 1/4(256), 1/2(64)]
        self.transformer = Transformer(
            config, img_size=img_size, vis=False, in_channels=in_channels
        )

        # ---- 新 M1: 输入端异构双流交叉注意力, 落点 1/16 token ----
        # 因子通道 = x[:, :4];  LU 通道 = x[:, 4:10]（6 类 one-hot）
        # stem 先各下采样 4 倍, 再插值到 1/16（即 8×8）, 与原模型完全一致.
        self.m1_input = InputFactorLUCrossAttn(
            factor_ch=_FACTOR_CH,
            lu_ch=in_channels - _FACTOR_CH,
            out_ch=_HIDDEN_SIZE,
            target_hw=(img_size // 16, img_size // 16),
            num_heads=_M1_HEADS,
            src="hetero",
            dropout=0.0,
        )

        # ---- 固定解码器 ----
        self.decoder = _FixedDecoder()

        # ---- 分割头: 3×3 卷积 + 1×1 输出 (官方结构, 无额外上采样) ----
        self.segmentation_head = SegmentationHead(
            in_channels=_DECODER_CHANNELS[-1],
            out_channels=class_num,
            kernel_size=3,
        )

    def forward(self, x):
        """x: [B, 10, 128, 128] -> logits [B, 1, 128, 128]."""
        # 保留原始输入: M1 必须从"最原始"的因子/LU 通道取特征,
        # 不能使用主干编码后的 token（这是与旧 M1 的关键区别）.
        x_raw = x

        # 主干: token [B, 64, 768]; skip [1/8, 1/4, 1/2]
        tokens, _, features = self.transformer(x)

        # M1: [B, 768, 8, 8] -> 展平回 token [B, 64, 768]
        # gate 零初始化, 训练初期 delta 为 0, 模型严格等价于无 M1 的基线.
        delta = self.m1_input(x_raw)
        tokens = tokens + delta.flatten(2).transpose(1, 2)

        # 解码: 跨尺度融合 + 4 级 DySample 上采样
        x = self.decoder(tokens, features)

        # 分割头: [B, 16, 128, 128] -> [B, 1, 128, 128]
        return self.segmentation_head(x)


class DyScaleCTransUNet(nn.Module):
    """论文最终模型 DyScaleC-TransUNet（固定结构, 不含任何可插拔开关）。

    用法
    ----
        model = DyScaleCTransUNet(in_channels=10, class_num=1, img_size=128)
        logits = model(x)          # x: [B, 10, 128, 128]

    参数
    ----
    in_channels : 输入通道数, 默认 10（前 4 驱动因子 + 后 6 LU one-hot）
    class_num   : 输出类别数, 二分类变化检测固定为 1
    img_size    : 输入边长, 默认 128（必须是 16 的整数倍）

    注意: 本类刻意不提供 up_type / fusion_type / m1_src / m1_pos 等参数,
          这三处改进已固定为最终方案; 需要做消融时请使用 models/transunet.py。
    """

    def __init__(self, in_channels=10, class_num=1, img_size=128):
        super().__init__()
        self.vit = _FixedVisionTransformer(
            in_channels=in_channels, class_num=class_num, img_size=img_size
        )

    def forward(self, x):
        return self.vit(x)


if __name__ == "__main__":
    # 形状自检: 128×128 输入, 10 通道, 二分类
    model = DyScaleCTransUNet(in_channels=10, class_num=1, img_size=128)
    print("参数量: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6))
    print("输出形状:", tuple(model(torch.randn(2, 10, 128, 128)).shape))
