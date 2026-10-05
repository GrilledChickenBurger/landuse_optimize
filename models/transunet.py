# github:
# https://github.com/Beckschen/TransUNet  (官方实现)
# 论文: TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation (Chen et al., 2021, arXiv:2102.04306)
# 来源: 官方仓库 Beckschen/TransUNet (论文作者开源), 前向结构完全对齐官方 R50-ViT-B_16.
# 差异: 去掉 load_from 预训练权重 (从头训练); img_size=128 (官方 224); n_classes=1 (二分类).
# 参数量: ~105.2M (in_channels=10, class_num=1, img_size=128)

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Dropout, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair

from .attention import build_attention
from .blocks import build_block, StdConv2d
from .decoder_ops import build_upsample, build_conv, build_fusion, Conv2dReLU, HighResSkipProj
from .fusion import FactorLUCroAttention


class ConfigDict(dict):
    """ml_collections.ConfigDict 的极简替代, 支持属性访问."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


def get_r50_b16_config():
    """官方 R50-ViT-B_16 配置 (config.patches.grid 由调用方按 img_size 覆盖)."""
    config = ConfigDict()
    config.patches = ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'seg'
    config.patches.grid = (14, 14)
    config.resnet = ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 1
    config.n_skip = 3
    config.activation = 'softmax'
    return config


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


# ---- ResNetV2 (hybrid encoder backbone) ----

# StdConv2d / conv3x3 / conv1x1 / PreActBottleneck 已迁至 models/blocks/preact.py (族三)


class ResNetV2(nn.Module):
    """Pre-activation (v2) ResNet, 16 倍下采样.

    block_type 为主干 block 开关 (preact 默认 / convnext), 见 models/blocks/。
    """
    def __init__(self, block_units, width_factor, in_channels=3, block_type="preact",
                 high_res_skip=False):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(in_channels, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
        ]))

        # 新增 3.1 / B1: 在 root 之前导出一路原分辨率浅层特征, 1×1 对齐 16 通道作末级 skip
        self.high_res = HighResSkipProj(in_channels, 16) if high_res_skip else None

        # 阶段单元由 block_type 开关决定 (见 models/blocks/), 默认 preact 与现状一致
        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', build_block(block_type, width, width * 4, width))] +
                [(f'unit{i:d}', build_block(block_type, width * 4, width * 4, width)) for i in range(2, block_units[0] + 1)],
            ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', build_block(block_type, width * 4, width * 8, width * 2, stride=2))] +
                [(f'unit{i:d}', build_block(block_type, width * 8, width * 8, width * 2)) for i in range(2, block_units[1] + 1)],
            ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', build_block(block_type, width * 8, width * 16, width * 4, stride=2))] +
                [(f'unit{i:d}', build_block(block_type, width * 16, width * 16, width * 4)) for i in range(2, block_units[2] + 1)],
            ))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        high_res = self.high_res(x) if self.high_res is not None else None
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body) - 1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i + 1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        features = features[::-1]
        if high_res is not None:
            # 新增 3.1 / B1: 末级 skip 追加在列表尾部 (对应 skip_channels[3] = 16)
            features.append(high_res)
        return x, features


# ---- ViT (transformer encoder) ----

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings."""
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:  # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
                                         width_factor=config.resnet.width_factor,
                                         in_channels=in_channels,
                                         block_type=getattr(config, "resblock_type", "preact"),
                                         high_res_skip=getattr(config, "high_res_skip", False))
            in_channels = self.hybrid_model.width * 16

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        # 注意力模块由 attn_type 配置开关决定, 见 models/attention/
        self.attn = build_attention(
            attn_type=getattr(config, "attn_type", "vanilla"),
            dim=config.hidden_size,
            num_heads=config.transformer["num_heads"],
            grid_size=getattr(config, "attn_grid_size", 0),
            attn_dropout=config.transformer["attention_dropout_rate"],
            vis=vis,
            **getattr(config, "attn_kwargs", {}),
        )

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(layer)

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis, in_channels=3):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size, in_channels=in_channels)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights, features


# ---- Decoder ----

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True,
                 up_type="bilinear", conv_type="conv2drelu", level=None):
        super().__init__()
        # 卷积块由 conv_type 开关决定 (见 models/decoder_ops/), 默认 conv2drelu 与现状一致;
        # level 为 decoder 级索引, 供 large_kernel 按级选取核尺寸
        self.conv1 = build_conv(
            conv_type,
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
            level=level,
        )
        self.conv2 = build_conv(
            conv_type,
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
            level=level,
        )
        # 上采样算子由 up_type 开关决定, 默认 bilinear 等价 nn.UpsamplingBilinear2d(scale_factor=2)
        self.up = build_upsample(up_type, in_channels, scale=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # decoder 可插拔算子开关 (见 models/decoder_ops/); 默认项与官方 TransUNet 等价
        up_type = getattr(config, "up_type", "bilinear")
        conv_type = getattr(config, "conv_type", "conv2drelu")
        fusion_type = getattr(config, "fusion_type", "concat")
        # 新增 3.4: 并联局部算子 (在既有 3×3 卷积旁并联大核局部支路)
        if getattr(config, "parallel_local", False):
            conv_type = "parallel_local"
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4 - self.config.n_skip):
                skip_channels[3 - i] = 0
        else:
            skip_channels = [0, 0, 0, 0]
        # 新增 3.2: up_type=carafe 时仅末级并联 CARAFE, 其余级保持 bilinear
        n_blocks = len(in_channels)
        block_up_types = [up_type] * n_blocks
        if up_type == "carafe":
            block_up_types = ["bilinear"] * (n_blocks - 1) + ["carafe"]
        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch, up_type=block_up_types[i],
                         conv_type=conv_type, level=i)
            for i, (in_ch, out_ch, sk_ch) in enumerate(
                zip(in_channels, out_channels, skip_channels))
        ]
        self.blocks = nn.ModuleList(blocks)
        # 多分辨率融合算子 (默认 concat 为占位, 不改变逐级 skip 行为)
        used_skip_channels = [c for c in skip_channels if c != 0]
        self.fusion = build_fusion(fusion_type, levels=len(blocks),
                                   channels=used_skip_channels)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        if features is not None:
            # 多分辨率融合 (默认 concat 原样返回; 跨尺度交互在新增 3.3 接入)
            features = self.fusion(features)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False, in_channels=3,
                 use_m1=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis, in_channels=in_channels)
        # M1: 编码器输出端驱动因子×LU 交叉注意力 (decoder 之前并联接入)
        self.m1 = FactorLUCroAttention(
            dim=config.hidden_size,
            num_heads=config.transformer["num_heads"],
            dropout=config.transformer["attention_dropout_rate"],
        ) if use_m1 else None
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        if self.m1 is not None:
            x = self.m1(x)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits


def _normalize_attn_kwargs(attn_kwargs) -> dict:
    """attn_kwargs 来自 YAML, 经 get_yaml 后可能已被递归转成 Namespace, 这里统一成 dict。"""
    if attn_kwargs is None:
        return {}
    if hasattr(attn_kwargs, "__dict__"):
        return dict(vars(attn_kwargs))
    return dict(attn_kwargs)


class TransUNet(nn.Module):
    """
    TransUNet, 对齐 Beckschen/TransUNet 官方 R50-ViT-B_16 实现 (从头训练).

    attn_type 为编码器注意力模块开关 (vanilla / agent / swinv2), 见 models/attention/;
    attn_kwargs 为所选模块的专属超参 (如 agent 的 num_agent)。
    up_type / conv_type / fusion_type 为 decoder 可插拔算子开关 (bilinear | dysample | carafe /
    conv2drelu | large_kernel | parallel_local / concat | cross_scale), 见 models/decoder_ops/;
    resblock_type 为 R50 主干 block 开关 (preact | convnext), 见 models/blocks/;
    parallel_local (新增 3.4) 与 high_res_skip / n_skip (新增 3.1 / B1) 为新增类开关;
    use_m1 为编码器输出端信息交互开关, 见 models/fusion/。
    """
    def __init__(self, in_channels: int = 3, class_num: int = 1, img_size: int = 224,
                 attn_type: str = "vanilla", attn_kwargs=None, use_m1: bool = False,
                 up_type: str = "bilinear", conv_type: str = "conv2drelu",
                 fusion_type: str = "concat", resblock_type: str = "preact",
                 parallel_local: bool = False, high_res_skip: bool = False,
                 n_skip: int = 3):
        super().__init__()
        config = get_r50_b16_config()
        config.n_classes = class_num
        config.patches.grid = (img_size // 16, img_size // 16)
        config.attn_type = attn_type
        config.attn_kwargs = _normalize_attn_kwargs(attn_kwargs)
        # 混合编码器对输入做 16 倍下采样, 故 token 网格边长为 img_size // 16
        config.attn_grid_size = img_size // 16
        # decoder 可插拔算子开关 (见 models/decoder_ops/)
        config.up_type = up_type
        config.conv_type = conv_type
        config.fusion_type = fusion_type
        # R50 主干 block 开关 (见 models/blocks/)
        config.resblock_type = resblock_type
        # 新增类开关 (新增 3.4 / 3.1)
        config.parallel_local = parallel_local
        config.high_res_skip = high_res_skip
        # B1 高分辨率注入需要 4 级 skip; 开启时自动提升, 避免高分辨率特征被静默丢弃
        config.n_skip = 4 if (high_res_skip and n_skip < 4) else n_skip
        self.vit = VisionTransformer(config, img_size=img_size,
                                     num_classes=class_num,
                                     zero_head=False, vis=False,
                                     in_channels=in_channels,
                                     use_m1=use_m1)

    def forward(self, x):
        return self.vit(x)


if __name__ == '__main__':
    transunet = TransUNet(in_channels=10, class_num=1, img_size=128)
    print(sum(p.numel() for p in transunet.parameters()))
    print(transunet(torch.randn(1, 10, 128, 128)).shape)
