# 论文: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
#       (Dosovitskiy et al., ICLR 2021, arXiv:2010.11929)
# 来源: google-research/vision_transformer (官方实现, JAX/Flax, vit_jax/models_vit.py).
#   差异: 官方只有 JAX/Flax 版且只做图像分类 (输出 cls token -> num_classes),
#        本文件按官方结构用 PyTorch 重写, 并加一个轻量解码器做逐像素预测:
#     1) 保留官方的 PatchEmbed(卷积切块) / cls token(零初始化) / 可学习位置编码
#        (正态 std=0.02) / PreNorm Encoder1DBlock / 末尾 encoder_norm。
#     2) 分割头丢掉 cls token, 把 token 网格 reshape 回特征图 -> 3x3 卷积细化 ->
#        双线性上采样回输入分辨率 -> 1x1 输出 (官方无分割头, 属本项目新增)。
#     3) 官方 ViT-Base 是 img 224 / patch 16 (14x14 token), 本项目输入 128x128 且需要
#        密集预测, 故默认 patch=8 (16x16=256 token); dim / depth / heads / dropout 均对齐
#        官方 ViT-S_16 配置 (dim=384 / depth=12 / heads=6 / dropout=0.0 / attn_dropout=0.0);
#        所有超参均可在 YAML 的 model.vit 节覆写。
# 参数量 (in_ch=10, out_ch=1, patch_size=8, dim=384, depth=12, heads=6): ~22.3M

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """切块 + 线性投影 (对齐官方 embedding 层: 用卷积等价实现 s2d + emb)."""

    def __init__(self, in_ch: int, dim: int, patch_size: int):
        super(PatchEmbed, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                    # (B, dim, H/p, W/p)
        return x.flatten(2).transpose(1, 2)  # (B, N, dim)


class MlpBlock(nn.Module):
    """Transformer MLP (对齐官方 MlpBlock: Dense -> GELU -> Dropout -> Dense -> Dropout)."""

    def __init__(self, dim: int, mlp_dim: int, dropout: float = 0.1):
        super(MlpBlock, self).__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mlp_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop1(self.act(self.fc1(x)))
        return self.drop2(self.fc2(x))


class Attention(nn.Module):
    """多头自注意力 (对齐官方 MultiHeadDotProductAttention, broadcast_dropout=False)."""

    def __init__(self, dim: int, heads: int, attn_dropout: float = 0.1, proj_dropout: float = 0.1):
        super(Attention, self).__init__()
        assert dim % heads == 0, 'dim 必须能被 heads 整除'
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class Encoder1DBlock(nn.Module):
    """Transformer 编码块 (对齐官方 Encoder1DBlock: PreNorm + 残差)."""

    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float = 0.1, attn_dropout: float = 0.1):
        super(Encoder1DBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, attn_dropout=attn_dropout, proj_dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MlpBlock(dim, mlp_dim, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """Vision Transformer + 轻量分割解码器。

    Parameters
    ----------
    in_ch: int
        输入通道数 (本项目 = len(factor_list) + 6 = 10)。
    out_ch: int
        输出通道数 (二分类变化掩码 = 1)。
    img_size: int
        输入边长, 128。
    patch_size: int
        切块边长, 默认 8 -> 16x16 = 256 个 token。
    dim: int
        token 嵌入维度, 默认 384 (与官方 ViT-S_16 一致; 官方 ViT-B_16 为 768)。
    depth: int
        Transformer 块数, 默认 12 (官方 ViT-S_16 / B_16 均为 12)。
    heads: int
        多头注意力头数, 默认 6 (与官方 ViT-S_16 一致; 官方 ViT-B_16 为 12)。
    mlp_ratio: float
        MLP 隐藏层扩展倍率 (官方为 4)。
    dropout: float
        残差/MLP 输出的 dropout, 默认 0.0 (对齐官方 ViT-S_16 / B_16 的 dropout_rate;
        0.1 仅出现在官方的 testing 调试配置中)。
    attn_dropout: float
        注意力权重 dropout, 默认 0.0 (同上, 对齐官方 ViT-S_16 / B_16)。
    decoder_dim: int
        分割解码器中间通道数。
    """

    def __init__(self, in_ch: int = 10, out_ch: int = 1, img_size: int = 128, patch_size: int = 8,
                 dim: int = 384, depth: int = 12, heads: int = 6, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, attn_dropout: float = 0.0, decoder_dim: int = 256):
        super(ViT, self).__init__()
        assert img_size % patch_size == 0, 'img_size 必须能被 patch_size 整除'
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.patch_embed = PatchEmbed(in_ch, dim, patch_size)

        # 官方的 cls token (零初始化) 与可学习位置编码 (正态 std=0.02, from BERT)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))
        self.pos_drop = nn.Dropout(dropout)
        nn.init.normal_(self.pos_embed, std=0.02)

        mlp_dim = int(dim * mlp_ratio)
        self.blocks = nn.ModuleList([
            Encoder1DBlock(dim, heads, mlp_dim, dropout=dropout, attn_dropout=attn_dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

        # 分割解码器 (本项目新增): token 网格 -> 逐像素 logits
        self.proj = nn.Conv2d(dim, decoder_dim, kernel_size=1)
        self.refine = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.GELU(),
        )
        self.head = nn.Conv2d(decoder_dim, out_ch, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[-2:]
        b = x.shape[0]
        # 位置编码与 token 网格都按 img_size 预计算, 输入尺寸必须一致
        assert input_size == (self.img_size, self.img_size), \
            f"ViT 的 img_size={self.img_size}, 但输入为 {tuple(input_size)}; " \
            f"改变输入尺寸时需同步修改 model.vit.img_size"

        x = self.patch_embed(x)                                  # (B, N, dim)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed          # (B, N+1, dim)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # 丢掉 cls token, 其余 token reshape 回 token 网格
        x = x[:, 1:]
        g = self.grid_size
        x = x.transpose(1, 2).reshape(b, -1, g, g)                # (B, dim, g, g)

        x = self.proj(x)
        x = self.refine(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        return self.head(x)
