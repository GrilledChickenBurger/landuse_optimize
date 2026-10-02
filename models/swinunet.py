# 论文: Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation
#       (Cao et al., 2022, arXiv:2105.05537)
# 来源: HuCaoFighting/Swin-Unet (官方 PyTorch 实现,
#       networks/swin_transformer_unet_skip_expand_decoder_sys.py).
# 差异:
#   1) 官方依赖 timm (DropPath/to_2tuple/trunc_normal_) 与 einops (PatchExpand 的
#      rearrange); 本项目复用 models/swin.py 内已自行实现的同名组件, 并把 PatchExpand
#      的 einops rearrange 改写为纯 PyTorch 的 view+permute, 不引入新依赖.
#   2) 官方 img_size=224 下各 stage 分辨率 56/28/14/7 恰好能被 window_size=7 整除,
#      无需窗口 padding; 本项目 img_size=128 下分辨率为 32/16/8/4, 与 window_size=7
#      不能整除, 故编码器复用 swin.py BasicLayer 的窗口补齐逻辑 (pad 到 window_size
#      整数倍再裁回), 解码器 BasicLayerUp 采用同样处理.
#   3) 官方输出 num_classes 为多类 (Synapse 多器官), 本项目单通道二分类 (out_ch=1),
#      输出头为 Conv2d(embed_dim, 1, 1), 其余结构 (编码器 4 层 Swin + 解码器 3 层
#      PatchExpand + 对称跳跃连接 + FinalPatchExpand_X4) 完全对齐官方 SwinTransformerSys.
#   4) 官方用 SGD + poly 学习率; 本项目沿用 hybrid_train 的 AdamW + cosine, 仅结构
#      与损失对齐官方 (损失见 utils/training_utils.py 的 cal_loss_ce_dice_swinunet).
# 参数量 (in_ch=10, out_ch=1, patch_size=4, Swin-T 量级): ~27.1M

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin import (
    to_2tuple, DropPath, window_partition, window_reverse,
    Mlp, WindowAttention, SwinTransformerBlock, PatchMerging,
    BasicLayer, PatchEmbed,
)


class PatchExpand(nn.Module):
    """2x 上采样层 (对齐官方 PatchExpand).

    官方用 einops.rearrange('b h w (p1 p2 c) -> b (h p1) (w p2) c', p1=p2=2) 实现;
    此处改写为纯 PyTorch 的 view + permute + reshape.
    """

    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        p = self.dim_scale
        x = x.view(B, H, W, p, p, C // (p * p))
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * p, W * p, C // (p * p))
        x = x.view(B, -1, C // (p * p))
        x = self.norm(x)
        return x


class FinalPatchExpand_X4(nn.Module):
    """4x 上采样层 (对齐官方 FinalPatchExpand_X4), 通道回到 dim。"""

    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        p = self.dim_scale
        x = x.view(B, H, W, p, p, C // (p * p))
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * p, W * p, C // (p * p))
        x = x.view(B, -1, C // (p * p))
        x = self.norm(x)
        return x


class BasicLayerUp(nn.Module):
    """解码器 stage: 若干 SwinTransformerBlock + PatchExpand 上采样 (对齐官方 BasicLayer_up).

    与编码器 BasicLayer 的窗口补齐策略一致: 低分辨率下先 pad 到 window_size 整数倍,
    跑完 blocks 再裁回原分辨率.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        window_size = min(window_size, input_resolution[0], input_resolution[1])
        self.window_size = window_size
        self.padded_resolution = (math.ceil(input_resolution[0] / window_size) * window_size,
                                  math.ceil(input_resolution[1] / window_size) * window_size)
        shift_size = 0 if window_size == 1 else window_size // 2

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=self.padded_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else shift_size,
                                 mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop, drop_path=drop_path[i],
                                 norm_layer=norm_layer)
            for i in range(depth)])

        if upsample:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        H, W = self.input_resolution
        Hp, Wp = self.padded_resolution
        B, L, C = x.shape

        if (Hp, Wp) != (H, W):
            x = x.view(B, H, W, C)
            x = F.pad(x, (0, 0, 0, Wp - W, 0, Hp - H))
            x = x.view(B, Hp * Wp, C)

        for blk in self.blocks:
            x = blk(x)

        if (Hp, Wp) != (H, W):
            x = x.view(B, Hp, Wp, C)[:, :H, :W, :].contiguous().view(B, H * W, C)

        if self.upsample is not None:
            x = self.upsample(x)
        return x


class SwinUNet(nn.Module):
    """Swin-Unet: 纯 Transformer U-Net 变化检测模型。

    Parameters
    ----------
    in_ch: int, 输入通道数 (本项目 = len(factor_list) + 6 = 10)。
    out_ch: int, 输出通道数 (二分类变化掩码 = 1)。
    img_size: int, 输入边长, 128。
    patch_size: int, 切块边长, 官方默认 4 -> 32x32 token。
    embed_dim: int, 第 1 个 stage 通道数, 后续逐级翻倍。
    depths: tuple, 编码器各 stage 块数 (官方 Swin-T: (2,2,2,2))。
    depths_decoder: tuple, 解码器各 stage 块数 (官方 (2,2,2))。
    num_heads: tuple, 编码器各 stage 头数 (官方 Swin-T: (3,6,12,24))。
    window_size: int, 窗口边长, 官方默认 7。
    """

    def __init__(self, in_ch: int = 10, out_ch: int = 1, img_size: int = 128, patch_size: int = 4,
                 embed_dim: int = 96, depths=(2, 2, 2, 2), depths_decoder=(2, 2, 2),
                 num_heads=(3, 6, 12, 24), window_size: int = 7, mlp_ratio: float = 4.,
                 qkv_bias: bool = True, drop_rate: float = 0., attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.1, patch_norm: bool = True):
        super().__init__()
        depths = tuple(depths)
        depths_decoder = tuple(depths_decoder)
        num_heads = tuple(num_heads)
        self.num_layers = len(depths)
        assert self.num_layers == 4, 'SwinUNet 编码器固定 4 个 stage'
        assert len(depths_decoder) == 3, 'SwinUNet 解码器固定 3 个 stage'
        self.embed_dim = embed_dim
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_ch, embed_dim=embed_dim,
            norm_layer=nn.LayerNorm if patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution  # (H/4, W/4)
        self.patches_resolution = patches_resolution
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 编码器: 4 层 BasicLayer (复用 swin.py, 含窗口补齐)
        dpr_enc = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            dim = int(embed_dim * 2 ** i_layer)
            layer = BasicLayer(
                dim=dim,
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                  patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer], num_heads=num_heads[i_layer],
                window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr_enc[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=nn.LayerNorm,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None)
            self.layers.append(layer)
        self.norm = nn.LayerNorm(int(embed_dim * 2 ** (self.num_layers - 1)))

        # 解码器: 3 层 BasicLayerUp + 跳跃连接降维
        dpr_dec = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))]
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i in range(3):
            dim = int(embed_dim * 2 ** (3 - i))
            self.concat_back_dim.append(nn.Linear(2 * dim, dim))
            layer_up = BasicLayerUp(
                dim=dim,
                input_resolution=(patches_resolution[0] // (2 ** (3 - i)),
                                  patches_resolution[1] // (2 ** (3 - i))),
                depth=depths_decoder[i], num_heads=num_heads[3 - i],
                window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr_dec[sum(depths_decoder[:i]):sum(depths_decoder[:i + 1])],
                norm_layer=nn.LayerNorm, upsample=True)
            self.layers_up.append(layer_up)

        self.norm_up = nn.LayerNorm(embed_dim)
        self.up = FinalPatchExpand_X4(input_resolution=patches_resolution, dim=embed_dim)
        self.output = nn.Conv2d(embed_dim, out_ch, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)          # (B, L, 96)
        x = self.pos_drop(x)

        # 编码器 + 记录各层输入作跳跃连接
        x_downsample = []
        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)
        x = self.norm(x)                  # 瓶颈

        # 解码器
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[3 - inx]], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        x = self.norm_up(x)               # (B, L, 96), L = (H/4)*(W/4)
        x = self.up(x)                    # FinalPatchExpand_X4 -> (B, H*W, 96)
        B, _, C = x.shape
        x = x.view(B, self.img_size, self.img_size, C).permute(0, 3, 1, 2)
        x = self.output(x)                # (B, out_ch, H, W)
        return x
