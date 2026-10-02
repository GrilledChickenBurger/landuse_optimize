# 论文: CrossFormer: A Versatile Vision Transformer Based on Cross-scale Attention
#       (Wang et al., ICLR 2022, arXiv:2108.00154)
# 来源: cheerss/CrossFormer (官方 PyTorch 实现, models/crossformer.py).
# 差异:
#   1) 官方依赖 timm (DropPath/to_2tuple/trunc_normal_); 本项目复用 models/swin.py 内
#      已自行实现的 DropPath/to_2tuple/Mlp, trunc_normal_ 用 nn.init.trunc_normal_ 替代,
#      不引入 timm.
#   2) 官方为分类骨干 (avgpool + Linear head); 本项目沿用 swin.py 的轻量多尺度解码器
#      改造为逐像素分割: 各 stage 输出 1x1 降维到 decoder_dim 后按 2^i 上采样, concat,
#      再 1x1 输出并上采样回原尺寸. 骨干的 CEL 跨尺度嵌入 / LSDA 长短距离注意力 /
#      DPB 动态位置偏置全部照搬官方.
#   3) 官方默认 img_size=224、crs_interval=[8,4,2,1] (针对 224 设计); 本项目 img_size=128
#      下启用 adaptive_interval=True, 使 LDA 的采样间隔按 stage 分辨率自适应, 避免过度 padding.
#   4) 官方 num_classes 为分类类别数; 本项目 out_ch=1 (二分类变化掩码).
# 参数量 (in_ch=10, out_ch=1, CrossFormer-T 量级): ~19.6M

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin import to_2tuple, DropPath, Mlp

NEG_INF = -1000000


class DynamicPosBias(nn.Module):
    """动态位置偏置 DPB: 用 MLP 基于相对坐标生成位置偏置 (对齐官方 DynamicPosBias)。"""

    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(nn.LayerNorm(self.pos_dim), nn.ReLU(inplace=True),
                                  nn.Linear(self.pos_dim, self.pos_dim))
        self.pos2 = nn.Sequential(nn.LayerNorm(self.pos_dim), nn.ReLU(inplace=True),
                                  nn.Linear(self.pos_dim, self.pos_dim))
        self.pos3 = nn.Sequential(nn.LayerNorm(self.pos_dim), nn.ReLU(inplace=True),
                                  nn.Linear(self.pos_dim, self.num_heads))

    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases)
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos


class Attention(nn.Module):
    """带动态位置偏置的多头自注意力 (对齐官方 Attention, 即 LSDA 的注意力核)。"""

    def __init__(self, dim, group_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0., position_bias=True):
        super().__init__()
        self.dim = dim
        self.group_size = group_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.position_bias = position_bias

        if position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)
            position_bias_h = torch.arange(1 - self.group_size[0], self.group_size[0])
            position_bias_w = torch.arange(1 - self.group_size[1], self.group_size[1])
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))
            biases = biases.flatten(1).transpose(0, 1).float()
            self.register_buffer("biases", biases, persistent=False)

            coords_h = torch.arange(self.group_size[0])
            coords_w = torch.arange(self.group_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.group_size[0] - 1
            relative_coords[:, :, 1] += self.group_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.group_size[1] - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index, persistent=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.position_bias:
            pos = self.pos(self.biases)
            relative_position_bias = pos[self.relative_position_index.view(-1)].view(
                self.group_size[0] * self.group_size[1], self.group_size[0] * self.group_size[1], -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossFormerBlock(nn.Module):
    """CrossFormer Block: LSDA 长短距离注意力 (SDA/LDA 交替) + DPB (对齐官方)。"""

    def __init__(self, dim, input_resolution, num_heads, group_size=7, interval=8, lsda_flag=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_patch_size=1,
                 pad_type=0, use_extra_conv=True, use_cpe=False, no_mask=False, adaptive_interval=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.group_size = group_size
        self.interval = interval
        self.lsda_flag = lsda_flag
        self.mlp_ratio = mlp_ratio
        self.pad_type = pad_type
        self.use_extra_conv = use_extra_conv
        self.use_cpe = use_cpe

        if min(self.input_resolution) <= self.group_size:
            self.lsda_flag = 0
            self.group_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, group_size=to_2tuple(self.group_size), num_heads=num_heads,
                              qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=drop, position_bias=(not use_cpe))

        if adaptive_interval:
            self.interval = int(np.ceil(self.input_resolution[0] / self.group_size))

        if self.use_extra_conv:
            self.ex_conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
            self.ex_ln = norm_layer(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # attention mask (对 padding 区域屏蔽)
        attn_mask = None
        if not no_mask:
            H, W = self.input_resolution
            size_div = self.interval * self.group_size if self.lsda_flag == 1 else self.group_size
            pad_w = (size_div - W % size_div) % size_div
            pad_h = (size_div - H % size_div) % size_div
            pad_l = pad_t = 0
            pad_r = pad_w - pad_l
            pad_b = pad_h - pad_t
            Hp = H + pad_h
            Wp = W + pad_w
            mask = torch.zeros((1, Hp, Wp, 1))
            if pad_h > 0:
                mask[:, -pad_b:, :, :] = -1
                mask[:, : pad_t, :, :] = -1
            if pad_w > 0:
                mask[:, :, -pad_r:, :] = -1
                mask[:, :, : pad_l, :] = -1
            if self.lsda_flag == 0:
                G = self.group_size
                nG = Hp * Wp // G ** 2
                if pad_w > 0 or pad_h > 0:
                    mask = mask.reshape(1, Hp // G, G, Wp // G, G, 1).permute(0, 1, 3, 2, 4, 5).contiguous()
                    mask = mask.reshape(nG, 1, G * G)
                    attn_mask = torch.zeros((nG, G * G, G * G))
                    attn_mask = attn_mask.masked_fill(mask < 0, NEG_INF)
            else:
                I = self.interval
                G = self.group_size
                Rh, Rw = Hp // (G * I), Wp // (G * I)
                nG = I ** 2 * Rh * Rw
                if pad_w > 0 or pad_h > 0:
                    mask = mask.reshape(1, Rh, G, I, Rw, G, I, 1).permute(0, 1, 4, 3, 6, 2, 5, 7).contiguous()
                    mask = mask.reshape(nG, 1, G * G)
                    attn_mask = torch.zeros((nG, G * G, G * G))
                    attn_mask = attn_mask.masked_fill(mask < 0, NEG_INF)
        self.register_buffer("attn_mask", attn_mask, persistent=False)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        size_div = self.interval * self.group_size if self.lsda_flag == 1 else self.group_size
        pad_w = (size_div - W % size_div) % size_div
        pad_h = (size_div - H % size_div) % size_div
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        _, Hp, Wp, _ = x.shape

        if self.lsda_flag == 0:
            G = self.group_size
            x = x.reshape(B, Hp // G, G, Wp // G, G, C).permute(0, 1, 3, 2, 4, 5)
            x = x.reshape(B * Hp * Wp // G ** 2, G ** 2, C)
        else:
            I = self.interval
            G = self.group_size
            Rh, Rw = Hp // (G * I), Wp // (G * I)
            x = x.reshape(B, Rh, G, I, Rw, G, I, C).permute(0, 1, 4, 3, 6, 2, 5, 7).contiguous()
            x = x.reshape(B * Rh * Rw * I * I, G * G, C)

        x = self.attn(x, mask=self.attn_mask)

        if self.lsda_flag == 0:
            G = self.group_size
            x = x.reshape(B, Hp // G, Wp // G, G, G, C).permute(0, 1, 3, 2, 4, 5).contiguous()
        else:
            I = self.interval
            G = self.group_size
            Rh, Rw = Hp // (G * I), Wp // (G * I)
            x = x.reshape(B, Rh, Rw, I, I, G, G, C).permute(0, 1, 5, 3, 2, 6, 4, 7).contiguous()
        x = x.view(B, Hp, Wp, C)

        if pad_w > 0 or pad_h > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.use_extra_conv:
            x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            x = self.ex_conv(x)
            x = x.permute(0, 2, 3, 1).view(B, H * W, C).contiguous()
            x = self.ex_ln(x)
        return x


class PatchMerging(nn.Module):
    """Patch Merging: 多卷积 (stride=2) 下采样 + 拼接 (对齐官方, 区别于 Swin 的线性降维)。"""

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, patch_size=[2],
                 num_input_patch_size=1):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.patch_size = patch_size
        self.norm = norm_layer(dim)
        self.reductions = nn.ModuleList()
        for i, ps in enumerate(patch_size):
            if i == len(patch_size) - 1:
                out_dim = 2 * dim // 2 ** i
            else:
                out_dim = 2 * dim // 2 ** (i + 1)
            stride = 2
            padding = (ps - stride) // 2
            self.reductions.append(nn.Conv2d(dim, out_dim, kernel_size=ps, stride=stride, padding=padding))

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = self.norm(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        xs = [reduction(x).flatten(2).transpose(1, 2) for reduction in self.reductions]
        x = torch.cat(xs, dim=2)
        return x


class Stage(nn.Module):
    """一个 stage: 若干 CrossFormerBlock + 可选 PatchMerging 下采样 (对齐官方 Stage)。"""

    def __init__(self, dim, input_resolution, depth, num_heads, group_size, interval,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 patch_size_end=[4], num_patch_size=None, use_cpe=False, pad_type=0,
                 no_mask=False, adaptive_interval=False, use_acl=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList()
        for i in range(depth):
            lsda_flag = 0 if (i % 2 == 0) else 1
            use_extra_conv = ((i + 1) % 3 == 0) and (i < depth - 1) and use_acl
            self.blocks.append(CrossFormerBlock(
                dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                group_size=group_size[i], interval=interval, lsda_flag=lsda_flag,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, num_patch_size=num_patch_size,
                use_extra_conv=use_extra_conv, use_cpe=use_cpe, pad_type=pad_type,
                no_mask=no_mask, adaptive_interval=adaptive_interval))

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer,
                                         patch_size=patch_size_end, num_input_patch_size=num_patch_size)
        else:
            self.downsample = None

    def forward_features(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(nn.Module):
    """CEL 跨尺度嵌入层: 多个不同 kernel_size 的卷积 patch embed + 拼接 (对齐官方 PatchEmbed)。"""

    def __init__(self, img_size=224, patch_size=[4], in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[0] // patch_size[0]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.projs = nn.ModuleList()
        for i, ps in enumerate(patch_size):
            if i == len(patch_size) - 1:
                dim = embed_dim // 2 ** i
            else:
                dim = embed_dim // 2 ** (i + 1)
            stride = patch_size[0]
            padding = (ps - patch_size[0]) // 2
            self.projs.append(nn.Conv2d(in_chans, dim, kernel_size=ps, stride=stride, padding=padding))
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        B, C, H, W = x.shape
        xs = [proj(x).flatten(2).transpose(1, 2) for proj in self.projs]
        x = torch.cat(xs, dim=2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class CrossFormer(nn.Module):
    """CrossFormer 骨干 + 轻量多尺度解码器 (像素级变化检测)。

    Parameters
    ----------
    in_ch: int, 输入通道数 (本项目 = len(factor_list) + 6 = 10)。
    out_ch: int, 输出通道数 (二分类变化掩码 = 1)。
    img_size: int, 输入边长, 128。
    patch_size: list[int], CEL 跨尺度嵌入的卷积核列表 (官方 CrossFormer-T: [4,8,16,32])。
    embed_dim: int, 第 1 个 stage 通道数 (官方 CrossFormer-T: 64)。
    depths: tuple, 各 stage 块数 (官方 CrossFormer-T: (1,1,8,6))。
    num_heads: tuple, 各 stage 头数 (官方 CrossFormer-T: (2,4,8,16))。
    group_size: list, 各 stage 分组大小 (官方 (7,7,7,7))。
    merge_size: list[list[int]], 各 PatchMerging 的卷积核列表 (官方 [[2,4],[2,4],[2,4]])。
    crs_interval: list, LDA 采样间隔 (官方 [8,4,2,1]; img_size=128 下启用 adaptive_interval)。
    decoder_dim: int, 解码器统一通道数。
    """

    def __init__(self, in_ch: int = 10, out_ch: int = 1, img_size: int = 128,
                 patch_size=(4, 8, 16, 32), embed_dim: int = 64,
                 depths=(1, 1, 8, 6), num_heads=(2, 4, 8, 16),
                 group_size=(7, 7, 7, 7), merge_size=((2, 4), (2, 4), (2, 4)),
                 crs_interval=(8, 4, 2, 1), mlp_ratio=(4., 4., 4., 4.),
                 qkv_bias: bool = True, drop_rate: float = 0., attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.1, patch_norm: bool = True,
                 group_type: str = 'constant', adaptive_interval: bool = True,
                 decoder_dim: int = 256):
        super().__init__()
        depths = list(depths)
        num_heads = list(num_heads)
        patch_size = list(patch_size)
        merge_size = [list(m) for m in merge_size]
        group_size = list(group_size)
        mlp_ratio = list(mlp_ratio)
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_ch, embed_dim=embed_dim,
            norm_layer=nn.LayerNorm if patch_norm else None)
        self.patches_resolution = self.patch_embed.patches_resolution
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        group_size_per_block = self._compute_group_size(group_size, depths, self.patches_resolution, group_type)

        num_patch_sizes = [len(patch_size)] + [len(m) for m in merge_size]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            patch_size_end = merge_size[i_layer] if i_layer < self.num_layers - 1 else None
            num_patch_size = num_patch_sizes[i_layer]
            layer = Stage(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                  self.patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer], num_heads=num_heads[i_layer],
                group_size=group_size_per_block[i_layer], interval=crs_interval[i_layer],
                mlp_ratio=self.mlp_ratio[i_layer], qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=nn.LayerNorm,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                patch_size_end=patch_size_end, num_patch_size=num_patch_size,
                adaptive_interval=adaptive_interval)
            self.layers.append(layer)

        # 轻量多尺度解码器 (同 swin.py)
        dims = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.to_fused = nn.ModuleList([
            nn.Sequential(nn.Conv2d(dim, decoder_dim, 1), nn.Upsample(scale_factor=2 ** i))
            for i, dim in enumerate(dims)])
        self.to_segmentation = nn.Sequential(
            nn.Conv2d(self.num_layers * decoder_dim, decoder_dim, 1),
            nn.Conv2d(decoder_dim, out_ch, 1))

        self.apply(self._init_weights)

    def _compute_group_size(self, group_size, depths, resolution, group_type='constant'):
        """生成每 block 的 group size (对齐官方 compute_group_size)。"""
        rst_group_size = []
        min_size = 4
        total_depth = sum(depths)
        step_size = (1 - min_size / resolution[0]) / total_depth
        group_fraction = np.arange(min_size / resolution[0], 1.0, step_size)
        cnt = 0
        for i_stage in range(len(depths)):
            rst_group_size.append([])
            cur_resolution = resolution[0] // 2 ** i_stage
            for _ in range(depths[i_stage]):
                if group_type == 'constant':
                    gz = group_size[i_stage]
                elif group_type == 'linear':
                    gz = cur_resolution * group_fraction[cnt]
                    gz = max(4, int(np.ceil(gz)))
                elif group_type == 'linear_div':
                    gz = cur_resolution * group_fraction[cnt]
                    if gz > cur_resolution // 2:
                        gz = cur_resolution if gz > cur_resolution * 3 / 4 or i_stage != 2 else cur_resolution // 2
                    gz = max(4, int(np.ceil(gz)))
                elif group_type == 'alter':
                    gz = cur_resolution * group_fraction[cnt]
                    if gz > cur_resolution // 2:
                        gz = cur_resolution if cnt % 2 != 0 or i_stage != 2 else cur_resolution // 2
                    gz = max(4, int(np.ceil(gz)))
                else:
                    gz = group_size[i_stage]
                rst_group_size[i_stage].append(gz)
                cnt += 1
        return rst_group_size

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        input_size = x.shape[-2:]
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        features = []
        for i, layer in enumerate(self.layers):
            x = layer.forward_features(x)
            B, L, C = x.shape
            g = self.patches_resolution[0] // (2 ** i)
            feat = x.transpose(1, 2).reshape(B, C, g, g)
            features.append(self.to_fused[i](feat))
            x = layer.downsample(x) if layer.downsample is not None else x

        logits = self.to_segmentation(torch.cat(features, dim=1))
        logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
        return logits
