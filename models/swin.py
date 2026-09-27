# 论文: Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
#       (Liu et al., ICCV 2021, arXiv:2103.14030)
# 来源: microsoft/Swin-Transformer (官方 PyTorch 实现, models/swin_transformer.py).
#   差异:
#     1) 官方仓库只提供分类头 (avgpool -> Linear); 官方分割版是 UperNet (另仓).
#        本文件沿用 segformer.py 的轻量解码风格: 各 stage 输出 1x1 降维到同一通道后
#        concat, 再 1x1 输出并上采样回原尺寸. 官方依赖的 timm
#        (DropPath / to_2tuple / trunc_normal_) 在本文件内自行实现, 不引入新依赖.
#     2) 官方 BasicLayer.forward 不做窗口 padding, 要求 H/W 能被 window_size 整除;
#        本项目 128 输入下 stage1~4 的分辨率为 32/16/8/4, 32、16、8 都不能被 7 整除,
#        故按 mmseg 的做法: 每个 BasicLayer 内部先补到 window_size 整数倍、跑完再裁掉,
#        注意力 mask 也按补齐后的分辨率计算.
#     3) 官方只在最后一个 stage 后加 LayerNorm; 这里每个 stage 输出后各加一个 LayerNorm
#        再送解码器 (多尺度特征需要一致归一化).
#     4) 已照搬官方的 _init_weights: 所有 nn.Linear 用 trunc_normal_(std=.02)、bias 置 0,
#        nn.LayerNorm 置 weight=1/bias=0 (PyTorch 默认即如此, 等价 no-op).
# 参数量 (in_ch=10, out_ch=1, patch_size=4, Swin-T 量级): ~28.2M

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def to_2tuple(x):
    if isinstance(x, (tuple, list)):
        return tuple(x)
    return (x, x)


class DropPath(nn.Module):
    """Stochastic Depth (对齐 timm.models.layers.DropPath)."""

    def __init__(self, drop_prob: float = 0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        random_tensor.div_(keep_prob)
        return x * random_tensor


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    """Transformer MLP (对齐官方 Mlp)."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    """基于窗口的多头自注意力 (W-MSA), 含相对位置偏置 (对齐官方 WindowAttention).

    Args:
        dim (int): 输入通道数。
        window_size (tuple[int]): 窗口的高、宽。
        num_heads (int): 注意力头数。
        qkv_bias (bool): qkv 线性层是否加偏置。
        qk_scale (float | None): 覆盖默认的 head_dim ** -0.5 缩放。
        attn_drop (float): 注意力权重的 dropout。
        proj_drop (float): 输出投影的 dropout。
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # 窗口内每个 token 的相对位置索引
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: (num_windows*B, N, C)
            mask: (0/-inf) mask, shape (num_windows, Wh*Ww, Wh*Ww) 或 None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
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


class SwinTransformerBlock(nn.Module):
    """Swin Transformer 块 (W-MSA / SW-MSA 交替, 对齐官方 SwinTransformerBlock).

    Args:
        dim (int): 输入通道数。
        input_resolution (tuple[int]): 输入分辨率 (此处传入补齐到窗口整数倍后的分辨率)。
        num_heads (int): 注意力头数。
        window_size (int): 窗口尺寸。
        shift_size (int): SW-MSA 的位移量。
        mlp_ratio (float): MLP 隐藏层扩展倍率。
        drop / attn_drop / drop_path: 各类 dropout。
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # 窗口比输入还大时不做窗口划分
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # 计算 SW-MSA 的注意力 mask
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # 循环位移
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # 划分窗口
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # 还原循环位移
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # 残差 + FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    """下采样层: 2x2 邻域拼接后线性降维 (对齐官方 PatchMerging)."""

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer(nn.Module):
    """一个 stage 的 Swin Transformer (对齐官方 BasicLayer, 另加窗口补齐/裁剪)."""

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # 原始分辨率 (H, W)
        self.depth = depth

        # 官方要求 H/W 能被 window_size 整除; 这里按 mmseg 做法补齐到整数倍, 跑完再裁掉
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

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward_features(self, x):
        """只跑本 stage 的 Transformer 块, 返回下采样前的多尺度特征。

        官方把 downsample 也放在 BasicLayer.forward 里; 这里拆成两步是为了让各 stage
        输出的分辨率依次为 H/4、H/8、H/16、H/32 (即 32/16/8/4), 便于解码器按 2^i 统一上采样。
        """
        H, W = self.input_resolution
        Hp, Wp = self.padded_resolution
        B, L, C = x.shape

        # 补齐到窗口整数倍
        if (Hp, Wp) != (H, W):
            x = x.view(B, H, W, C)
            x = F.pad(x, (0, 0, 0, Wp - W, 0, Hp - H))
            x = x.view(B, Hp * Wp, C)

        for blk in self.blocks:
            x = blk(x)

        # 裁回原始分辨率
        if (Hp, Wp) != (H, W):
            x = x.view(B, Hp, Wp, C)[:, :H, :W, :].contiguous().view(B, H * W, C)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(nn.Module):
    """图像切块嵌入 (对齐官方 PatchEmbed)."""

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class Swin(nn.Module):
    """Swin Transformer + 轻量多尺度解码器 (像素级变化检测)。

    Parameters
    ----------
    in_ch: int
        输入通道数 (本项目 = len(factor_list) + 6 = 10)。
    out_ch: int
        输出通道数 (二分类变化掩码 = 1)。
    img_size: int
        输入边长, 128。
    patch_size: int
        切块边长, 官方默认 4 -> 32x32 个 token。
    embed_dim: int
        第 1 个 stage 的通道数, 后续 stage 逐级翻倍。
    depths: tuple
        各 stage 的 Swin Transformer 块数 (官方 Swin-T: (2, 2, 6, 2))。
    num_heads: tuple
        各 stage 的注意力头数 (官方 Swin-T: (3, 6, 12, 24))。
    window_size: int
        窗口边长, 官方默认 7。
    mlp_ratio: float
        MLP 隐藏层扩展倍率。
    drop_rate / attn_drop_rate: float
        dropout / 注意力 dropout。
    drop_path_rate: float
        随机深度最大概率, 各块按 linspace 递增 (官方默认 0.1)。
    patch_norm: bool
        切块嵌入后是否加 LayerNorm。
    decoder_dim: int
        解码器统一通道数。
    """

    def __init__(self, in_ch: int = 10, out_ch: int = 1, img_size: int = 128, patch_size: int = 4,
                 embed_dim: int = 96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), window_size: int = 7,
                 mlp_ratio: float = 4., qkv_bias: bool = True, drop_rate: float = 0.,
                 attn_drop_rate: float = 0., drop_path_rate: float = 0.1, patch_norm: bool = True,
                 decoder_dim: int = 256):
        super(Swin, self).__init__()
        depths = tuple(depths)
        num_heads = tuple(num_heads)
        self.num_layers = len(depths)
        assert self.num_layers == 4, 'Swin 固定 4 个 stage, depths 需为长度 4 的序列'
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_ch, embed_dim=embed_dim,
            norm_layer=nn.LayerNorm if patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.pos_drop = nn.Dropout(p=drop_rate)

        # 随机深度: 各块概率按 linspace 递增
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        self.stage_norms = nn.ModuleList()
        for i_layer in range(self.num_layers):
            dim = int(embed_dim * 2 ** i_layer)
            layer = BasicLayer(dim=dim,
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=nn.LayerNorm,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None)
            self.layers.append(layer)
            self.stage_norms.append(nn.LayerNorm(dim))

        # 轻量解码器: 各 stage 统一到 decoder_dim 后上采样到 H/4 再融合
        dims = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.to_fused = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, decoder_dim, 1),
                nn.Upsample(scale_factor=2 ** i)
            ) for i, dim in enumerate(dims)])

        self.to_segmentation = nn.Sequential(
            nn.Conv2d(self.num_layers * decoder_dim, decoder_dim, 1),
            nn.Conv2d(decoder_dim, out_ch, 1),
        )

        # 对齐官方 SwinTransformer.__init__ 末尾的 self.apply(self._init_weights)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """官方 SwinTransformer._init_weights: Linear 用 trunc_normal_(std=.02), LN 置 1/0.

        官方对 nn.LayerNorm 的赋值与 PyTorch 默认值相同 (weight=1, bias=0), 等价 no-op;
        对 nn.Linear 则把 PyTorch 默认的 kaiming_uniform 换成 trunc_normal(std=.02),
        这一点会影响早期收敛与可承受的学习率, 故照搬.
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        input_size = x.shape[-2:]
        x = self.patch_embed(x)   # (B, L, C)
        x = self.pos_drop(x)

        fused = []
        for i_layer, (layer, norm, to_fused) in enumerate(zip(self.layers, self.stage_norms, self.to_fused)):
            x = layer.forward_features(x)
            feat = norm(x)
            B, L, C = feat.shape
            g = self.patches_resolution[0] // (2 ** i_layer)
            assert L == g * g, f'stage {i_layer} 的 token 数 {L} 与分辨率 {g}x{g} 不匹配'
            feat = feat.transpose(1, 2).reshape(B, C, g, g)
            fused.append(to_fused(feat))
            x = layer.downsample(x) if layer.downsample is not None else x

        logits = self.to_segmentation(torch.cat(fused, dim=1))
        logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
        return logits
