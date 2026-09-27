# 论文: Image-to-Image Translation with Conditional Adversarial Networks
#       (Isola et al., CVPR 2017, arXiv:1611.07004; 即 pix2pix / 条件 GAN)
# 来源: junyanz/pytorch-CycleGAN-and-pix2pix (论文作者官方 PyTorch 实现, models/networks.py).
#   差异:
#     1) 只移植生成器 UnetGenerator / UnetSkipConnectionBlock, 不含判别器 —— 本项目
#        create_model() 返回单网络, 训练循环是 model(x) -> logits 模式;
#        生成器的 U-Net 结构本身即可作为像素级预测网络单独训练与公平对比.
#     2) 官方 outermost 输出层后接 nn.Tanh() (输出限定 [-1, 1]); 本项目损失是
#        BCE/Focal(Dice), 需要 raw logits, 故默认改为 Identity, 可用 output_activation="tanh"
#        还原官方行为.
#     3) 官方 unet_128 -> num_downs=7, unet_256 -> num_downs=8; 本项目输入 128x128,
#        故默认 num_downs=7 (128 / 2^7 = 1, 瓶颈 1x1).
#     4) 官方 define_G 默认 norm="batch" (pix2pix 论文设置), 此处保持一致;
#        CycleGAN 用 instance, 可通过 norm="instance" 切换.
# 参数量 (in_ch=10, out_ch=1, num_downs=7, ngf=64): ~41.8M

import functools

import torch
import torch.nn as nn


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type="batch"):
    """返回归一化层构造器 (对齐官方 get_norm_layer)."""
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == "none":
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


class UnetSkipConnectionBlock(nn.Module):
    """带跳跃连接的 U-Net 子模块 (对齐官方 UnetSkipConnectionBlock).

    X -------------------identity----------------------
    |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False,
                 innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # 加上跳跃连接
            return torch.cat([x, self.model(x)], 1)


class UnetGenerator(nn.Module):
    """pix2pix 的 U-Net 生成器 (对齐官方 UnetGenerator).

    Parameters
    ----------
    input_nc: int
        输入通道数 (本项目 = len(factor_list) + 6 = 10)。
    output_nc: int
        输出通道数 (二分类变化掩码 = 1)。
    num_downs: int
        下采样次数; num_downs=7 时 128x128 输入在瓶颈处变为 1x1。
    ngf: int
        最后一层卷积的通道基数, 内部逐级放大到 ngf*8。
    norm_layer: 归一化层构造器。
    use_dropout: bool
        中间层是否使用 Dropout(0.5)。
    """

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()
        # 由内向外递归构建 U-Net
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None,
                                             norm_layer=norm_layer, innermost=True)
        for _ in range(num_downs - 5):  # 中间层保持 ngf * 8
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        # 通道数从 ngf * 8 逐级降到 ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                             outermost=True, norm_layer=norm_layer)

    def forward(self, x):
        return self.model(x)


class CGAN(nn.Module):
    """条件 GAN (pix2pix) 的生成器, 作为变化检测网络使用。

    结构与官方 UnetGenerator 一致, 仅去掉 outermost 的 Tanh 使输出为 raw logits,
    同时保留 output_activation="tanh" 以还原官方 pix2pix 行为。

    Parameters
    ----------
    in_ch: int
        输入通道数。
    out_ch: int
        输出通道数。
    num_downs: int
        下采样次数, 128x128 输入推荐 7。
    ngf: int
        生成器基础通道数 (官方默认 64)。
    norm: str
        归一化类型: batch | instance | none (官方 define_G 默认 batch)。
    use_dropout: bool
        是否使用 Dropout(0.5)。
    output_activation: str 或 None
        None -> Identity (raw logits, 本项目默认); "tanh" -> 官方 pix2pix 的 Tanh。
    """

    def __init__(self, in_ch: int = 10, out_ch: int = 1, num_downs: int = 7, ngf: int = 64,
                 norm: str = "batch", use_dropout: bool = False, output_activation: str = None):
        super(CGAN, self).__init__()
        norm_layer = get_norm_layer(norm_type=norm)
        self.netG = UnetGenerator(in_ch, out_ch, num_downs, ngf=ngf,
                                  norm_layer=norm_layer, use_dropout=use_dropout)

        if output_activation is None:
            self.activation = Identity()
        elif output_activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"不支持的 output_activation: {output_activation} (可选 None / 'tanh')")

    def forward(self, x):
        return self.activation(self.netG(x))
