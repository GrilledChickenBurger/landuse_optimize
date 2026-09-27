# 论文: Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting
#       (Shi et al., NeurIPS 2015, arXiv:1506.04214)
# 来源: ndrplz/ConvLSTM_pytorch (社区最通用的 PyTorch 实现).
#   说明: PyTorch 官方仓库未提供 ConvLSTM; 论文作者的官方代码为 Torch/Lua 版,
#        ndrplz 版是学界引用最多的 PyTorch 基线, 故以它为准.
#   差异: ConvLSTMCell / ConvLSTM 与上游逐行一致; 上游只提供序列层, 没有像素级网络,
#        本文件在其上补齐 ConvLSTM_Seg 编解码封装 (4 级 ConvLSTM 编码 + U-Net 风格解码).
#
# 输入约定: 支持 (B, T, C, H, W) 与 (B, C, H, W) 两种输入, 4D 输入按 T=1 处理.
#   本项目样本是单期 (4 驱动因子 + 6 通道 LU one-hot, 共 10 通道), 默认 T=1,
#   此时 ConvLSTM 退化为带输入门/遗忘门/输出门的门控卷积.
# 参数量 (in_ch=10, out_ch=1, filters=(64,128,256,512)): ~42.4M

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    ConvLSTM 单元 (对齐 ndrplz/ConvLSTM_pytorch).

    门控在卷积特征图上做逐元素运算 (而非全连接), 因此能同时建模空间与时间依赖。

    Parameters
    ----------
    input_dim: int
        输入张量的通道数。
    hidden_dim: int
        隐状态 h / 记忆 c 的通道数。
    kernel_size: (int, int)
        卷积核尺寸。
    bias: bool
        卷积是否使用偏置。
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # 沿通道维拼接输入与上一步隐状态

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    """
    多层 ConvLSTM (对齐 ndrplz/ConvLSTM_pytorch).

    Parameters
    ----------
    input_dim: int
        输入通道数。
    hidden_dim: int
        隐状态通道数 (每层)。
    kernel_size: tuple 或 list[tuple]
        卷积核尺寸, 可以是单个尺寸 (所有层共用) 或每层一个。
    num_layers: int
        堆叠的 ConvLSTM 层数。
    batch_first: bool
        输入的第 0 维是否为 batch。设为 False 时输入为 (T, B, C, H, W)。
    bias: bool
        卷积是否使用偏置。
    return_all_layers: bool
        是否返回每一层的输出, 否则只返回最后一层。

    Input
    -----
    (B, T, C, H, W) (batch_first=True) 或 (T, B, C, H, W) (batch_first=False)

    Output
    ------
    (layer_output_list, last_state_list)
        layer_output_list: 每层输出堆叠 (B, T, hidden_dim, H, W) 的列表
        last_state_list:   每层末态 [h, c] 的列表
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # 让 kernel_size 与 hidden_dim 都成为长度为 num_layers 的列表
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            # (T, B, C, H, W) -> (B, T, C, H, W)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class conv_block(nn.Module):
    """(Conv3x3-BN-ReLU) x2, 解码器基础块 (与 unet.py 同构)."""

    def __init__(self, in_ch: int, out_ch: int):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class up_conv(nn.Module):
    """最近邻上采样 x2 + Conv3x3-BN-ReLU, 解码器上采样块 (与 unet.py 同构)."""

    def __init__(self, in_ch: int, out_ch: int):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


def _pool_seq(x: torch.Tensor, pool: nn.Module) -> torch.Tensor:
    """对 (B, T, C, H, W) 序列逐帧做空间下采样, 保持时间维不变."""
    b, t, c, h, w = x.shape
    y = pool(x.reshape(b * t, c, h, w))
    return y.reshape(b, t, y.shape[1], y.shape[2], y.shape[3])


class ConvLSTM_Seg(nn.Module):
    """
    ConvLSTM 变化检测网络: ConvLSTM 编码器 + U-Net 风格解码器。

    编码器每级先池化再做 ConvLSTM, 取**最后时刻**的隐状态作为该级空间跳跃特征;
    解码器沿用 unet.py 的 up_conv + concat 跳跃 + conv_block 结构, 输出单通道 logits。

    Parameters
    ----------
    in_ch: int
        输入通道数 (本项目 = len(factor_list) + 6)。
    out_ch: int
        输出通道数 (二分类变化掩码 = 1)。
    filters: tuple
        各级 ConvLSTM 的隐状态通道数, 长度固定为 4 (对应 128/64/32/16 四级)。
    kernel_size: int
        卷积核尺寸, 各层共用。
    num_layers: int
        每级 ConvLSTM 内部堆叠的层数。
    bias: bool
        卷积是否使用偏置。
    """

    def __init__(self, in_ch: int = 10, out_ch: int = 1, filters=(64, 128, 256, 512),
                 kernel_size: int = 3, num_layers: int = 1, bias: bool = True):
        super(ConvLSTM_Seg, self).__init__()
        filters = tuple(filters)
        assert len(filters) == 4, 'ConvLSTM_Seg 固定 4 个编码阶段, filters 需为长度 4 的序列'
        self.filters = filters

        # 编码器: 4 级 ConvLSTM (128 -> 64 -> 32 -> 16), 每级之间按需池化
        dims = (in_ch,) + filters
        self.enc = nn.ModuleList([
            ConvLSTM(input_dim=dims[i], hidden_dim=dims[i + 1], kernel_size=(kernel_size, kernel_size),
                     num_layers=num_layers, batch_first=True, bias=bias)
            for i in range(len(filters))
        ])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 瓶颈: 16 -> 8
        self.bottleneck = ConvLSTM(input_dim=filters[-1], hidden_dim=filters[-1],
                                   kernel_size=(kernel_size, kernel_size),
                                   num_layers=num_layers, batch_first=True, bias=bias)

        # 解码器: 逐级上采样 + concat 跳跃 + conv_block
        self.up4 = up_conv(filters[3], filters[2])
        self.dec4 = conv_block(filters[3] + filters[2], filters[2])
        self.up3 = up_conv(filters[2], filters[1])
        self.dec3 = conv_block(filters[2] + filters[1], filters[1])
        self.up2 = up_conv(filters[1], filters[0])
        self.dec2 = conv_block(filters[1] + filters[0], filters[0])
        self.up1 = up_conv(filters[0], filters[0])
        self.dec1 = conv_block(filters[0] * 2, filters[0])

        self.head = nn.Conv2d(filters[0], out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(1)  # (B, C, H, W) -> (B, 1, C, H, W)

        seq = x
        skips = []
        for i, enc in enumerate(self.enc):
            if i > 0:
                seq = _pool_seq(seq, self.pool)
            outputs, _ = enc(seq)          # ConvLSTM 返回 (layer_output_list, last_state_list)
            seq = outputs[-1]              # (B, T, filters[i], H, W), 只保留末层输出
            skips.append(seq[:, -1])       # 取最后时刻作为空间跳跃特征

        seq = _pool_seq(seq, self.pool)
        outputs, _ = self.bottleneck(seq)
        x = outputs[-1][:, -1]

        x = self.dec4(torch.cat([self.up4(x), skips[3]], dim=1))   # 16 x 16
        x = self.dec3(torch.cat([self.up3(x), skips[2]], dim=1))   # 32 x 32
        x = self.dec2(torch.cat([self.up2(x), skips[1]], dim=1))   # 64 x 64
        x = self.dec1(torch.cat([self.up1(x), skips[0]], dim=1))   # 128 x 128

        return self.head(x)
