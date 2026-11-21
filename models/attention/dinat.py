# DiNAT: Dilated Neighborhood Attention Transformer
# 论文: Dilated Neighborhood Attention Transformer (Hassani et al., 2022, arXiv:2209.15001)
# 来源: SHI-Labs/Neighborhood-Attention-Transformer 的 dilated 变体。
# 差异: 同 NeighborhoodAttention, 用纯 PyTorch 空洞邻域 mask 实现, 无 CUDA 依赖。

from .neighborhood import NeighborhoodAttention


class DilatedNeighborhoodAttention(NeighborhoodAttention):
    """空洞邻域注意力 (DiNAT)。

    在 Neighborhood Attention 基础上, 邻域窗口内以 dilation 步长采样 (而非连续像素),
    从而在 token 数不变的前提下指数级扩大感受野。dilation 默认 2。
    """

    def __init__(self, dim, num_heads, grid_size=0, attn_dropout=0.0, vis=False,
                 kernel_size=3, dilation=2, **kwargs):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            grid_size=grid_size,
            attn_dropout=attn_dropout,
            vis=vis,
            kernel_size=kernel_size,
            dilation=dilation,
            **kwargs,
        )
