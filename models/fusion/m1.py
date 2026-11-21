"""M1: 驱动因子 × LU 交叉注意力模块.

在 TransUNet 编码器输出端 (decoder 之前) 并联接入:
  1. 将编码器输出的 token 特征投影为因子分支 F 与 LU 分支 L;
  2. 以 F 为 query、L 为 key/value 做交叉注意力, 建模"给定驱动条件下各地类的变化倾向";
  3. 经可学习门控与残差融合, 输出形状与输入一致 [B, N, C].

门控 gate 初始化为 0, 训练初期该模块为恒等映射 (严格等价于原 TransUNet),
训练中逐步开启, 保证从已有起点稳定学习.
"""
import torch
import torch.nn as nn


class FactorLUCroAttention(nn.Module):
    """驱动因子 × LU 交叉注意力.

    forward(x) -> x + gate * CrossAttn(factor_proj(x), lu_proj(x), lu_proj(x))
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) 必须能被 num_heads ({num_heads}) 整除")
        self.factor_proj = nn.Linear(dim, dim)  # 驱动因子分支 (query)
        self.lu_proj = nn.Linear(dim, dim)      # LU 状态分支 (key/value)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        # 可学习门控, 初始化为 0 → 恒等映射, 训练中逐步开启
        self.gate = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, C] -> [B, N, C]"""
        F = self.factor_proj(x)                 # query
        L = self.lu_proj(x)                     # key/value
        attn_out, _ = self.cross_attn(F, L, L)  # [B, N, C]
        return x + self.gate * attn_out
