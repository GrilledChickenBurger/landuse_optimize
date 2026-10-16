"""
编码器输出端信息交互模块族。

本族模块以布尔开关接入 TransUNet 编码器输出端 (decoder 之前), 并联新增, 不替换既有主链路。
统一作用位置: VisionTransformer.forward 中 transformer 输出之后、decoder 之前。
统一接口: forward(x) -> [B, N, C]
  x   : [B, N, C] 的 token 序列 (与编码器注意力族一致)
  out : [B, N, C], 形状与输入一致

包含:
  m1 - FactorLUCroAttention, 驱动因子 × LU 交叉注意力 (新增方案 3.5)
"""

from .m1 import FactorLUCroAttention
from .m1_input import InputFactorLUCrossAttn

__all__ = ["FactorLUCroAttention", "InputFactorLUCrossAttn"]
