"""
TransUNet 编码器注意力的可插拔模块库。

通过 attn_type 配置开关, 选择替换 TransUNet 编码器中的标准多头自注意力 (MHSA)。
统一接口: forward(x) -> (out, weights)
  x       : [B, N, C] 的 token 序列
  out     : [B, N, C], 形状与输入一致, 可直接替换
  weights : vis=True 时返回 [B, h, N, N] 注意力权重, 否则为 None

支持的类型:
  vanilla - 官方 TransUNet 的 MHSA (默认; 参数命名与官方一致, 历史 checkpoint 可继续加载)
  agent   - Agent Attention (ECCV 2024), 代理 token 两步注意力
  swinv2  - Swin Transformer V2 (CVPR 2022), 缩放余弦注意力 + Log-CPB 相对位置偏置
"""

import torch.nn as nn

from .vanilla import VanillaAttention
from .agent import AgentAttention
from .swin_v2 import ScaledCosineAttention

# attn_type -> 实现类
ATTENTION_REGISTRY = {
    "vanilla": VanillaAttention,
    "agent": AgentAttention,
    "swinv2": ScaledCosineAttention,
}


def build_attention(attn_type: str,
                    dim: int,
                    num_heads: int,
                    grid_size: int = 0,
                    attn_dropout: float = 0.0,
                    vis: bool = False,
                    **kwargs) -> nn.Module:
    """按 attn_type 构造注意力模块。

    Parameters
    ----------
    attn_type : str, ATTENTION_REGISTRY 的键
    dim : int, token 通道数 (TransUNet 中为 config.hidden_size)
    num_heads : int, 注意力头数
    grid_size : int, token 网格边长 (混合编码器 16 倍下采样后为 img_size // 16)
    attn_dropout : float, 注意力 dropout
    vis : bool, 是否返回注意力权重
    kwargs : 具体模块的专属超参 (如 agent 的 num_agent)
    """
    if attn_type not in ATTENTION_REGISTRY:
        raise ValueError(
            f"不支持的 attn_type: {attn_type!r}, 可选值为 {sorted(ATTENTION_REGISTRY)}"
        )
    return ATTENTION_REGISTRY[attn_type](
        dim=dim,
        num_heads=num_heads,
        grid_size=grid_size,
        attn_dropout=attn_dropout,
        vis=vis,
        **kwargs,
    )
