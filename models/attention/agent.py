# Agent Attention: On the Integration of Softmax and Linear Attention
# 论文: ECCV 2024 (arXiv:2312.08874)
# 来源: LeapLabTHU/Agent-Attention 官方实现的核心机制。
# 差异: 官方实现含有分辨率相关的相对位置 bias (an_bias / na_bias / ah_bias / aw_bias),
#       本项目 token 网格固定为 grid_size x grid_size, 故省略这些 bias,
#       只保留 "代理生成 -> 代理聚合 -> 代理广播" 的两步注意力核心。

import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentAttention(nn.Module):
    """用 M 个代理 token 把标准注意力拆成两步, 复杂度由 O(N^2) 降到 O(N*M):

    1) 代理聚合: 代理作为 query 对全体 K/V 做注意力, 汇总全局信息 (Attn(A, K, V) + A);
    2) 代理广播: 各 token 以聚合后的代理为 K/V 做注意力, 取回全局信息 (Attn(Q, A, A))。

    代理 token 由输入经深度卷积增强局部信息后, 沿 token 维自适应均值池化生成。
    """

    def __init__(self, dim, num_heads, grid_size=0, num_agent=16,
                 attn_dropout=0.0, vis=False, **kwargs):
        super().__init__()
        if num_agent < 1:
            raise ValueError(f"num_agent 必须为正整数, 当前为 {num_agent}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.grid_size = grid_size
        self.num_agent = num_agent
        self.vis = vis

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dwc = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(attn_dropout)

    def _to_agent(self, x):
        """[B, h, N, d] -> [B, h, M, d], 沿 token 维做自适应均值池化 (不要求 N 整除 M)。"""
        B, h, N, d = x.shape
        x = x.reshape(B * h, N, d).transpose(1, 2)      # [B*h, d, N]
        x = F.adaptive_avg_pool1d(x, self.num_agent)    # [B*h, d, M]
        return x.transpose(1, 2).reshape(B, h, self.num_agent, d)

    def forward(self, hidden_states):
        B, N, C = hidden_states.shape

        qkv = self.qkv(hidden_states).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)            # [B, h, N, d]

        # 代理生成: 深度卷积增强局部信息, 再沿 token 维池化成 M 个代理
        x = hidden_states
        g = self.grid_size
        if g > 0 and g * g == N:
            x = hidden_states.transpose(1, 2).reshape(B, C, g, g)
            x = self.dwc(x).reshape(B, C, N).transpose(1, 2)
        agent = self._to_agent(x.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3))

        # 步骤 1: 代理聚合 Attn(A, K, V) + A
        attn = (agent @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_dropout(attn.softmax(dim=-1))
        agent = attn @ v + agent

        # 步骤 2: 代理广播 Attn(Q, A, A)
        attn = (q @ agent.transpose(-2, -1)) * self.scale
        weights = attn.softmax(dim=-1)
        attn = self.attn_dropout(weights)

        out = (attn @ agent).transpose(1, 2).reshape(B, N, C)
        out = self.proj_dropout(self.proj(out))
        return out, (weights if self.vis else None)
