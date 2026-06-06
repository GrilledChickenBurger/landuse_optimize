# ablation 实验特有工具: 随机 mask 生成
import os
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
from scipy import ndimage
from collections import deque

from utils.io_utils import ensure_folder


def make_random_connected_mask(pred_diff: torch.Tensor,
                               hard_restrict_base: torch.Tensor,
                               seed: int) -> torch.Tensor:
    """生成与 pred_diff 连通块数和面积都匹配的随机 0/1 mask, 限制在 hard_restrict_base 内."""
    rng = np.random.RandomState(seed)

    pd_np = pred_diff.detach().cpu().numpy().squeeze().astype(bool)
    base_np = hard_restrict_base.detach().cpu().numpy().squeeze().astype(bool)

    structure = np.ones((3, 3), dtype=bool)
    labeled, n_blocks = ndimage.label(pd_np, structure=structure)
    if n_blocks == 0:
        return torch.zeros_like(pred_diff, dtype=torch.bool)
    areas = [int((labeled == i).sum()) for i in range(1, n_blocks + 1)]
    areas.sort(reverse=True)

    H, W = pd_np.shape
    out = np.zeros((H, W), dtype=bool)

    for target_area in areas:
        candidates = np.argwhere(base_np & (~out))
        if len(candidates) == 0:
            break
        seed_idx = rng.randint(0, len(candidates))
        sy, sx = candidates[seed_idx]

        filled = 0
        queue = deque([(sy, sx)])
        visited_local = set([(sy, sx)])
        while queue and filled < target_area:
            y, x = queue.popleft()
            if not (0 <= y < H and 0 <= x < W):
                continue
            if out[y, x] or not base_np[y, x]:
                continue
            out[y, x] = True
            filled += 1
            # 使用 rng 的 permutation 替代全局 _py_random.shuffle，避免污染全局随机状态
            neighbor_order = rng.permutation(8)
            neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            for ni in neighbor_order:
                dy, dx = neighbor_offsets[ni]
                ny, nx = y + dy, x + dx
                if (ny, nx) not in visited_local and 0 <= ny < H and 0 <= nx < W:
                    visited_local.add((ny, nx))
                    queue.append((ny, nx))

    mask_t = torch.tensor(out, dtype=torch.bool).reshape(1, 1, H, W)
    return mask_t
