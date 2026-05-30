# step0: 训练前的驱动因子分布参数统计
# 输出 JSON 文件供后续训练/优化阶段的归一化使用
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from typing import Dict, Any

import tqdm
import torch
from torch.utils.data import DataLoader

from datasets.base_dataloader import get_dataset
from libs.const import ALL_LOG_FACTORS
from utils.tools import get_yaml


class RunningStats:
    """驱动因子均值/方差的批量 Welford 累加器。

    `update` 接收任意形状张量（可带 mask），按批合并到运行统计量中；合并公式为
    精确等价于逐像素累加的 Chan/Welford 并行版本。
    """
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: torch.Tensor, mask: torch.Tensor = None):
        if mask is not None:
            x = x[mask]
        if x.numel() == 0:
            return
        x = x.reshape(-1).float()
        n_new = x.numel()
        new_mean = x.mean().item()
        new_var = x.var(unbiased=False).item() if n_new > 1 else 0.0

        delta = new_mean - self.mean
        total_n = self.n + n_new
        self.mean = (self.n * self.mean + n_new * new_mean) / total_n
        self.M2 += new_var * n_new + delta * delta * self.n * n_new / total_n
        self.n = total_n

    def finalize(self):
        if self.n <= 1:
            return self.mean, 0.0
        var = self.M2 / (self.n - 1)
        return self.mean, var ** 0.5


def get_params(config_path: str) -> Dict[str, Any]:
    """
    计算各驱动因子的均值/方差并保存到 JSON，供后续训练/优化阶段的归一化使用。

    统计时会跳过 nodata 像素（src_lu == 0），并对 ALL_LOG_FACTORS 中的因子取 log1p。

    Parameters:
    ----------
    config_path : str
        statisticConfig.yaml 的路径，需包含以下键：
        - root: 数据根目录
        - shape: 张量形状 [C, H, W]
        - factor_list: 需统计的驱动因子列表
        - year: 基准年份
        - batch_size: 批次大小
        - shuffle: 是否打乱数据
        - result_path: 输出路径模板，支持 "{year}" 占位符

    Returns:
    --------
    params : Dict[str, Any]
        每个因子的统计参数字典，格式为:
        {factor_name: {"n": 有效像素数, "miu": 均值, "epsilon": 标准差 + 1e-6}}
        同时写入 result_path 指定的 JSON 文件。
    """
    config = get_yaml(config_path)
    dataset = get_dataset(type="statistic", data_config=config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=config.shuffle)

    stats = {name: RunningStats() for name in config.factor_list}

    for data in tqdm.tqdm(dataloader, desc="Computing factor statistics"):
        factors = data["factors"].float()                       # (B, F, H, W)
        nodata_mask = (data["src_lu"].long() != 0).squeeze(1)   # (B, H, W), True=有效

        for i, name in enumerate(config.factor_list):
            x = factors[:, i]                                   # (B, H, W)
            if name in ALL_LOG_FACTORS:
                x = torch.log1p(x)
            stats[name].update(x, mask=nodata_mask)

    params = {}
    for name, rs in stats.items():
        miu, std = rs.finalize()
        params[name] = {
            "n": rs.n,
            "miu": float(miu),
            "epsilon": float(std + 1e-6),
        }
    print(params)

    result_path = config.result_path.format(year=config.year)
    with open(result_path, "w") as f:
        json.dump(params, f, indent=4)

    return params


if __name__ == "__main__":
    get_params("configs/statisticConfig.yaml")
