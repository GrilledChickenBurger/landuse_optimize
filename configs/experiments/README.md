# experiments 配置归档

按实验类型归档各专用优化配置。**当前为"归档 + 保留旧路径兼容"过渡态**：
运行中的代码仍引用 `configs/` 根下的旧路径（如 `configs/optimizerConfig_stability.yaml`），
本目录为同一配置的归档副本；待各实验批次收尾、引用统一迁移后，旧路径可移除。

## 目录

- `stability/`  稳定性实验（10 样本 × default × 5 重复）
  - `optimizerConfig_stability.yaml`    SA 稳定性（transunet/year15, vsp=0.8, T_end=50）
  - `optimizerConfig_stability_ga.yaml` GA 稳定性
- `ablation/`   SA 消融 A/B/C 三组（guided / 无空间约束 / 随机区域约束）
  - `optimizerConfig_ablation.yaml`     SA 消融专用配置
- `tornado/`    扰动（tornado）敏感性实验
  - `optimizerConfig_tornado_{common,sa,ga}.yaml`

## 备注

- 稳定性分片脚本已改为直接使用根目录的 `configs/optimizerConfig_stability*.yaml` 与 `configs/score_cfg.yaml`。
- `configs/optimizerConfig_ablation.yaml` 与 `configs/optimizerConfig_tornado_common.yaml` 已按本目录归档副本补齐到根路径；ablation / tornado 分片脚本可直接运行。若修改本目录归档副本，需同步根路径副本。
