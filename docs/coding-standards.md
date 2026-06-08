# 项目代码规范

> 本规范覆盖整个 LU 优化系统的代码组织、可复用性、注释与跨层交互约定，作为后续添加代码的统一指导。

---

## 1. 项目层次架构

项目严格分为六层，每层只与相邻层交互，禁止跨层直接调用：

```
configs/          YAML 配置（Namespace）
  ↓
pipelines/        流程编排：加载配置 → 准备数据 → 调用下层 → 保存结果
  ↓
datasets/         数据加载：Dataset → DataLoader → data_pack (list)
models/           ANN 模型定义：create_model() 工厂统一创建
runner/           优化器封装：create_runner() 工厂统一创建
  ↓
calculator/       目标函数计算：纯函数，无状态
  ↓
utils/            工具函数：按功能细分模块
libs/             常量、颜色映射、碳库查找表
```

### 1.1 各层职责边界

| 层级 | 职责 | 禁止做的事 |
|------|------|-----------|
| **pipelines** | 编排工作流：读取 YAML、构造 DataLoader、循环调用 runner、保存 summary | 不直接实现模型结构、目标函数、优化算法 |
| **datasets** | 从磁盘加载 TIFF → 归一化/one-hot → 返回 `data_pack` | 不调用 model、runner、calculator |
| **models** | 定义 `nn.Module` 子类，实现 `forward()` | 不读取配置、不保存结果 |
| **runner** | 封装 SA/GA 完整运行逻辑，对外暴露 `run()` → 返回 `best_plan` | 不重新训练 ANN、不直接修改 `libs.const` |
| **calculator** | 计算目标函数各项得分（change/es/landscape） | 不保存文件、不修改全局状态 |
| **utils** | 提供通用工具函数 | 不引入层间循环依赖（如 utils 不 import pipelines） |
| **libs** | 定义常量、颜色映射、查找表 | 不引入业务逻辑 |

### 1.2 工厂函数约定

- **模型**：`models/__init__.py::create_model(model_type, in_ch, out_ch, model_config=None)`
- **Runner**：`runner/__init__.py::create_runner(runner_type, config, common_config, model_config, device, data_dict, index, save_path, ...)`

所有工厂接收 `**model_config` 透传，内部支持 `argparse.Namespace` 和 `dict` 两种格式。

---

## 2. 配置与数据传递

### 2.1 配置是 Namespace

- 所有 YAML 通过 `get_yaml(path)` 递归转为 `argparse.Namespace`，使用**点号访问**：`config.data.factor_list`
- 深层逻辑中**禁止**直接操作 `dict`。如果必须转换，使用 `vars(ns)` 或 `hasattr/getattr`
- 默认配置兜底：`get_default_score_cfg()` 在 `score_cfg=None` 时提供与旧硬编码一致的默认值

### 2.2 data_pack 格式约定

`data_pack` 是 `List`，索引位置固定，**不按字典键名访问**：

| 阶段 | 索引 0 | 索引 1 | 索引 2 | 索引 3 | 索引 4 | 索引 5 |
|------|--------|--------|--------|--------|--------|--------|
| **训练** | `factors` | `srcLU` (one-hot) | `diff` | `idx_label` | `changed_flag` | — |
| **优化** | `factors` | `restrict` (RA) | `es_data` | `srcLU` | `tgtLU` | `idx_label` |

- `factors` 形状：`(factor_num, H, W)` 或 `(factor_num-2, H, W)`（优化阶段不含 RA/LU）
- `srcLU` one-hot 通道数固定为 **6**（6 类简化 LU）
- 模型输入：`torch.cat([factors_src, lu_src], dim=1)` → `in_ch = len(factor_list) + 6`

### 2.3 process_batch 约定

Pipeline 层的 `process_batch` 负责将 `data_pack`（list）转为 `data_dict`（dict）：

```python
def process_batch(data_pack, config) -> Dict[str, torch.Tensor]:
    factors = torch.tensor(data_pack[0], dtype=torch.float32)
    factor_dict = get_data_dict(factors, config.data.factor_list)
    ...
    return final_data_dict   # 键名与 factor_list / es_list 对应
```

- 所有张量保持 4D：`(1, C, H, W)` 或 `(1, 1, H, W)`
- `data_dict` 必须包含的键：`ori_plan`, `final_plan`, `factors`, `restrict`，以及各因子键名

---

## 3. 代码可复用规范

### 3.1 DRY 原则

类似功能必须提取内部辅助函数，禁止复制粘贴。典型场景：

- 两个 CSV 写出函数如果逻辑只差扫描的目录不同 → 提取 `_write_*_core(groups, sort_key)`
- 多个后处理函数扫描同一目录 → 提取 `_load_*_data()` 统一预加载
- 多个 calculator 子函数需要相同的预处理 → 提取到 `calculator_utils.py`

### 3.2 优先复用已有工具

新增代码前，先检查 `utils/` 中是否已有类似功能：

| 需求 | 已有工具 |
|------|---------|
| 创建目录 | `utils.io_utils.ensure_folder()` |
| 读取 YAML 为 Namespace | `utils.config_utils.get_yaml()` |
| 按点号路径写配置 | `utils.tools._set_by_path()` |
| 读取 `convergence.json` 字段 | `utils.exp_sweep_utils._read_conv_metric()` |
| 加载 ANN 模型 | `utils.runner_utils.load_ANN_model()` |
| 生成 `pred_diff` | `utils.runner_utils.get_pred_diff()` |
| 保存 TIFF + performance JSON | `utils.runner_utils.save_plan_and_get_performance()` |
| 计算 ES 性能指标 | `utils.performance.write_performance_score()` |
| 代表性样本 n_pred/level 映射 | `utils.exp_sweep_utils._load_representative_meta()` |

### 3.3 配置覆写规范（消融/参数扫描实验）

```python
# 1. 深拷贝基础配置
score_cfg = copy.deepcopy(base_score_cfg)

# 2. 按点号路径覆写
for dotted_path, v in exp['overrides'].items():
    _set_by_path(score_cfg, dotted_path, v)

# 3. 同步到模块级常量
apply_const_overrides(score_cfg)

# ... 运行实验 ...

# 4. 实验结束后恢复，防止跨实验状态泄漏
restore_const_defaults()
```

**强制要求**：每个实验覆写后，必须在实验循环结束后调用 `restore_const_defaults()`。

---

## 4. 模型与训练规范

### 4.1 模型类命名

- 主模型：**PascalCase**，如 `AttU_Net`、`Segformer`、`TransUNet`
- 辅助模块：**snake_case**，如 `conv_block`、`up_conv`、`attention_block`

### 4.2 前向传播

- 不在 `forward()` 末尾加 Sigmoid（由外部 `F.sigmoid()` 或 `BCEWithLogitsLoss` 处理）
- `Segformer` / `TransUNet` 需在 `forward` 内用 `F.interpolate` 上采样回原尺寸

### 4.3 Checkpoint 格式

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': epoch_loss,
    'config': config,          # 完整 Namespace，用于断点续训
    'timestamp': run_timestamp,
}
```

### 4.4 训练循环结构

```python
for epoch in range(start_epoch, epochs):
    model.train()
    for i_batch, data_pack in enumerate(train_dataloader):
        loss, outputs, diff = process_batch(data_pack, model, device)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
    # 验证 + 保存 best / checkpoint
```

---

## 5. Runner 规范

### 5.1 统一初始化签名

```python
create_runner(
    runner_type='SA',                # 'SA' | 'GA'
    config=type_config,              # SA/GA 专属参数 (Namespace)
    common_config=runner_config.common_params,
    model_config=opt_cfg.model,      # ANN 模型配置
    device=device,
    data_dict=data_dict,             # Dict[str, torch.Tensor]
    index='10_764_5',                # str
    save_path=save_path,
    weight_dict=weight_dict,         # ES 权重
    use_landscape=False,
    guided=True,
    disable_early_stop=False,
    score_cfg=score_cfg,             # Namespace
    random_restrict_mask=None,       # Optional[torch.Tensor]
)
```

### 5.2 `run()` 方法结构

```python
def run(self):
    # 1. 提前退出：无变化区域
    if self.pred_diff_pixelnum < PIXEL_NUM_THRESHOLD_ISCHANGE:
        return self.ori_plan

    # 2. 主循环（SA: 温度下降；GA: 逐代进化）
    ...

    # 3. 保存结果（与 SA/GA 无关的通用逻辑）
    rUtils.save_plan_and_get_performance(...)
    self._dump_convergence(...)
    return self.best_plan
```

### 5.3 结果保存职责边界

**Runner 内部保存**：
- `plans_{idx}/`：best_plan.TIF、ori_plan.TIF、tgt_plan.TIF
- `convergence.json`：`final_best`、`es_score`、`iter_to_95pct`、`wall_time_sec`、`total_iter`
- `performance.json`：Change_area / ES / Landscape / FINAL 四项指标
- `total_score_list.npy`：逐迭代目标函数值
- `history/` / `charts/`：可视化图

**通用保存函数**：`utils/runner_utils.save_plan_and_get_performance()` 统一调用 `utils.performance.write_performance_score()` 和 `save_all_plans()`。

### 5.4 避免 runner 内部重复计算

如果迭代过程中已保存分项值（如 `self.es_score_list`），`_dump_convergence` 应**直接读取**，不再重新调用 `cal_tgt_function`：

```python
# SA
es_score = float(self.es_score_list[self.best_score_iter])

# GA
es_score = float(self.es_score_list[self.best_score_generation])
```

---

## 6. Calculator 规范

### 6.1 纯函数原则

- `calculator/` 下的所有函数必须是**纯函数**：无全局状态修改、无文件 I/O、无副作用。
- 外部状态（如 `info_dict`、`score_cfg`）通过参数传入。

### 6.2 函数签名风格

```python
def cal_tgt_function(cur_plan: torch.Tensor,
                     cur_diff: torch.Tensor,
                     pred_diff: torch.Tensor,
                     ori_plan: torch.Tensor,
                     info_dict: Dict[str, Any],
                     es_weight_dict: Dict[str, float],
                     use_landscape: bool = False,
                     T_ratio: float = 0.0,
                     score_cfg: Optional[Namespace] = None,
                     change_scale_n: Optional[int] = None
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float], float]:
```

- 返回元组需按固定顺序：`(tgt_func, change_score, es_score, es_dict, landscape_score)`
- `score_cfg` 为 `Namespace`，`None` 时由 `get_default_score_cfg()` 兜底

### 6.3 模块分工

| 文件 | 职责 |
|------|------|
| `tgt_function_cal.py` | 总目标函数组装，lambda 自适应调度 |
| `cal_change_score.py` | 变化得分（距离地图、连通块分析） |
| `cal_es_score.py` | ES 得分（粮食、碳汇、水源、文化服务） |
| `cal_landscape_score.py` | 景观破碎化得分 |

---

## 7. 数据记录与后处理规范

### 7.1 原始记录表字段一致性

`stability_runs.csv` / `ablation_runs.csv` 等原始记录表的字段必须与 `docs/ablation/字段标准.md` 保持一致：

```
sample_id, n_pred, level, algorithm, config, run_idx, F_final, ESV, Runtime, T95
```

### 7.2 convergence.json 作为最小记录单元

所有 run 级核心指标优先写入 `convergence.json`，后处理脚本从中提取，不直接耦合 runner 内部状态：

| 字段 | 含义 |
|------|------|
| `final_best` | 最终综合目标函数值 F(S) |
| `es_score` | 最终生态系统服务得分 ESV |
| `wall_time_sec` | 总运行时间 Runtime |
| `iter_to_95pct` | 达到最优值 95% 的迭代步数 T95 |
| `total_iter` | 总迭代数 |

### 7.3 后处理函数避免重复扫描

多个后处理函数如果扫描同一目录，必须提取统一的预加载函数：

```python
def _load_ablation_data(ablation_root: str):
    """统一扫描 A/B/C 三组，返回 (per_group, samples)，供多个 table 函数复用。"""
    ...
```

---

## 8. 注释与文档规范

### 8.1 函数 docstring

- **公共函数**：必须有 docstring，说明参数、返回值、副作用。类型注解推荐但非强制。
- **内部辅助函数**（`_` 开头）：至少有一行说明用途。

```python
def build_stability_table(ablation_root: str, exp_name: str, save_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """跨样本聚合单个实验组的统计指标, 输出 csv.

    Parameters:
    ----------
    ablation_root : str
        结果根目录
    exp_name : str
        实验组名, 如 'A_guided'
    save_path : str, optional
        输出路径, 默认 <ablation_root>/plots/stability_table_<exp_name>.csv

    Returns:
    --------
    List[Dict[str, Any]]
        每行是一个样本的聚合统计字典
    """
```

### 8.2 Pipeline 文档分层

| 位置 | 内容 | 不涉及 |
|------|------|--------|
| `docs/` | 逻辑梳理、字段标准、数据整合流程、操作手册 | 代码细节 |
| `pipelines/docs/` | 每个 pipeline/模式的内部方法逻辑、数据流、关键覆写点 | 项目整体介绍 |
| `pipelines/README.md` | 项目整体逻辑联系、模块依赖关系、入口函数索引 | 代码实现细节 |

### 8.3 修改旧代码时同步更新

如果重构了函数行为（如合并重复逻辑、改名、改返回值），必须同步更新：
- 函数自身的 docstring
- 所有调用处的注释（如有特殊说明）
- 相关文档（如 `pipelines/README.md` 中的引用说明）

---

## 9. 新增代码审查清单

在提交新增代码前，逐项检查：

- [ ] **层次正确**：新增代码位于正确的层次（pipeline / runner / calculator / utils / models / datasets）？
- [ ] **DRY**：是否提取了与已有代码重复的逻辑？
- [ ] **避免重复 I/O**：是否对同一目录/数据重复扫描？
- [ ] **避免重复计算**：是否在 runner 内部引入了不必要的重计算？
- [ ] **字段一致**：新增函数的字段/列名是否与 `docs/ablation/字段标准.md` 一致？
- [ ] **状态恢复**：配置覆写后是否调用了 `restore_const_defaults()`？
- [ ] **工具复用**：是否优先复用了 `utils/` 中的已有工具函数？
- [ ] **文档同步**：是否在 `pipelines/README.md` 或对应 `pipelines/docs/` 文件中补充了说明？
- [ ] **纯函数检查**：`calculator/` 下的新函数是否无副作用、无文件 I/O？
- [ ] **类型检查**：`data_pack` 是否按索引位置解包（而非字典键名）？

---

## 10. Git 远程推送与代理

### 10.1 代理配置

本机 Git 配置了 HTTP/HTTPS 代理 `127.0.0.1:7890`（Clash/机场类工具）：

```bash
git config --global http.proxy  http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890
```

### 10.2 常见问题：代理不可用导致推送失败

如果代理工具未启动，`git push`/`git pull`/`git fetch` 会报错：

```
fatal: unable to access 'https://github.com/...': 
Failed to connect to 127.0.0.1 port 7890 after 2059 ms: Couldn't connect to server
```

**排查步骤：**

```bash
# 1. 确认代理是否配置
git config --global --get http.proxy
git config --global --get https.proxy

# 2. 确认代理是否在运行
curl -x http://127.0.0.1:7890 -I https://github.com 2>&1 | head -3
```

### 10.3 临时绕过代理推送

如果代理暂不可用但需要立即推送，可以临时取消代理，推送后恢复：

```bash
# 取消代理
git config --global --unset http.proxy
git config --global --unset https.proxy

# 执行推送
git push origin master

# 推送完成后立即恢复代理
git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890
```

> ⚠️ **注意**：推送完成后务必恢复代理设置，否则后续 `git fetch`/`git pull` 等操作会因网络不通而失败。
