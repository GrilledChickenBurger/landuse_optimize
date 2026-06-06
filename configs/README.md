# Configs

## 这个目录做什么

项目的所有 pipeline 都通过 YAML 配置文件驱动——没有 CLI 参数解析，所有参数和路径都写在配置文件中。每个配置文件对应一个 pipeline 阶段，由 `main.py` 中对应的入口函数加载。配置通过 `utils/config_utils.get_yaml()` 加载为 `argparse.Namespace`，支持点号访问（如 `config.data.batch_size`）。

## 配置文件与 Pipeline 的对应关系

| Pipeline 阶段                | 入口函数                    | 配置文件                                  | 说明                                 |
| ---------------------------- | --------------------------- | ----------------------------------------- | ------------------------------------ |
| Step 0: 计算驱动因子统计参数 | `run_statistic_pipeline()`  | `statisticConfig.yaml`                    | 为训练/优化阶段的归一化提供均值/方差 |
| Step 1: 训练变化检测模型     | `run_train_pipeline()`      | `trainConfig.yaml`                        | 训练 ANN 模型                        |
| Step 1: 验证已训练模型       | `run_valid_pipeline()`      | `trainConfig.yaml`                        | 复用训练配置                         |
| Step 2a: 单样本 SA/GA 优化   | `run_runner_pipeline()`     | `optimizerConfig.yaml`                    | 运行优化器                           |
| Step 2b: 批量 GA 策略实验    | `run_experiment_pipeline()` | `optimizerConfig.yaml`                    | 遍历 GA 变体                         |
| 消融实验                     | `run_ablation_pipeline()`   | `optimizerConfig.yaml` + `score_cfg.yaml` | 敏感性扫描                           |

```
statisticConfig → [计算归一化参数]
                        ↓
trainConfig     → [训练 ANN 模型] → pred_diff
                        ↓
optimizerConfig → [SA/GA 搜索最优方案] → 最优 LU 方案
                        ↑
score_cfg       → [消融实验：调整目标函数参数]
```

## 配置加载机制

- **`get_yaml(path)`**：加载 YAML 文件，递归将 dict 转为 `argparse.Namespace`
- **点号访问**：`config.data.factor_list`、`config.training.model_name`
- **路径模板**：含 `{year}`、`{model_name}`、`{run_timestamp}` 的字符串由各 pipeline 在运行时 `.format()`
- **score_cfg 的特殊处理**：
  - `get_default_score_cfg()` 返回硬编码默认值（与 YAML 一致），供非消融 pipeline 使用
  - `apply_const_overrides(score_cfg)` 可在消融时覆写 `libs/const.py` 的模块级常量（如 `PIXEL_NUM_THRESHOLD_ISCHANGE`、`WATER_SUPPLY_PARAM`）

## trainConfig.yaml — 训练配置

对应 Stage-1：训练 ANN 变化检测模型。

### device

| 字段      | 值     | 说明                |
| --------- | ------ | ------------------- |
| `use_gpu` | `true` | 使用 CUDA（如可用） |

### data

| 字段           | 值                        | 说明                                                              |
| -------------- | ------------------------- | ----------------------------------------------------------------- |
| `root`         | `"_data\training_data"`   | 训练数据根目录                                                    |
| `split`        | `"train"`                 | 数据集划分（`train`/`val`）                                       |
| `shape`        | `[1, 128, 128]`           | 张量形状 `[通道数, 高, 宽]`                                       |
| `factor_list`  | `["DEM","DL","GDP","RK"]` | 驱动因子列表，模型输入通道数 = len(factor_list) + 6（LU one-hot） |
| `pack`         | `[[10, 15]]`              | 训练数据时间范围 `[start_year, end_year]`，`[-1,-1]` 表示全部     |
| `lu_labels`    | 19 个 LU 类编码           | 不含 `0`（nodata），与 optimizerConfig 不同                       |
| `shuffle`      | `true`                    | 训练时打乱数据                                                    |
| `normalize`    | `true`                    | 归一化驱动因子（使用 statisticConfig 计算的均值/方差）            |
| `batch_size`   | `8`                       | 批次大小                                                          |
| `only_changed` | `false`                   | 是否仅使用有变化的样本                                            |

### training

| 字段                  | 值                                           | 说明                                                             |
| --------------------- | -------------------------------------------- | ---------------------------------------------------------------- |
| `model_name`          | `"transunet"`                                | 模型类型：`attunet` / `unet` / `fcn` / `segformer` / `transunet` |
| `save_frequency`      | `20`                                         | 每 N 个 epoch 保存 checkpoint                                    |
| `print_frequency`     | `50`                                         | 每 N 个 batch 打印损失                                           |
| `visualize_frequency` | `200`                                        | 每 N 个 batch 可视化输出                                         |
| `num_epochs`          | `100`                                        | 总训练轮数                                                       |
| `is_tensorboard`      | `true`                                       | 启用 TensorBoard                                                 |
| `log_frequency`       | `4`                                          | TB 日志记录频率（每 N 个 batch）                                 |
| `log_dir`             | `"logs\{model_name}\{run_timestamp}"`        | TB 日志路径模板                                                  |
| `checkpoint_dir`      | `"checkpoints\{model_name}\{run_timestamp}"` | Checkpoint 路径模板                                              |

### optimizer

训练代码根据 `model_name` 选择优化器配置：`getattr(config.optimizer, model_type, config.optimizer.default)`

| 字段            | default  | segformer | 说明         |
| --------------- | -------- | --------- | ------------ |
| `name`          | `"adam"` | `"adamw"` | 优化器类型   |
| `learning_rate` | `1e-4`   | `5e-5`    | 学习率       |
| `weight_decay`  | `1e-5`   | `1e-4`    | L2 正则      |
| `beta_1`        | `0.9`    | `0.9`     | Adam β1      |
| `amsgrad`       | `false`  | `false`   | AMSGrad 变体 |
| `epsilon`       | `1e-8`   | `1e-8`    | Adam ε       |
| `gradient_clip` | `1.0`    | `1.0`     | 梯度裁剪范数 |

### validation

| 字段         | 值           | 说明             |
| ------------ | ------------ | ---------------- |
| `split`      | `"val"`      | 验证集划分       |
| `pack`       | `[[10, 15]]` | 验证数据时间范围 |
| `batch_size` | `1`          | 验证批次大小     |
| `shuffle`    | `false`      | 验证不打乱       |

### model

模型架构参数，仅在对应模型类型被选中时使用。

**segformer**：

| 字段              | 值                | 说明                      |
| ----------------- | ----------------- | ------------------------- |
| `dims`            | `[32,64,160,256]` | 各 stage 通道宽度         |
| `heads`           | `[1,2,5,8]`       | 各 stage 注意力头数       |
| `ff_expansion`    | `[4,4,4,4]`       | 各 stage FFN 扩展比       |
| `reduction_ratio` | `[8,4,2,1]`       | 各 stage SR 比率          |
| `num_layers`      | `2`               | 每 stage Transformer 块数 |
| `decoder_dim`     | `128`             | 解码器嵌入维度            |

**transunet**：

| 字段           | 值    | 说明                 |
| -------------- | ----- | -------------------- |
| `img_dim`      | `128` | 输入 H/W             |
| `out_channels` | `32`  | 首层通道数           |
| `head_num`     | `4`   | 注意力头数           |
| `mlp_dim`      | `512` | Transformer MLP 维度 |
| `block_num`    | `4`   | Transformer 块数     |
| `patch_dim`    | `16`  | Patch 大小           |

## optimizerConfig.yaml — 优化配置

对应 Stage-2：SA/GA 搜索最优 LU 方案。是所有配置中最复杂的文件。

### device / data

与 trainConfig 类似，但有以下差异：

- `root`：`"_data\optimize_data"`（优化阶段数据，包含驱动因子 + ES 数据 + LU）
- 新增 `es_list`：`["RAIN","NJW","ZSL","PODU","WATER","CITY","GD"]` — 生态系统服务数据列表
- 新增 `year`：`"10"` — 基准年份，用于路径模板
- `lu_labels` 包含 `0`（nodata），共 20 个编码
- 新增 `maxent_model_path`：`"checkpoints\maxent-model_sklearn_{year}.ela"`
- `shuffle` 为 `false`，`batch_size` 为 `1`

### model

与 trainConfig 不同，这里指定的是**已训练好的模型**，用于生成 pred_diff：

| 字段                      | 值                                      | 说明                       |
| ------------------------- | --------------------------------------- | -------------------------- |
| `type`                    | `"attunet"`                             | 模型类型                   |
| `model_path`              | `"checkpoints\attunet_best_260212.pth"` | 预训练权重路径             |
| `segformer` / `transunet` | 同 trainConfig                          | 架构参数（模型创建时需要） |

### runner

| 字段            | 值          | 说明                                                                     |
| --------------- | ----------- | ------------------------------------------------------------------------ |
| `type`          | `"GA"`      | 优化器类型：`"SA"` 或 `"GA"`                                             |
| `use_landscape` | `true`      | 是否启用景观破碎化约束                                                   |
| `optimize_num`  | `3`         | 每次运行优化的样本数                                                     |
| `scenario`      | `"default"` | 场景类型，可选 `"default"`/`"enterprise"`/`"farmer"`/`"tourist"`/`"all"` |

**场景与 ES 权重**：

不同场景代表不同的利益相关者视角，通过 `runner.weight.<scenario>` 指定四项 ES 的加权权重：

| 场景       | food | carbon | water | culture | 解读                     |
| ---------- | ---- | ------ | ----- | ------- | ------------------------ |
| default    | 0.25 | 0.25   | 0.25  | 0.25    | 均衡发展                 |
| enterprise | 0.0  | 0.5    | 0.5   | 0.0     | 企业视角：碳汇与水源优先 |
| farmer     | 0.8  | 0.1    | 0.1   | 0.0     | 农民视角：粮食生产优先   |
| tourist    | 0.0  | 0.0    | 0.0   | 1.0     | 游客视角：文化服务优先   |

**common_params**（优化器共享资源路径）：

| 字段                | 值                                              | 说明                                                    |
| ------------------- | ----------------------------------------------- | ------------------------------------------------------- |
| `carbon_pool_path`  | `"libs\carbon_pool.csv"`                        | 碳库查找表                                              |
| `maxent_model_path` | `"checkpoints\maxent-model_sklearn_{year}.ela"` | 需与 `data.maxent_model_path` 一致                      |
| `frg_model_path`    | `"libs\frg_model.fca"`                          | 破碎化模型                                              |
| `maxent_list`       | `["CITY","DL","LU","PODU","WATER"]`             | MaxEnt 输入变量列表，其中 `"LU"` 使用当前方案的 LU 类型 |

### SA — 模拟退火参数

| 字段                  | 值             | 说明               |
| --------------------- | -------------- | ------------------ |
| `T_start`             | `100`          | 初始温度           |
| `T_end`               | `0.1`          | 终止温度           |
| `cooling_rate`        | `0.9995`       | 几何降温率         |
| `max_noimprove_iter`  | `1000`         | 连续无改进早停阈值 |
| `max_beststuck_iter`  | `3000`         | 最优解停滞早停阈值 |
| `print_frequency`     | `100`          | 每 N 步打印        |
| `visualize_frequency` | `500`          | 每 N 步可视化      |
| `performance`         | `true`         | 记录性能指标       |
| `save_path`           | `"results\SA"` | 结果保存目录       |

### GA — 遗传算法参数

| 字段                  | 值                | 说明                                       |
| --------------------- | ----------------- | ------------------------------------------ |
| `ori_pop_size`        | `30`              | 初始种群大小                               |
| `final_pop_size`      | `8`               | 最终种群大小                               |
| `max_gen`             | `3000`            | 最大迭代代数                               |
| `max_noimprove_gen`   | `100`             | 连续无改进早停阈值                         |
| `max_beststuck_gen`   | `400`             | 最优解停滞早停阈值                         |
| `crossover_rate`      | `0.8`             | 交叉概率                                   |
| `mutation_rate`       | `1.0`             | 变异概率                                   |
| `scale_ratio`         | `[0.2, 0.5, 0.3]` | 初始种群扰动尺度比例 `[small, mid, large]` |
| `print_frequency`     | `10`              | 每 N 代打印                                |
| `visualize_frequency` | `200`             | 每 N 代可视化                              |
| `performance`         | `true`            | 记录性能指标                               |
| `save_path`           | `"results\GA"`    | 结果保存目录                               |

**GA.strategy** — 策略配置键，决定 GA 变体的组装方式：

| 字段                     | 当前值       | 可选值                        | 说明                                       |
| ------------------------ | ------------ | ----------------------------- | ------------------------------------------ |
| `parent_pool`            | `"elite"`    | `"all"` / `"elite"`           | 父本来源：全部种群 or 精英子集             |
| `elite_ratio_for_pool`   | `0.2`        | —                             | `parent_pool=elite` 时，取前 N% 作为父本池 |
| `parent_selection`       | `"roulette"` | `"roulette"` / `"tournament"` | 选择方式                                   |
| `elite_ratio`            | `0.1`        | —                             | 精英保留比例                               |
| `non_crossover_handling` | `"copy"`     | `"copy"` / `"copy_mutate"`    | 未交叉个体的处理                           |
| `elite_mutation`         | `False`      | `True` / `False`              | 精英是否也参与变异                         |

文件末尾注释中记录了 GA1–GA7 七种实验变体的配置，供 `pipelines/experiment.py` 批量实验时参考。

### output

| 字段          | 值                        | 说明             |
| ------------- | ------------------------- | ---------------- |
| `save_result` | `false`                   | 是否保存优化结果 |
| `save_path`   | `"results\optimize_data"` | 结果保存目录     |

## statisticConfig.yaml — 统计配置

对应 Step 0：计算驱动因子的归一化参数（均值/方差），输出 JSON 文件供后续 pipeline 使用。

结构扁平（没有嵌套分组），所有字段在顶层：

| 字段          | 值                              | 说明             |
| ------------- | ------------------------------- | ---------------- |
| `root`        | `"_data\optimize_data"`         | 数据根目录       |
| `shape`       | `[1, 128, 128]`                 | 张量形状         |
| `factor_list` | `["DEM","DL","GDP","RK"]`       | 需统计的驱动因子 |
| `year`        | `"20"`                          | 基准年份         |
| `shuffle`     | `false`                         | 不打乱           |
| `batch_size`  | `8`                             | 批次大小         |
| `result_path` | `"_data\statistic_{year}.json"` | 输出路径模板     |

数据加载使用专用的 `StatisticDataset`（`datasets/base_dataloader.py`）——它只读取驱动因子和源年 LU 的 TIFF，跳过 ES 数据和目标年 LU 的磁盘 I/O。因此该配置不再需要 optimizerConfig 中的 `es_list`、`lu_labels` 等字段。

## score_cfg.yaml — 评分参数配置

**核心思想**：集中管理目标函数和约束中的所有"魔数"，供消融实验覆写。

这个文件与 `calculator/README.md` 中的配置表有重叠，但角度不同：

- calculator/README 讲的是「这些参数在目标函数中怎么用」
- 这里讲的是「这些参数在哪里集中管理、如何被消融实验覆写」

### tgt_function — 目标函数参数

| 字段             | 默认值 | 说明                                    |
| ---------------- | ------ | --------------------------------------- |
| `lambda_ES_high` | `1000` | pred_diff 像素数 > pixel_high 时的 λ_ES |
| `lambda_ES_low`  | `5000` | pred_diff 像素数 < pixel_low 时的 λ_ES  |
| `lambda_LS_high` | `5`    | pred_diff 像素数 > pixel_high 时的 λ_LS |
| `lambda_LS_low`  | `1`    | pred_diff 像素数 < pixel_low 时的 λ_LS  |
| `pixel_high`     | `6000` | λ 分段上界                              |
| `pixel_low`      | `300`  | λ 分段下界                              |

### change_score — 变化得分参数

| 字段                 | 默认值 | 说明                                |
| -------------------- | ------ | ----------------------------------- |
| `inside_reward_coef` | `10`   | 变化区域落入 pred_diff 内的奖励倍率 |

### runner_utils — 优化器工具参数

| 字段                | 默认值 | 说明                                        |
| ------------------- | ------ | ------------------------------------------- |
| `sigmoid_threshold` | `0.5`  | `get_pred_diff()` 中二值化 pred_diff 的阈值 |
| `grid_num`          | `4`    | `get_grid_restrict_area()` 的网格划分数     |

### landscape — 景观得分参数

| 字段          | 默认值 | 说明                                |
| ------------- | ------ | ----------------------------------- |
| `pure_base`   | `0.3`  | pure_score 基准权重                 |
| `pure_change` | `0.2`  | pure_score 随 change_ratio 缩放系数 |

### const — 可覆写的常量

这些字段通过 `libs/const.apply_const_overrides()` 覆写模块级常量：

| 字段                           | 默认值                                     | 对应常量                       | 说明                                                                           |
| ------------------------------ | ------------------------------------------ | ------------------------------ | ------------------------------------------------------------------------------ |
| `pixel_num_threshold_ischange` | `30`                                       | `PIXEL_NUM_THRESHOLD_ISCHANGE` | 变化像素数低于此值视为无变化                                                   |
| `water_supply_param`           | 二级类字典 (11/12/21/.../67 共 25 项)      | `WATER_SUPPLY_PARAM`           | 各 LU 二级类径流系数（林地/草地内部按覆盖度区分；水域 41-46=0，建设用地 51-53=1，未利用地 61-67=1） |

### ablation — 消融实验开关

| 字段                   | 默认值  | 说明                                                         |
| ---------------------- | ------- | ------------------------------------------------------------ |
| `guided`               | `true`  | 设为 `false` 时跳过 `pred_grid_restrict`（无 ANN 引导约束）  |
| `disable_change_score` | `false` | 设为 `true` 时目标函数屏蔽 change_score 项（测试无引导消融） |
| `disable_early_stop`   | `false` | 设为 `true` 时禁用 noimprove/beststuck 早停（用于实验 A）    |

### 默认值一致性约束

YAML 中的值必须与 `utils/config_utils._DEFAULT_SCORE_CFG_DICT` 中的硬编码默认值以及 `calculator/`、`utils/runner_utils.py`、`libs/const.py` 中的硬编码值保持一致。修改一处而不改其他会破坏消融覆写的语义。

## experimentConfig.yaml — 实验配置（归档）

与 optimizerConfig 结构相同但参数不同，目前没有代码引用，属于归档配置。与 optimizerConfig 的差异：

| 字段                               | optimizerConfig | experimentConfig |
| ---------------------------------- | --------------- | ---------------- |
| `SA` 区块                          | 有              | 无               |
| `output` 区块                      | 有              | 无               |
| `GA.ori_pop_size`                  | `30`            | `20`             |
| `GA.max_gen`                       | `3000`          | `1500`           |
| `GA.max_noimprove_gen`             | `100`           | `50`             |
| `GA.max_beststuck_gen`             | `400`           | `200`            |
| `GA.strategy.elite_ratio_for_pool` | `0.2`           | `0.3`            |
| `GA.strategy.elite_ratio`          | `0.1`           | `0.2`            |

可以理解为更小预算、更快跑完的 GA 实验配置，需手动在 `main.py` 中替换配置路径才能使用。

## 配置间的共享与差异

trainConfig 和 optimizerConfig 有多个同名字段但值不同，这是因为两个阶段的数据和运行模式不同：

| 字段              | trainConfig             | optimizerConfig         | 原因                            |
| ----------------- | ----------------------- | ----------------------- | ------------------------------- |
| `data.root`       | `"_data\training_data"` | `"_data\optimize_data"` | 训练用训练集，优化用优化集      |
| `data.lu_labels`  | 19 个编码（无 0）       | 20 个编码（含 0）       | 训练时忽略 nodata，优化时需处理 |
| `data.shuffle`    | `true`                  | `false`                 | 训练需随机，优化需确定性        |
| `data.batch_size` | `8`                     | `1`                     | 训练可批处理，优化逐样本        |
| `data.es_list`    | 无                      | 7 项 ES 数据            | ES 仅在优化阶段使用             |
| `model` 的角色    | 指定要训练的模型类型    | 指定已训练好的模型路径  | 阶段不同：训练 vs 推理          |
