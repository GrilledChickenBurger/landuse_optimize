# 混合训练集训练 pipeline
# 遍历 attunet/unet/segformer/transunet 四个模型, 依次使用 HybridDataset 训练。
# 验证集使用 val split 的 15-20 时间对; 复用 train.py 的 process_batch / visualize_outputs
# 与 training_utils 的优化器/损失工具, 不修改 train.py.
import os
import math
import time
import json
import random
import shutil
import argparse
import hashlib
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils import tensorboard
from torch.utils.data import DataLoader

from models import create_model
from datasets.base_dataloader import get_dataset
from datasets.hybrid_dataloader import HybridDataset
from utils.tools import ensure_folder, get_yaml
from utils.training_utils import (
    get_optimizer, ExpoAverageMeter, cal_loss, MetricsMeter,
    get_original_loss_fn,
)
from pipelines.train import process_batch, visualize_outputs


def _seed_everything(seed: int) -> None:
    """固定 python / numpy / torch(cpu+cuda) 全局随机源.

    必须在任何消耗随机数的动作之前调用, 尤其要早于:
      1) HybridDataset(...) 构造 —— 其数据增强的随机翻转
         (RandomHorizontalFlip/RandomVerticalFlip) 在构造期即施加, 消耗 torch 全局 RNG;
      2) create_model(...) —— 权重初始化消耗 torch 全局 RNG.
    多模型串行训练时, 若不在每个模型前重置, 后一个模型的初始 RNG 状态取决于前一个模型的
    消耗量, 导致换 model_list 顺序或子集后结果漂移.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _is_better(score: float, best: float, higher_is_better: bool) -> bool:
    """统一选点比较: higher_is_better=True 取更大值, 否则取更小值."""
    return score > best if higher_is_better else score < best


def _metric_score(metrics: dict, select_metric: str) -> float:
    """从 MetricsMeter 指标字典按选点准则取分值 (仅支持指标类准则)."""
    if select_metric.startswith('f1'):
        return float(metrics.get('f1_score', 0.0))
    if select_metric.startswith('iou'):
        return float(metrics.get('iou', 0.0))
    raise ValueError("select_metric=%s 不支持从指标字典取分值." % select_metric)


def _resolve_loss_fn(config, model_type: str):
    """按 config.training.loss_type 解析损失函数.

    - "original":   各模型逐一对齐原始文献/官方实现
                    (attunet → Sørensen-Dice; unet → 加权 CE + weight map;
                     segformer → 未加权 CE ≡ BCE; transunet → 0.5*CE + 0.5*Dice)
    - "focal_dice": 统一 Focal + Dice 各 0.5, 保证多模型结构公平对比
    """
    loss_type = getattr(config.training, 'loss_type', 'focal_dice')
    if loss_type == 'original':
        return get_original_loss_fn(model_type)
    if loss_type == 'focal_dice':
        return cal_loss
    raise ValueError(f"未知的 training.loss_type: {loss_type} (可选 original / focal_dice).")


def _loss_name(config, model_type: str) -> str:
    """返回归档用的损失函数名称."""
    loss_type = getattr(config.training, 'loss_type', 'focal_dice')
    if loss_type == 'original':
        return f"original({get_original_loss_fn(model_type).__name__})"
    return f"{loss_type}({cal_loss.__name__})"


def _build_val_dataset(config) -> 'torch.utils.data.Dataset':
    """构造验证集: val split + validation.pack (默认 15-20)."""
    val_config = config.validation
    val_data_config = argparse.Namespace(**vars(config.data))
    val_data_config.split = val_config.split
    val_data_config.pack = val_config.pack
    return get_dataset(type="training", data_config=val_data_config)


def _write_model_report(result: dict, model_type: str, checkpoint_dir: str,
                        run_timestamp: str, train_stats: dict, config,
                        train_seed: Optional[int] = None) -> str:
    """把单模型验证指标 + 说明文档归档到 results/_model_performance/{model}/{ts}/ 下.

    同时写 metrics.json / info.json / README.md 三份文件, 返回归档目录.
    """
    report_dir = getattr(config, 'result', None)
    if report_dir is None:
        report_dir = os.path.join("results", "_model_performance", model_type, run_timestamp)
    else:
        report_dir = report_dir.model_perf_dir.format(model_name=model_type, run_timestamp=run_timestamp)
    ensure_folder(report_dir)

    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

    # 训练配置 (只保留可 JSON 序列化的基础类型)
    optimizer_config = _resolve_optimizer_config(config, model_type)
    model_cfg = getattr(config.model, model_type, None)
    training_config = {
        'model': model_type,
        'profile': getattr(config, 'profile', 'server'),
        'factor_list': config.data.factor_list,
        'batch_size': _model_batch_size(config, model_type),
        'num_epochs': config.training.num_epochs,
        'loss': _loss_name(config, model_type),
        'train_size': config.hybrid.train_size,
        'ratio_1015': config.hybrid.ratio_1015,
        'ratio_1520': config.hybrid.ratio_1520,
        'changed_unchanged_ratio': config.hybrid.changed_unchanged_ratio,
        'seed': config.hybrid.seed,  # 数据抽样/增强种子
        'train_seed': train_seed if train_seed is not None else int(getattr(config.hybrid, 'train_seed', 0)),
        'select_metric': str(getattr(config.training, 'select_metric', 'val_loss')),
        'attn_type': getattr(model_cfg, 'attn_type', None),
        'use_m1': getattr(model_cfg, 'use_m1', None),
        'optimizer': {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                      for k, v in vars(optimizer_config).items()},
    }
    val_dataset_info = {
        'split': config.validation.split,
        'pack': config.validation.pack,
    }

    # 1. metrics.json (验证指标)
    metrics = dict(result)
    metrics['model'] = model_type
    metrics['checkpoint'] = best_model_path
    metrics['run_timestamp'] = run_timestamp
    metrics['data_composition'] = train_stats
    with open(os.path.join(report_dir, "metrics.json"), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    # 2. info.json (结构化说明)
    info = {
        'model': model_type,
        'checkpoint': best_model_path,
        'run_timestamp': run_timestamp,
        'training_config': training_config,
        'data_composition': train_stats,
        'val_dataset': val_dataset_info,
    }
    with open(os.path.join(report_dir, "info.json"), 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=4, ensure_ascii=False)

    # 3. README.md (人可读)
    float_metrics = {k: v for k, v in metrics.items()
                     if isinstance(v, (int, float)) and not isinstance(v, bool)}
    metric_lines = "\n".join(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |"
                             for k, v in float_metrics.items())
    md = (
        f"# {model_type} 验证报告\n\n"
        f"- 运行时间戳: `{run_timestamp}`\n"
        f"- 模型 checkpoint: `{best_model_path}`\n\n"
        f"## 训练配置\n\n"
        f"- 因子列表: `{training_config['factor_list']}`\n"
        f"- batch_size: `{training_config['batch_size']}`\n"
        f"- num_epochs: `{training_config['num_epochs']}`\n"
        f"- 损失函数: `{training_config['loss']}`\n"
        f"- 目标训练集样本量 N (train_size): `{training_config['train_size']}`\n"
        f"- 10-15 变化样本加入比例 r1: `{training_config['ratio_1015']}`\n"
        f"- 15-20 变化样本加入比例 r2: `{training_config['ratio_1520']}`\n"
        f"- 变化:非变化 目标比例 r_cu: `{training_config['changed_unchanged_ratio']}`\n"
        f"- 随机种子(数据抽样/增强): `{training_config['seed']}`\n"
        f"- 训练种子(权重初始化/翻转/批顺序): `{training_config['train_seed']}`\n"
        f"- checkpoint 选点准则: `{training_config['select_metric']}`\n\n"
        f"## 训练集数据构成 (来源与比例)\n\n"
        f"- 池子容量 10-15: 变化 {train_stats['n_changed_1015_pool']}, 未变化 {train_stats['n_unchanged_1015_pool']}\n"
        f"- 池子容量 15-20: 变化 {train_stats['n_changed_1520_pool']}, 未变化 {train_stats['n_unchanged_1520_pool']}\n"
        f"- 抽样原始变化样本: 10-15 {train_stats['n_raw_1015']} + 15-20 {train_stats['n_raw_1520']} "
        f"= {train_stats['n_raw_changed']}\n"
        f"- 目标变化总数: {train_stats['target_changed']} "
        f"(10-15 {train_stats['n_changed_aug_1015']} / 15-20 {train_stats['n_changed_aug_1520']})\n"
        f"- 配额增强: 10-15 每样本 +{train_stats['augment_base_1015']} 份 (另 {train_stats['augment_rem_1015']} 个 +1), "
        f"15-20 每样本 +{train_stats['augment_base_1520']} 份 (另 {train_stats['augment_rem_1520']} 个 +1)\n"
        f"- 未变化样本: 10-15 {train_stats['n_unchanged_1015']} + 15-20 {train_stats['n_unchanged_1520']} "
        f"= {train_stats['n_unchanged']}\n"
        f"- 增强后变化样本: {train_stats['n_changed_aug']}\n"
        f"- 最终训练集总数: {train_stats['n_actual']} (目标 {train_stats['train_size']}, "
        f"偏差 {train_stats['deviation']:+.2%})\n\n"
        f"## 验证集\n\n"
        f"- split: `{val_dataset_info['split']}`, pack: `{val_dataset_info['pack']}`\n\n"
        f"## 验证指标\n\n"
        f"| 指标 | 值 |\n|---|---|\n{metric_lines}\n"
    )
    with open(os.path.join(report_dir, "README.md"), 'w', encoding='utf-8') as f:
        f.write(md)

    return report_dir


def _model_batch_size(config, model_type: str) -> int:
    """按「机器档位 × 模型」解析训练 batch_size.

    先取 data.batch_size_overrides.{profile}.{model}, 缺失时回退默认 data.batch_size.
    profile 由顶层 config.profile 指定 (local / server).
    """
    profile = getattr(config, 'profile', 'server')
    overrides = getattr(config.data, 'batch_size_overrides', None)
    group = getattr(overrides, profile, None) if overrides is not None else None
    if group is not None:
        return getattr(group, model_type, config.data.batch_size)
    return config.data.batch_size


def _resolve_num_workers(config) -> int:
    """按机器档位解析 DataLoader 进程数: data.num_workers_overrides.{profile}.

    本机(Windows) 走 spawn, 建 worker 需把整个内存数据集序列化过去(实测 ~8.5s/DataLoader),
    而本机 batch 只有 1~8、__getitem__ 仅做索引, worker 无收益, 故配置里给 0;
    服务器(Linux fork) 零成本共享数据集, 给 4 把 collate/pin 挪出训练关键路径.
    未配置时回退: 本机 0, 其它(Linux) 4.
    """
    default = 0 if os.name == 'nt' else 4
    overrides = getattr(config.data, 'num_workers_overrides', None)
    if overrides is None:
        return default
    return int(getattr(overrides, getattr(config, 'profile', 'server'), default))


def _init_worker(worker_id: int):
    """DataLoader 子进程初始化: 把 torch 线程压到 1.

    主进程线程数已按 TORCH_NUM_THREADS 限制, 但 fork 出的 worker 会各自继承一份;
    若 worker 仍开多线程, 多进程 × 多线程会远超容器 CPU 配额, 引发 OpenMP 自旋抢核.
    """
    torch.set_num_threads(1)


def _apply_model_overrides(config, overrides: Optional[Dict[str, Dict]]) -> None:
    """把 {模型名: {字段: 值}} 的覆盖项写入 config.model.<模型名>.

    用于在配置文件不变的前提下跑同一模型的变体 (如 TransUNet 的三种 attn_type)。
    """
    for model_name, fields in (overrides or {}).items():
        model_cfg = getattr(config.model, model_name, None)
        if model_cfg is None:
            raise ValueError(f"配置中没有 model.{model_name}, 无法覆盖字段 {list(fields)}")
        for key, value in fields.items():
            setattr(model_cfg, key, value)


def _apply_hparam_overrides(config, model_list: List[str],
                            batch_size_override: Optional[int],
                            learning_rate_override: Optional[float]) -> None:
    """把档位专属 batch_size / learning_rate 写回 config (仅内存, 不落盘).

    用于模块扫描时给单个档位 (如显存占用大、易 OOM 的 transunet 变体) 单独降 batch / 调 lr:
      - batch_size 覆盖 data.batch_size_overrides.{profile}.{model};
      - learning_rate 覆盖 optimizer.{profile}.{model}.learning_rate.
    仅影响本次 hybrid_train_pipeline 调用, 不修改配置文件.
    """
    if batch_size_override is None and learning_rate_override is None:
        return
    profile = getattr(config, 'profile', 'server')
    bs_overrides = getattr(config.data, 'batch_size_overrides', None)
    bs_group = getattr(bs_overrides, profile, None) if bs_overrides is not None else None
    opt_group = getattr(getattr(config, 'optimizer', None), profile, None)
    for model_type in model_list:
        if batch_size_override is not None and bs_group is not None:
            setattr(bs_group, model_type, int(batch_size_override))
        if learning_rate_override is not None and opt_group is not None:
            model_opt = getattr(opt_group, model_type, None) or getattr(opt_group, 'default', None)
            if model_opt is not None:
                model_opt.learning_rate = float(learning_rate_override)


def _resolve_optimizer_config(config, model_type: str):
    """按机器档位解析优化器配置: optimizer.{profile}.{model}, 缺失时回退该档的 default."""
    profile = getattr(config, 'profile', 'server')
    group = getattr(config.optimizer, profile)
    return getattr(group, model_type, group.default)


def _apply_lr(optimizer, base_lrs, step, warmup_steps, decay_steps, min_lr, use_cosine):
    """按全局步 step 设置学习率: 线性 warmup → cosine 衰减到 min_lr.

    warmup 窗口内 lr 由 base/warmup_steps 线性升至 base; 窗口后若启用 cosine
    则按剩余步数余弦衰减到 min_lr, 否则保持峰值 lr.
    """
    if warmup_steps > 0 and step < warmup_steps:
        factor = (step + 1) / warmup_steps
        lrs = [b * factor for b in base_lrs]
    elif use_cosine:
        p = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
        lrs = [min_lr + (b - min_lr) * 0.5 * (1.0 + math.cos(math.pi * p)) for b in base_lrs]
    else:
        lrs = base_lrs
    for group, lr in zip(optimizer.param_groups, lrs):
        group['lr'] = lr


def _train_single_model(config, model_type: str, device: torch.device, run_timestamp: str,
                        checkpoint_path: Optional[str] = None,
                        train_seed: Optional[int] = None) -> dict:
    """使用混合训练集训练单个模型."""
    data_config = config.data
    train_config = config.training
    hybrid_config = config.hybrid
    optimizer_config = _resolve_optimizer_config(config, model_type)

    # 训练种子: 必须在 HybridDataset(构造期做随机翻转增强)与 create_model(权重初始化)之前固定,
    # 否则两轮之间增强结果与初始权重都不可复现.
    if train_seed is None:
        train_seed = int(getattr(hybrid_config, 'train_seed', 0))
    _seed_everything(train_seed)
    print(f"\n================= 开始训练模型【{model_type}】 (train_seed={train_seed}) =================")

    # 1. 训练集(混合) + 验证集(val + 15-20)
    train_dataset = HybridDataset(
        root=data_config.root,
        shape=data_config.shape,
        condition_list=data_config.factor_list,
        train_size=hybrid_config.train_size,
        ratio_1015=hybrid_config.ratio_1015,
        ratio_1520=hybrid_config.ratio_1520,
        changed_unchanged_ratio=hybrid_config.changed_unchanged_ratio,
        normalize=data_config.normalize,
        seed=hybrid_config.seed,
    )
    # num_workers/pin_memory 仅并行化数据搬运, 不改变 batch 内容与顺序
    # (数据集在构造期即固定、__getitem__ 只做索引, 完全确定性)
    num_workers = _resolve_num_workers(config)
    dl_extra = dict(num_workers=num_workers,
                    pin_memory=device.type == "cuda",
                    persistent_workers=num_workers > 0,
                    worker_init_fn=_init_worker)
    print(f"DataLoader: num_workers={num_workers}, pin_memory={dl_extra['pin_memory']}")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=_model_batch_size(config, model_type),
        shuffle=data_config.shuffle,
        **dl_extra,
    )
    val_dataset = _build_val_dataset(config)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.validation.batch_size,
        shuffle=config.validation.shuffle,
        **dl_extra,
    )

    # 2. 模型 + 优化器
    model = create_model(
        model_type=model_type,
        in_ch=len(data_config.factor_list) + 6,  # 增加 6 通道 LU one-hot
        out_ch=1,
        model_config=getattr(config.model, model_type, {}),
    ).to(device)
    optimizer = get_optimizer(
        name=optimizer_config.name,
        parameters=model.parameters(),
        learning_rate=optimizer_config.learning_rate,
        weight_decay=optimizer_config.weight_decay,
        beta_1=optimizer_config.beta_1,
        amsgrad=optimizer_config.amsgrad,
        epsilon=optimizer_config.epsilon,
    )

    # 2.1 学习率调度: 前 warmup_epochs 线性 warmup, 之后 cosine 衰减到 min_lr (按总步数)
    warmup_epochs = getattr(train_config, 'warmup_epochs', 0)
    lr_sched_cfg = getattr(train_config, 'lr_scheduler', None)
    use_cosine = lr_sched_cfg is not None and getattr(lr_sched_cfg, 'name', 'none') == 'cosine'
    min_lr = getattr(lr_sched_cfg, 'min_lr', 1e-6) if lr_sched_cfg is not None else 1e-6
    # 峰值学习率取配置值(而非优化器当前值), 保证断点续训时调度进度不受影响
    base_lrs = [optimizer_config.learning_rate] * len(optimizer.param_groups)
    steps_per_epoch = len(train_dataloader)
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = max(train_config.num_epochs * steps_per_epoch, warmup_steps + 1)
    decay_steps = max(total_steps - warmup_steps, 1)
    print(f"学习率调度: warmup {warmup_epochs} epoch ({warmup_steps} 步)"
          + (f" + cosine → min_lr={min_lr} (总 {total_steps} 步)" if use_cosine else " (无 cosine)"))

    # 损失函数: 由 config.training.loss_type 决定 (original / focal_dice)
    loss_fn = _resolve_loss_fn(config, model_type)
    print(f"损失函数: {_loss_name(config, model_type)}")

    # 3. 输出目录 (model_name 不同自动分目录)
    checkpoint_dir = train_config.checkpoint_dir.format(model_name=model_type, run_timestamp=run_timestamp)
    log_dir = train_config.log_dir.format(model_name=model_type, run_timestamp=run_timestamp)
    ensure_folder(checkpoint_dir)
    ensure_folder(log_dir)
    writer = tensorboard.SummaryWriter(log_dir=log_dir) if train_config.is_tensorboard else None

    # 4. 训练循环初始化
    epochs = train_config.num_epochs
    # 选点准则: "f1_0.5" 按验证 F1@0.5 选最优(越高越好); "val_loss" 回退旧口径(越低越好).
    select_metric = str(getattr(train_config, 'select_metric', 'val_loss')).lower()
    if select_metric not in ('f1_0.5', 'f1', 'iou', 'iou_0.5', 'val_loss'):
        raise ValueError(f"未知的 training.select_metric: {select_metric}")
    higher_is_better = select_metric != 'val_loss'
    best_score = -float('inf') if higher_is_better else float('inf')
    best_val_loss = float('inf')  # 历史最小 val_loss, 仅用于归档/日志, 不参与选点
    start_epoch = 0
    epochs_since_improvement = 0
    tensorboard_step = 0

    # 4.1 断点续训设置 (与 train.py 的 training_pipeline 保持一致)
    if checkpoint_path:
        print(f"================= 开始加载checkpoint: {checkpoint_path}... =================")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint.get('epochs_since_improvement', 0)
        # 选点准则必须与 checkpoint 一致, 否则新旧口径混用会导致 best_model 误覆盖.
        ckpt_metric = str(checkpoint.get('select_metric', 'val_loss')).lower()
        if ckpt_metric != select_metric:
            raise ValueError(
                f"checkpoint 选点准则({ckpt_metric})与当前配置({select_metric})不一致, "
                f"不支持跨准则 resume; 请统一配置, 或删除 {checkpoint_path} 重新训练."
            )
        # 恢复历史最优分值.
        if 'best_score' in checkpoint:
            best_score = checkpoint['best_score']
        elif 'best_val_loss' in checkpoint:
            # 兼容仅存 best_val_loss 的旧版 checkpoint (仅 val_loss 准则下等价).
            best_score = checkpoint['best_val_loss']
        else:
            # 缺历史最优: 备份已有 best_model.pth, 并用当前权重跑一遍验证建立基线,
            # 避免 resume 后第一个(可能更差的)epoch 立刻覆盖历史最优.
            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
            backup_path = os.path.join(checkpoint_dir, "best_model_pre_resume.pth")
            if os.path.exists(best_model_path) and not os.path.exists(backup_path):
                shutil.copyfile(best_model_path, backup_path)
                print(f"[warn] checkpoint 无历史最优字段, 历史最优已备份为 {backup_path}")
            print("[warn] checkpoint 无历史最优字段, 用当前权重跑一遍验证以建立选点基线...")
            model.eval()
            baseline_loss = ExpoAverageMeter()
            baseline_metric = MetricsMeter(device=device, threshold=0.5, accumulate_counts=True)
            with torch.no_grad():
                for data_pack in val_dataloader:
                    loss, outputs, diff = process_batch(data_pack, model, device, loss_fn=loss_fn)
                    baseline_loss.update(loss.item())
                    baseline_metric.update(torch.sigmoid(outputs), diff)
            if higher_is_better:
                best_score = _metric_score(baseline_metric.get_metrics(), select_metric)
            else:
                best_score = baseline_loss.avg
            print(f"[warn] 选点基线 {select_metric}={best_score:.4f}")
        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']
        tensorboard_step = start_epoch * len(train_dataloader)
        run_timestamp = checkpoint.get('timestamp', run_timestamp)
        checkpoint_dir = os.path.dirname(checkpoint_path)
        print(f"从epoch {start_epoch}继续训练，全局步数: {tensorboard_step}")

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = ExpoAverageMeter()
        batch_time = ExpoAverageMeter()
        start = time.time()

        for i_batch, data_pack in enumerate(train_dataloader):
            loss, outputs, diff = process_batch(data_pack, model, device, loss_fn=loss_fn)
            optimizer.zero_grad()
            loss.backward()
            try:
                nn.utils.clip_grad_norm_(model.parameters(), optimizer_config.gradient_clip)
            except AttributeError:
                pass
            optimizer.step()

            # 学习率调度: 线性 warmup + cosine 衰减 (按全局步数)
            _apply_lr(optimizer, base_lrs, tensorboard_step,
                      warmup_steps, decay_steps, min_lr, use_cosine)

            train_loss.update(loss.item())
            batch_time.update(time.time() - start)
            start = time.time()

            if writer:
                writer.add_scalar('Loss/train_batch', loss.item(), tensorboard_step)
            if tensorboard_step % train_config.print_frequency == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{i_batch}/{len(train_dataloader)}], '
                      f'Loss: {train_loss.val:.4f}({train_loss.avg:.4f}), Batch Time: {batch_time.val:.3f}({batch_time.avg:.3f})')
            if tensorboard_step % train_config.visualize_frequency == 0:
                visualize_outputs(outputs, diff, epoch=epoch, step=tensorboard_step,
                                  save_path=os.path.join(log_dir, f"epoch_{epoch+1}_step_{tensorboard_step}.png"))
            tensorboard_step += 1

        epoch_loss = train_loss.avg
        print(f'\nEpoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss:.4f}')
        if writer:
            writer.add_scalar('Loss/train_epoch', epoch_loss, (epoch + 1) * len(train_dataloader))

        # 验证: 同时统计 loss 与 F1/IoU@0.5 (后者用于选点)
        model.eval()
        val_loss = ExpoAverageMeter()
        val_metric = MetricsMeter(device=device, threshold=0.5, accumulate_counts=True)
        with torch.no_grad():
            for data_pack in val_dataloader:
                loss, outputs, diff = process_batch(data_pack, model, device, loss_fn=loss_fn)
                val_loss.update(loss.item())
                val_metric.update(torch.sigmoid(outputs), diff)
        val_epoch_loss = val_loss.avg
        best_val_loss = min(best_val_loss, val_epoch_loss)
        val_metrics = val_metric.get_metrics()
        val_score = _metric_score(val_metrics, select_metric) if higher_is_better else val_epoch_loss
        print(f'Epoch [{epoch+1}/{epochs}], Val Loss: {val_epoch_loss:.4f}, '
              f'F1@0.5: {val_metrics["f1_score"]:.4f}, IoU@0.5: {val_metrics["iou"]:.4f}\n')
        if writer:
            writer.add_scalar('Loss/val_epoch', val_epoch_loss, (epoch + 1) * len(train_dataloader))
            writer.add_scalar('F1/val', val_metrics['f1_score'], (epoch + 1) * len(train_dataloader))
            writer.add_scalar('IoU/val', val_metrics['iou'], (epoch + 1) * len(train_dataloader))
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], (epoch + 1) * len(train_dataloader))

        # 保存 best / checkpoint: 按 select_metric 选点 (f1_0.5 越高越好; val_loss 越低越好)
        if _is_better(val_score, best_score, higher_is_better):
            best_score = val_score
            epochs_since_improvement = 0
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
            print(f"Best model saved ({select_metric}: {best_score:.4f}, Val Loss: {val_epoch_loss:.4f})")
        else:
            epochs_since_improvement += 1
        print(f"Epochs since improvement: {epochs_since_improvement}")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            'epochs_since_improvement': epochs_since_improvement,
            'select_metric': select_metric,
            'best_score': best_score,
            'best_val_loss': best_val_loss,
            'config': config,
            'timestamp': run_timestamp,
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, "latest_checkpoint.pth"))

        if (epoch + 1) % train_config.save_frequency == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth"))

    if writer:
        writer.close()

    # 5. 训练结束: 加载 best_model.pth 做最终验证, 结果落盘 val_hybrid.json
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    val_loss = ExpoAverageMeter()
    metric = MetricsMeter(device=device, threshold=0.5)
    with torch.no_grad():
        for data_pack in val_dataloader:
            loss, outputs, diff = process_batch(data_pack, model, device, loss_fn=loss_fn)
            val_loss.update(loss.item())
            metric.update(torch.sigmoid(outputs), diff)

    final_val_loss = val_loss.avg
    result = {
        'model': model_type,
        'val_loss': round(final_val_loss, 6),
        'checkpoint_dir': checkpoint_dir,
        'best_model_path': best_model_path,
        'stats': train_dataset.stats,
    }
    result.update(metric.get_metrics())

    val_json_path = os.path.join(checkpoint_dir, "val_hybrid.json")
    with open(val_json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print(f"验证结果已保存: {val_json_path}")

    # 归档到 results/_model_performance/{model}/{ts}/ (含 metrics/info/README)
    report_dir = _write_model_report(
        result=result,
        model_type=model_type,
        checkpoint_dir=checkpoint_dir,
        run_timestamp=run_timestamp,
        train_stats=train_dataset.stats,
        config=config,
        train_seed=train_seed,
    )
    print(f"验证报告已归档: {report_dir}")

    print(f"模型【{model_type}】验证完成: Val Loss={final_val_loss:.4f}, "
          f"F1={result.get('f1_score', 0):.4f}, IoU={result.get('iou', 0):.4f}")

    print(f"\n================= 模型【{model_type}】训练完成! =================")
    return result


def hybrid_train_pipeline(config_path: str, model_list: Optional[List[str]] = None,
                          resume_timestamp: Optional[str] = None,
                          ratio_1015_override: Optional[float] = None,
                          ratio_1520_override: Optional[float] = None,
                          batch_size_override: Optional[int] = None,
                          learning_rate_override: Optional[float] = None,
                          model_config_overrides: Optional[Dict[str, Dict]] = None) -> None:
    """遍历模型列表, 依次用混合训练集训练.

    Parameters
    ----------
    config_path: str
        配置文件路径 (如 "configs\\hybridTrainConfig.yaml").
    model_list: List[str], optional
        要训练的模型列表; None 则使用 config.hybrid.model_list.
    resume_timestamp: str, optional
        断点续训: 传入上次训练的 run_timestamp (如 "20260212_045750"),
        则从 checkpoints/{model_name}/{resume_timestamp}/latest_checkpoint.pth 续训.
    ratio_1015_override: float, optional
        仅内存覆盖 config.hybrid.ratio_1015 (不落盘), 用于 r1 扫描.
    ratio_1520_override: float, optional
        仅内存覆盖 config.hybrid.ratio_1520 (不落盘), 用于 r2 扫描.
    batch_size_override: int, optional
        仅内存覆盖本档所有模型的训练 batch_size (不落盘), 用于给易 OOM 的档单独降 batch.
    learning_rate_override: float, optional
        仅内存覆盖本档所有模型的峰值 learning_rate (不落盘), 常与 batch_size_override 搭配缩放.
    model_config_overrides: Dict[str, Dict], optional
        按模型覆盖 config.model.<模型名> 的字段 (不落盘), 如
        {"transunet": {"attn_type": "agent"}}, 用于跑同一模型的注意力变体.
    """
    config = get_yaml(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() and config.device.use_gpu else "cpu")
    print(f"使用设备: {device}")

    if ratio_1015_override is not None:
        config.hybrid.ratio_1015 = ratio_1015_override  # 仅内存覆盖, 不落盘
    if ratio_1520_override is not None:
        config.hybrid.ratio_1520 = ratio_1520_override  # 仅内存覆盖, 不落盘
    _apply_model_overrides(config, model_config_overrides)
    model_list = model_list or config.hybrid.model_list
    _apply_hparam_overrides(config, model_list, batch_size_override, learning_rate_override)
    run_timestamp = resume_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    results = []
    # 训练种子基值: 第 i 个模型用 base + i, 保证同一 base 下各模型独立且整体可复现;
    # 可用环境变量 LU_TRAIN_SEED 覆盖, 便于「同一配置换种子重跑」做稳定性验收.
    train_seed_base = int(os.environ.get("LU_TRAIN_SEED", int(getattr(config.hybrid, 'train_seed', 0))))
    for idx, model_type in enumerate(model_list):
        model_seed = train_seed_base + idx
        checkpoint_path = None
        if resume_timestamp:
            # 断点续训: 有 latest_checkpoint.pth 则从断点续; 已有 val_hybrid.json 说明该模型
            # 已训完, 直接复用其结果并跳过; 两者皆无说明尚未开始, 从 epoch 0 全新训练.
            ckpt_dir = config.training.checkpoint_dir.format(model_name=model_type, run_timestamp=resume_timestamp)
            val_json_path = os.path.join(ckpt_dir, "val_hybrid.json")
            if os.path.exists(val_json_path):
                print(f"模型【{model_type}】已完成训练 (存在 val_hybrid.json), 跳过并复用其结果.")
                with open(val_json_path, 'r', encoding='utf-8') as f:
                    results.append(json.load(f))
                continue
            candidate = os.path.join(ckpt_dir, "latest_checkpoint.pth")
            if os.path.exists(candidate):
                checkpoint_path = candidate
                print(f"模型【{model_type}】检测到断点 {candidate}, 将从断点续训.")
        result = _train_single_model(config, model_type, device, run_timestamp,
                                     checkpoint_path=checkpoint_path, train_seed=model_seed)
        results.append(result)

    print(f"\n================= 全部模型训练完成! =================")

    # 自动串联阈值扫描: 用 4 个 best_model.pth 在不同阈值下跑验证
    model_checkpoints = {r['model']: r['best_model_path'] for r in results}
    doc_info = {
        'training_config': {
            'loss_type': getattr(config.training, 'loss_type', 'focal_dice'),
            'train_size': config.hybrid.train_size,
            'ratio_1015': config.hybrid.ratio_1015,
            'ratio_1520': config.hybrid.ratio_1520,
            'changed_unchanged_ratio': config.hybrid.changed_unchanged_ratio,
            'seed': config.hybrid.seed,
            'train_seed_base': train_seed_base,
            'select_metric': str(getattr(config.training, 'select_metric', 'val_loss')),
            'num_epochs': config.training.num_epochs,
            'factor_list': config.data.factor_list,
            'batch_size': config.data.batch_size,
        },
        'data_composition': results[0].get('stats', {}) if results else {},
    }
    try:
        from pipelines.threshold_sweep import threshold_sweep_pipeline
        threshold_sweep_pipeline(
            config_path=config_path,
            thresholds=None,
            models=model_checkpoints,
            run_timestamp=run_timestamp,
            doc_info=doc_info,
            model_config_overrides=model_config_overrides,
        )
    except Exception as e:
        print(f"\n[error] 阈值扫描失败 (不影响已完成的模型训练): {e}")


# 可插拔模块扫描 (HYBRID_TRANSUNET_LIST) 允许透传的 transunet 配置键
_TRANSUNET_OVERRIDE_KEYS = {
    "attn_type", "up_type", "conv_type", "fusion_type", "resblock_type",
    "parallel_local", "high_res_skip", "n_skip", "use_m1",
    "m1_src", "m1_pos",
}


def _parse_transunet_spec(spec: str) -> Dict[str, Any]:
    """解析一档 transunet 覆写, 如 "up_type=dysample+parallel_local=1" -> {...}。

    档内多个键用 "+" 连接; "baseline" 或空串表示不加覆写 (用配置文件默认值)。
    值自动转换为 bool / int / str, 便于在 shell 中书写。
    """
    spec = spec.strip()
    if not spec or spec.lower() == "baseline":
        return {}
    overrides: Dict[str, Any] = {}
    for item in spec.split("+"):
        if not item.strip():
            continue
        key, _, value = item.partition("=")
        key, value = key.strip(), value.strip()
        if key not in _TRANSUNET_OVERRIDE_KEYS:
            raise ValueError(
                f"HYBRID_TRANSUNET_LIST 不支持的键: {key!r}, 可选键为 {sorted(_TRANSUNET_OVERRIDE_KEYS)}"
            )
        low = value.lower()
        if low in ("1", "true", "yes", "on"):
            overrides[key] = True
        elif low in ("0", "false", "no", "off"):
            overrides[key] = False
        elif value.lstrip("-").isdigit():
            overrides[key] = int(value)
        else:
            overrides[key] = value
    return overrides


def _load_attn_kwargs_by_type(config) -> Dict[str, Dict[str, Any]]:
    """读取 config.hybrid.attn_kwargs_by_type, 统一转成 {attn_type: kwargs(dict)}。"""
    raw = getattr(config.hybrid, "attn_kwargs_by_type", None)
    if raw is None:
        return {}
    items = vars(raw).items() if hasattr(raw, "__dict__") else dict(raw).items()
    return {k: (dict(vars(v)) if hasattr(v, "__dict__") else dict(v or {})) for k, v in items}


def _load_module_overrides(config) -> Dict[str, Dict[str, Any]]:
    """读取 config.hybrid.module_overrides, 统一转成 {spec 原文: 超参 dict}。

    键为档位 spec 的精确原文 (如 "up_type=carafe"); 仅当档位 spec 与键完全一致时才命中。
    """
    hybrid_cfg = getattr(config, "hybrid", None)
    raw = getattr(hybrid_cfg, "module_overrides", None) if hybrid_cfg is not None else None
    if not raw:
        return {}
    items = vars(raw).items() if hasattr(raw, "__dict__") else dict(raw).items()
    return {k: (dict(vars(v)) if hasattr(v, "__dict__") else dict(v or {})) for k, v in items}


def _resolve_module_override(module_overrides: Dict[str, Dict[str, Any]], spec: str):
    """按档位 spec 精确取 (batch_size, learning_rate) 覆写, 缺省返回 (None, None)。"""
    entry = module_overrides.get(spec) or {}
    bs = entry.get("batch_size")
    lr = entry.get("learning_rate")
    return (int(bs) if bs is not None else None,
            float(lr) if lr is not None else None)


def _fmt_hparam_override(batch_size_override, learning_rate_override) -> str:
    """格式化档位超参覆写供日志打印; 无覆写时返回空串。"""
    parts = []
    if batch_size_override is not None:
        parts.append(f"batch_size={batch_size_override}")
    if learning_rate_override is not None:
        parts.append(f"learning_rate={learning_rate_override}")
    return (" | " + ", ".join(parts)) if parts else ""



# ===== Manifest-based resume (HYBRID_RESUME=1) =====
_MANIFEST_ROOT = os.path.join("results", "_manifests")


def _manifest_path(branch_id: str, seed: int) -> str:
    """按 (branch_id, seed) 哈希命名 manifest 文件, 隔离不同分支/seed 的状态."""
    h = hashlib.md5(f"{branch_id}|{seed}".encode("utf-8")).hexdigest()[:8]
    return os.path.join(_MANIFEST_ROOT, f"hybrid_{h}.json")


def _load_manifest(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            print(f"[warn] manifest 损坏, 视为首跑: {path}")
    return {}


def _save_manifest_atomic(path: str, manifest: dict) -> None:
    """原子写: 写临时文件后 os.replace 替换, 防 kill -9 中途截断."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _current_seed(config_path: str) -> int:
    """统一取 train_seed: LU_TRAIN_SEED 环境变量优先, 否则取 config.hybrid.train_seed."""
    cfg = get_yaml(config_path)
    return int(os.environ.get("LU_TRAIN_SEED", getattr(cfg.hybrid, "train_seed", 0)))


class _ResumeRunner:
    """统一管理一个 (branch_id, seed) 下各 spec 的 manifest 续训.

    所有 LIST 分支 + 默认分支共用; 新增 LIST 分支只需实例化本类并遍历.
    用法:
        runner = _ResumeRunner("transunet_list:...", seed)
        for spec in specs:
            resume_ts = runner.get(spec)
            hybrid_train_pipeline(..., resume_timestamp=resume_ts)
            runner.mark_done(spec)

    开关:
    - HYBRID_RESUME=1 启用 manifest 读写 (首跑也建 manifest, 重跑自动续)
    - HYBRID_FORCE_RETRAIN=1 即使有条目也忽略, 生成新时间戳全新训练
    不设 HYBRID_RESUME 时 get() 返回 None, 行为与改动前完全一致.
    """

    def __init__(self, branch_id: str, seed: int):
        self.branch_id = branch_id
        self.seed = seed
        self.resume_on = os.environ.get("HYBRID_RESUME", "0") == "1"
        self.force = os.environ.get("HYBRID_FORCE_RETRAIN", "0") == "1"
        self.mpath = _manifest_path(branch_id, seed) if self.resume_on else None
        self.manifest = _load_manifest(self.mpath) if self.resume_on else {}
        if self.resume_on and not self.manifest:
            self.manifest = {
                "branch": branch_id,
                "seed": seed,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "entries": {},
            }

    def get(self, spec_sig: str):
        """返回 resume_ts (str|None). resume 模式下解析/记录时间戳, 否则 None."""
        if not self.resume_on:
            return None
        entry_key = f"seed_{self.seed}|{spec_sig}"
        entry = self.manifest.get("entries", {}).get(entry_key, {})
        if not self.force and entry.get("run_timestamp"):
            run_ts = entry["run_timestamp"]   # 重跑: 复用已记录时间戳
            mode = "skip/resume"
        else:
            run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")  # 首跑/强制重训
            entry = {
                "run_timestamp": run_ts,
                "status": "in_progress",
                "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "finished_at": None,
            }
            mode = "retrain" if self.force else "first-run"
        self.manifest.setdefault("entries", {})[entry_key] = entry
        _save_manifest_atomic(self.mpath, self.manifest)   # 立即落盘, 防 crash 丢失
        print(f"[manifest] branch={self.branch_id} spec={spec_sig} "
              f"-> run_timestamp={run_ts} ({mode})")
        return run_ts

    def mark_done(self, spec_sig: str):
        if not self.resume_on:
            return
        entry_key = f"seed_{self.seed}|{spec_sig}"
        entry = self.manifest.setdefault("entries", {}).setdefault(entry_key, {})
        entry["status"] = "done"
        entry["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _save_manifest_atomic(self.mpath, self.manifest)


def run_hybrid_train_pipeline(config_path: Optional[str] = None) -> None:
    """混合训练最外层入口: 按环境变量分派扫描/对照组任务.

    所有分支统一支持断点续训, 由 HYBRID_RESUME=1 启用 manifest 机制:
    - 首跑: 预生成 run_timestamp 写入 manifest, 作为目录名固定下来
    - 重跑: 从 manifest 读回 run_timestamp, per-model 逻辑自动跳过/续训
    - 强制重训: HYBRID_FORCE_RETRAIN=1 忽略已有条目, 生成新时间戳全新训练

    环境变量按优先级从高到低:
    - HYBRID_TRANSUNET_LIST="baseline,up_type=dysample,resblock_type=convnext,use_m1=1"
        按档依次覆写 transunet 可插拔模块开关 (档内多键用 "+"), 用于模块扫描/消融;
        第一档不加覆写时训练全部 basemodel 作基线, 后续档只训 transunet 变体。
    - HYBRID_ATTN_LIST="vanilla,agent,swinv2"
        固定数据比例, 依次替换 TransUNet 编码器注意力模块, 每档自动生成独立 run_timestamp;
        各 attn_type 的专属超参取 config.hybrid.attn_kwargs_by_type。
    - HYBRID_RATIO_LIST="0.6,0.7,...,1.0"
        依次扫描 r1 (manifest 模式下所有档均支持断点续训, 不再需要 HYBRID_RATIO_RESUME)。
    - HYBRID_RATIO1520_LIST="0.1,...,0.9"
        固定 r1(取配置文件), 依次扫描 r2。
    - HYBRID_M1=1
        M1 对照轮: 拆 3 档 (base models / transunet / transunet+M1), 共用 train_seed 便于配对对比。
    未设置任何上述变量时: 单次训练 (HYBRID_RESUME_TS 仍可作 legacy 显式指定时间戳)。

    新增 LIST 分支接入续训只需 3 行 (实例化 _ResumeRunner + get + mark_done)。
    """
    config_path = config_path or os.path.join("configs", "hybridTrainConfig.yaml")
    transunet_list_str = os.environ.get("HYBRID_TRANSUNET_LIST") or None
    attn_list_str = os.environ.get("HYBRID_ATTN_LIST") or None
    r1_list_str = os.environ.get("HYBRID_RATIO_LIST") or None
    r2_list_str = os.environ.get("HYBRID_RATIO1520_LIST") or None

    if transunet_list_str:
        # 每档只训 transunet (idx=0 => train_seed=base+0), 使 baseline 与各变体共用同一
        # train_seed, 避免把 seed 差异混进模块效应; 5-base 横向对比请另起一次(不带本变量)。
        specs = [s.strip() for s in transunet_list_str.split(",") if s.strip()]
        module_overrides = _load_module_overrides(get_yaml(config_path))
        seed = _current_seed(config_path)
        runner = _ResumeRunner(f"transunet_list:{transunet_list_str}", seed)
        for idx, spec in enumerate(specs):
            overrides = _parse_transunet_spec(spec)
            bs_ov, lr_ov = _resolve_module_override(module_overrides, spec)
            resume_ts = runner.get(spec)   # spec 原文即 spec_signature
            print(f"\n########## [transunet 模块扫描 {idx + 1}/{len(specs)}] "
                  f"{spec} -> {overrides or '默认(无覆写)'}"
                  f"{_fmt_hparam_override(bs_ov, lr_ov)} ##########")
            hybrid_train_pipeline(
                config_path,
                model_list=["transunet"],
                model_config_overrides={"transunet": overrides},
                batch_size_override=bs_ov,
                learning_rate_override=lr_ov,
                resume_timestamp=resume_ts,
            )
            runner.mark_done(spec)
    elif attn_list_str:
        # 第一档训练全部 basemodel (model_list 取配置文件), 后续档只训 transunet 变体,
        # 避免无注意力开关的模型重复训练.
        attn_kwargs_by_type = _load_attn_kwargs_by_type(get_yaml(config_path))
        attn_types = [x.strip() for x in attn_list_str.split(",") if x.strip()]
        seed = _current_seed(config_path)
        runner = _ResumeRunner(f"attn_list:{attn_list_str}", seed)
        for idx, attn_type in enumerate(attn_types):
            spec_sig = f"attn={attn_type}"
            resume_ts = runner.get(spec_sig)
            print(f"\n########## 开始 attn_type = {attn_type} 的完整训练 ##########")
            hybrid_train_pipeline(
                config_path,
                model_list=None if idx == 0 else ["transunet"],
                model_config_overrides={"transunet": {
                    "attn_type": attn_type,
                    "attn_kwargs": attn_kwargs_by_type.get(attn_type, {}),
                }},
                resume_timestamp=resume_ts,
            )
            runner.mark_done(spec_sig)
    elif r1_list_str:
        # 依次扫描 r1; manifest 模式下所有档均支持断点续训, 旧 HYBRID_RATIO_RESUME 不再需要。
        r1s = [float(x) for x in r1_list_str.split(",")]
        seed = _current_seed(config_path)
        runner = _ResumeRunner(f"r1_list:{r1_list_str}", seed)
        for idx, r1 in enumerate(r1s):
            spec_sig = f"r1={r1}"
            resume_ts = runner.get(spec_sig)
            print(f"\n########## 开始 r1 = {r1} 的完整训练 ##########")
            hybrid_train_pipeline(config_path, resume_timestamp=resume_ts,
                                  ratio_1015_override=r1)
            runner.mark_done(spec_sig)
    elif r2_list_str:
        # r1 固定取配置文件值, 仅扫描 r2; manifest 模式下所有档均支持断点续训。
        r2s = [float(x) for x in r2_list_str.split(",")]
        seed = _current_seed(config_path)
        runner = _ResumeRunner(f"r2_list:{r2_list_str}", seed)
        for r2 in r2s:
            spec_sig = f"r2={r2}"
            resume_ts = runner.get(spec_sig)
            print(f"\n########## 开始 r2 = {r2} 的完整训练 (r1 取配置文件) ##########")
            hybrid_train_pipeline(config_path, resume_timestamp=resume_ts,
                                  ratio_1520_override=r2)
            runner.mark_done(spec_sig)
    elif os.environ.get("HYBRID_M1", "0") == "1":
        # M1 对照轮: 拆 3 档, 让 vanilla-transunet 与 m1-transunet 共用 train_seed(=base+0),
        # 二者唯一差异即 M1 模块, 便于配对对比; 其余 base 模型依次 +0..+n.
        base_models = [m for m in get_yaml(config_path).hybrid.model_list if m != "transunet"]
        m1_steps = [
            ("m1_step_0", {"model_list": base_models, "overrides": None}),
            ("m1_step_1", {"model_list": ["transunet"], "overrides": None}),
            ("m1_step_2", {"model_list": ["transunet"], "overrides": {"transunet": {"use_m1": True}}}),
        ]
        seed = _current_seed(config_path)
        runner = _ResumeRunner("m1_round", seed)
        for spec_sig, cfg in m1_steps:
            resume_ts = runner.get(spec_sig)
            print(f"\n########## [M1轮 {spec_sig}] ##########")
            hybrid_train_pipeline(
                config_path,
                model_list=cfg["model_list"],
                model_config_overrides=cfg["overrides"],
                resume_timestamp=resume_ts,
            )
            runner.mark_done(spec_sig)
    else:
        # 默认单次训练: HYBRID_RESUME_TS 作 legacy 显式指定时间戳; 否则走 manifest。
        legacy_ts = os.environ.get("HYBRID_RESUME_TS") or None
        if legacy_ts:
            hybrid_train_pipeline(config_path, resume_timestamp=legacy_ts)
        else:
            seed = _current_seed(config_path)
            runner = _ResumeRunner("single_train", seed)
            resume_ts = runner.get("default")
            hybrid_train_pipeline(config_path, resume_timestamp=resume_ts)
            runner.mark_done("default")
