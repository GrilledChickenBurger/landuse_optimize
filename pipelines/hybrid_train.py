# 混合训练集训练 pipeline
# 遍历 attunet/unet/segformer/transunet 四个模型, 依次使用 HybridDataset 训练。
# 验证集使用 val split 的 15-20 时间对; 复用 train.py 的 process_batch / visualize_outputs
# 与 training_utils 的优化器/损失工具, 不修改 train.py.
import os
import math
import time
import json
import shutil
import argparse
from datetime import datetime
from typing import List, Optional

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
                        run_timestamp: str, train_stats: dict, config) -> str:
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
        'seed': config.hybrid.seed,
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
        f"- 随机种子: `{training_config['seed']}`\n\n"
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
                        checkpoint_path: Optional[str] = None) -> dict:
    """使用混合训练集训练单个模型."""
    data_config = config.data
    train_config = config.training
    hybrid_config = config.hybrid
    optimizer_config = _resolve_optimizer_config(config, model_type)

    print(f"\n================= 开始训练模型【{model_type}】... =================")

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
    best_val_loss = float('inf')
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
        # 旧版 checkpoint 未存 best_val_loss: 先备份已有 best_model.pth, 避免 resume 后首个
        # val_loss (< inf) 立刻用当前(可能更差)权重覆盖历史最优.
        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']
        else:
            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
            backup_path = os.path.join(checkpoint_dir, "best_model_pre_resume.pth")
            if os.path.exists(best_model_path) and not os.path.exists(backup_path):
                shutil.copyfile(best_model_path, backup_path)
                print(f"[warn] checkpoint 无 best_val_loss 字段, 历史最优已备份为 {backup_path}")
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

        # 验证
        model.eval()
        val_loss = ExpoAverageMeter()
        with torch.no_grad():
            for data_pack in val_dataloader:
                loss, _, _ = process_batch(data_pack, model, device, loss_fn=loss_fn)
                val_loss.update(loss.item())
        val_epoch_loss = val_loss.avg
        print(f'Epoch [{epoch+1}/{epochs}], Val Loss: {val_epoch_loss:.4f}\n')
        if writer:
            writer.add_scalar('Loss/val_epoch', val_epoch_loss, (epoch + 1) * len(train_dataloader))
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], (epoch + 1) * len(train_dataloader))

        # 保存 best / checkpoint
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            epochs_since_improvement = 0
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
            print(f"Best model saved (Val Loss: {best_val_loss:.4f})")
        else:
            epochs_since_improvement += 1
        print(f"Epochs since improvement: {epochs_since_improvement}")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            'epochs_since_improvement': epochs_since_improvement,
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
    )
    print(f"验证报告已归档: {report_dir}")

    print(f"模型【{model_type}】验证完成: Val Loss={final_val_loss:.4f}, "
          f"F1={result.get('f1_score', 0):.4f}, IoU={result.get('iou', 0):.4f}")

    print(f"\n================= 模型【{model_type}】训练完成! =================")
    return result


def hybrid_train_pipeline(config_path: str, model_list: Optional[List[str]] = None,
                          resume_timestamp: Optional[str] = None) -> None:
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
    """
    config = get_yaml(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() and config.device.use_gpu else "cpu")
    print(f"使用设备: {device}")

    model_list = model_list or config.hybrid.model_list
    run_timestamp = resume_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    results = []
    for model_type in model_list:
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
        result = _train_single_model(config, model_type, device, run_timestamp, checkpoint_path=checkpoint_path)
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
        )
    except Exception as e:
        print(f"\n[error] 阈值扫描失败 (不影响已完成的模型训练): {e}")
