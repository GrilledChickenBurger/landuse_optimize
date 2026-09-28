# step2.5: 多阈值 × 多模型 变化检测指标扫描
# 对每个 sigmoid 二值化阈值, 依次评估各预训练模型在评测集上的表现
# 评测集口径由 config.sweep.eval_split 决定, 默认 "all" = 15-20 的 train+val 全量
# 设计: 数据只加载一次, 每个模型只推理一次; 推理时对每个阈值即时累积混淆矩阵计数,
#       不再缓存概率图, 与「先存概率图再重二值化」数学上逐位等价.
import os
import json
import argparse
from datetime import datetime
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

from models import create_model
from datasets.base_dataloader import get_dataset
from utils.tools import get_yaml, ensure_folder
from utils.dataset_utils import remove_lines_forbatch
from utils.training_utils import MetricsMeter

# 模型名称 -> checkpoint 路径
MODEL_CHECKPOINTS: Dict[str, str] = {
    "attunet":   os.path.join("checkpoints", "attunet",   "20260212_045750", "best_model.pth"),
    "unet":      os.path.join("checkpoints", "unet",      "20260212_155606", "best_model.pth"),
    "segformer": os.path.join("checkpoints", "segformer", "20260213_062110", "best_model.pth"),
    "transunet": os.path.join("checkpoints", "transunet", "20260214_052453", "best_model.pth"),
}

# 扫描阈值列表
THRESHOLDS: List[float] = [0.1, 0.3, 0.5, 0.7]

# 指标表格列(整数计数列与小数指标列分开格式化)
INT_COLS = ["tp", "fp", "tn", "fn", "total_samples", "total_pixels"]
FLOAT_COLS = ["accuracy(OA)", "precision(UA)", "recall(PA)", "f1_score", "iou"]
ALL_COLS = FLOAT_COLS + INT_COLS


def _build_dataloader(config):
    """构建评测集 DataLoader.

    评测口径由 config.sweep.eval_split 决定:
      "all" (默认): 把 train 与 val 两个 split 的 pack 拼接, 用于 15-20 全量口径评测;
      其他取值:     只评测该 split (如 "val").
    评测集不施加数据增强 (transform=False), 否则 train 侧会带上随机翻转.
    """
    data_config = argparse.Namespace(**vars(config.data))
    data_config.only_changed = False
    data_config.transform = False
    data_config.pack = getattr(config.validation, "pack", getattr(config.data, "pack", [[15, 20]]))

    # sweep 推理批大小: 优先 sweep.batch_size, 其次 data.batch_size_overrides[profile] 的 default/最大值,
    # 不再复用 validation.batch_size(=1), 否则 1815 batch/模型 × 5 模型串行单图推理极慢.
    sweep_config = getattr(config, "sweep", None)
    batch_size = getattr(sweep_config, "batch_size", None) if sweep_config is not None else None
    if batch_size is None:
        overrides = getattr(config.data, "batch_size_overrides", None)
        profile = getattr(config, "profile", "local")
        prof = getattr(overrides, profile, None) if overrides is not None else None
        if prof is not None:
            vals = [v for k, v in vars(prof).items() if isinstance(v, int)]
            batch_size = max(vals) if vals else config.data.batch_size
        else:
            batch_size = config.data.batch_size

    eval_split = getattr(sweep_config, "eval_split", "val") if sweep_config is not None else "val"
    data_config.eval_splits = ["train", "val"] if eval_split == "all" else [eval_split]

    # DataLoader 并行: 优先 sweep.num_workers, 否则沿用 data.num_workers_overrides[profile].
    num_workers = getattr(sweep_config, "num_workers", None) if sweep_config is not None else None
    if num_workers is None:
        nw_overrides = getattr(config.data, "num_workers_overrides", None)
        profile = getattr(config, "profile", "local")
        num_workers = getattr(nw_overrides, profile, 0) if nw_overrides is not None else 0

    datasets = []
    for split in data_config.eval_splits:
        data_config.split = split
        datasets.append(get_dataset(type="training", data_config=data_config))
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    return data_config, dataloader


def _infer_and_count(model, dataloader, device, thresholds: List[float]) -> Dict[float, Dict[str, float]]:
    """对评测集逐 batch 推理, 并按每个阈值即时累积混淆矩阵计数.

    不再缓存 sigmoid 概率图: 概率图体积大(1815×1×128×128 float32 ≈ 119MB/模型),
    且阈值段要逐图重读. 这里在 inference_mode 下逐 batch 直接对 4 个阈值各算一次
    混淆矩阵并累加, 与「先存概率图再重二值化」在数学上逐位等价(像素计数为整数).

    Returns
    -------
        counts: {threshold: {'tp','fp','tn','fn','total_samples'}}
    """
    model.eval()
    meters = {thr: MetricsMeter(threshold=thr, accumulate_counts=True, device=device) for thr in thresholds}
    with torch.inference_mode():
        for data_pack in dataloader:
            factors_src = torch.as_tensor(data_pack[0], dtype=torch.float32).to(device, non_blocking=True)
            lu_src = torch.as_tensor(data_pack[1], dtype=torch.float32).to(device, non_blocking=True)
            # GT 清洗与原实现逐位一致(在 CPU 完成后一次性上卡)
            diff = remove_lines_forbatch(
                torch.as_tensor(data_pack[2], dtype=torch.float32)
            ).to(device, non_blocking=True)

            inputs = torch.cat([factors_src, lu_src], dim=1)
            outputs = F.sigmoid(model(inputs))
            for thr in thresholds:
                meters[thr].update(outputs, diff)
    return {thr: m.get_metrics() for thr, m in meters.items()}


def _render_markdown_table(results: Dict[str, Dict], threshold: float) -> str:
    """渲染一张 markdown 表格: 行=模型, 列=指标"""
    models = list(results.keys())
    header = ["模型"] + ALL_COLS
    lines = [f"## 阈值 = {threshold}", ""]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for model in models:
        m = results[model]
        cells = [model]
        for col in ALL_COLS:
            v = m.get(col, 0)
            if col in FLOAT_COLS:
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(int(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _print_markdown_table(results: Dict[str, Dict], threshold: float) -> None:
    """控制台打印一张 markdown 表格 (复用 _render_markdown_table 的输出)."""
    print(f"\n================= 阈值 = {threshold} =================")
    print(_render_markdown_table(results, threshold))


def _write_sweep_report(out_dir: str, run_timestamp: str, models: Dict[str, str],
                        thresholds: List[float], doc_info: Dict, config, data_config,
                        all_results: Dict[float, Dict]) -> None:
    """写阈值扫描的说明文档 (info.json + README.md)."""
    eval_dataset_info = {
        'splits': list(data_config.eval_splits),
        'pack': data_config.pack,
    }
    training_config = (doc_info or {}).get('training_config', {})
    data_composition = (doc_info or {}).get('data_composition', {})

    # info.json
    info = {
        'run_timestamp': run_timestamp,
        'thresholds': thresholds,
        'models': models,
        'eval_dataset': eval_dataset_info,
        'training_config': training_config,
        'data_composition': data_composition,
        'results': {str(k): v for k, v in all_results.items()},
    }
    with open(os.path.join(out_dir, "info.json"), 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=4, ensure_ascii=False)

    # README.md
    model_lines = "\n".join(f"- `{name}`: `{path}`" for name, path in models.items())
    comp_lines = "\n".join(f"- `{k}`: {v}" for k, v in data_composition.items())
    md = (
        f"# 阈值扫描报告\n\n"
        f"- 运行时间戳: `{run_timestamp}`\n"
        f"- 阈值列表: `{thresholds}`\n\n"
        f"## 模型 checkpoint\n\n{model_lines}\n\n"
        f"## 训练配置\n\n"
        f"- 损失函数 (loss_type): `{training_config.get('loss_type')}`\n"
        f"- 目标训练集样本量 N (train_size): `{training_config.get('train_size')}`\n"
        f"- 10-15 变化样本加入比例 r1: `{training_config.get('ratio_1015')}`\n"
        f"- 15-20 变化样本加入比例 r2: `{training_config.get('ratio_1520')}`\n"
        f"- 变化:非变化 目标比例 r_cu: `{training_config.get('changed_unchanged_ratio')}`\n"
        f"- seed: `{training_config.get('seed')}`\n"
        f"- num_epochs: `{training_config.get('num_epochs')}`\n"
        f"- factor_list: `{training_config.get('factor_list')}`\n"
        f"- batch_size: `{training_config.get('batch_size')}`\n\n"
        f"## 训练集数据构成 (来源与比例)\n\n{comp_lines}\n\n"
        f"## 评测集 (口径)\n\n"
        f"- splits: `{eval_dataset_info['splits']}` (train+val 表示 15-20 全量口径)\n"
        f"- pack: `{eval_dataset_info['pack']}`\n\n"
    )
    with open(os.path.join(out_dir, "README.md"), 'w', encoding='utf-8') as f:
        f.write(md)
    print(f"已保存: {os.path.join(out_dir, 'info.json')}")
    print(f"已保存: {os.path.join(out_dir, 'README.md')}")


def threshold_sweep_pipeline(config_path: str,
                             thresholds: List[float] = None,
                             models: Dict[str, str] = None,
                             run_timestamp: str = None,
                             doc_info: Dict = None) -> None:
    """
    多阈值扫描: 对每个阈值, 在验证集上评估所有预训练模型并输出一张指标表格

    Parameters:
    ----------
        config_path: str, 配置文件路径 (hybridTrainConfig.yaml 或 trainConfig.yaml)
        thresholds: List[float], 待扫描的 sigmoid 二值化阈值列表; None 用默认 THRESHOLDS
        models: Dict[str, str], 模型名称 -> checkpoint 路径; None 用默认 MODEL_CHECKPOINTS
        run_timestamp: str, 输出目录时间戳; None 则当前时间生成
        doc_info: Dict, 训练配置与数据构成, 用于写说明文档
    """
    print("\n================= 开始多阈值模型扫描 =================")
    if thresholds is None:
        thresholds = THRESHOLDS
    if models is None:
        models = MODEL_CHECKPOINTS
    if run_timestamp is None:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. 加载配置与设备
    config = get_yaml(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() and config.device.use_gpu else "cpu")
    print(f"使用设备: {device}")

    # 2. 构建评测集(所有模型共用同一份数据, 只加载一次)
    data_config, dataloader = _build_dataloader(config)
    print(f"评测数据集: splits={data_config.eval_splits}, pack={data_config.pack}, "
          f"共 {len(dataloader)} 个batch")

    # 3. 每个模型推理一次, 并按所有阈值即时累积混淆矩阵计数
    #    all_counts[model_name][threshold] = metrics_dict
    all_counts: Dict[str, Dict[float, Dict]] = {}
    in_ch = len(config.data.factor_list) + 6  # 驱动因子 + LU one-hot 通道
    for model_name, ckpt_path in models.items():
        print(f"\n================= 推理模型: {model_name} =================")
        model = create_model(
            model_type=model_name,
            in_ch=in_ch,
            out_ch=1,
            model_config=getattr(config.model, model_name, {}),
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        all_counts[model_name] = _infer_and_count(model, dataloader, device, thresholds)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 4. 输出目录 (优先用 config.result.sweep_dir 模板, 否则回退旧目录)
    result_cfg = getattr(config, 'result', None)
    if result_cfg is not None:
        out_dir = result_cfg.sweep_dir.format(run_timestamp=run_timestamp)
    else:
        out_dir = os.path.join("checkpoints", "threshold_sweep")
    ensure_folder(out_dir)

    # 5. 整理每个阈值的指标表, 输出一张表 + 一个 json
    all_results: Dict[float, Dict] = {}
    for thr in thresholds:
        results: Dict[str, Dict] = {}
        for model_name in models.keys():
            metrics = dict(all_counts[model_name][thr])
            metrics["model_name"] = model_name
            results[model_name] = metrics

        _print_markdown_table(results, thr)
        all_results[thr] = results

        # 保存当前阈值的汇总 json
        save_data = {
            "threshold": thr,
            "data": data_config.pack,
            "models": {name: {k: (float(v) if isinstance(v, (int, float)) else v)
                              for k, v in m.items()}
                       for name, m in results.items()},
        }
        json_path = os.path.join(out_dir, f"metrics_thr{thr}.json")
        with open(json_path, "w") as f:
            json.dump(save_data, f, indent=4)
        print(f"已保存: {json_path}")

        # 保存当前阈值的 markdown 表格
        table_path = os.path.join(out_dir, f"table_thr{thr}.md")
        with open(table_path, "w", encoding='utf-8') as f:
            f.write(_render_markdown_table(results, thr))
        print(f"已保存: {table_path}")

    # 6. 写说明文档 (info.json + README.md)
    _write_sweep_report(out_dir, run_timestamp, models, thresholds, doc_info, config, data_config, all_results)

    print("\n================= 多阈值模型扫描完成! =================")
