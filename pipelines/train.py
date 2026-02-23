# step1: 训练变化检测模型预测LU change(Att U-Net|U-Net|FCN|...)
import os
import json
import time
import argparse
from datetime import datetime
from typing import List

import torch
import torch.nn as nn
from torch.utils import tensorboard
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models import create_model
from datasets.base_dataloader import get_dataset
from utils.tools import ensure_folder, get_yaml, visualize_tensor_list
from utils.dataset_utils import remove_lines_forbatch
import utils.training_utils as tUtils


def training_pipeline(config_path: str, checkpoint_path: str = None) -> None:
    """
    训练变化检测模型的主函数。

    Parameters:
    ----------
        config_path: str 配置文件路径
        checkpoint_path: str 断点续训的checkpoint文件路径，可选
    """
    print("\n================= 开始训练模型... =================")
    
    # 1. 加载配置
    config = get_yaml(config_path)
    print(f"配置文件 {config_path} 加载完成")
    
    # 2. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and config.device.use_gpu else "cpu")
    print(f"使用设备: {device}")

    # 3. 加载数据集
    train_dataset = get_dataset(type="training", data_config=config.data)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=config.data.shuffle,
    )
    
    # 加载验证集
    val_config = config.validation
    val_data_config = argparse.Namespace(**vars(config.data))
    val_data_config.split = val_config.split
    val_dataset = get_dataset(type="training", data_config=val_data_config)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=val_config.batch_size,
        shuffle=val_config.shuffle,
    )
    
    # 4. 初始化模型
    train_config = config.training
    model_type = train_config.model_name
    model = create_model(
        model_type=model_type,
        in_ch=len(config.data.factor_list)+6,  # 增加LU通道
        out_ch=1,
        model_config=getattr(config.model, model_type, {}),
    ).to(device)
    print(f"模型 {model_type} 初始化完成")
    
    # 5. 初始化优化器
    optimizer_config = getattr(config.optimizer, model_type, config.optimizer.default)
    optimizer = tUtils.get_optimizer(
        name=optimizer_config.name,
        parameters=model.parameters(),
        learning_rate=optimizer_config.learning_rate,
        weight_decay=optimizer_config.weight_decay,
        beta_1=optimizer_config.beta_1,
        amsgrad=optimizer_config.amsgrad,
        epsilon=optimizer_config.epsilon,
    )
    print(f"优化器 {optimizer_config.name} 初始化完成")

    # 6. 训练进度记录 初始化
    start_epoch = 0
    epochs_since_improvement = 0  # 记录自上次验证集指标提升以来的epoch数
    tensorboard_step = 0  # 全局步数，用于TensorBoard日志记录
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = train_config.checkpoint_dir.format(
        model_name=train_config.model_name,
        run_timestamp=run_timestamp)
    log_dir = train_config.log_dir.format(
        model_name=train_config.model_name,
        run_timestamp=run_timestamp)
    ensure_folder(checkpoint_dir)
    ensure_folder(log_dir)
    print(f"日志目录: {log_dir}")
    print(f"断点续训目录: {checkpoint_dir}")
    
    # 7. 设置TensorBoard日志
    if train_config.is_tensorboard:
        writer = tensorboard.SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard日志目录: {log_dir}")
    else:
        writer = None
    
    # 8. 断点续训设置
    if checkpoint_path:
        print(f"================= 开始加载checkpoint: {checkpoint_path}... =================")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint.get('epochs_since_improvement', 0)
        tensorboard_step = start_epoch * len(train_dataloader)
        run_timestamp = checkpoint.get('timestamp', run_timestamp)
        checkpoint_dir = os.path.dirname(checkpoint_path)
        print(f"从epoch {start_epoch}继续训练，全局步数: {tensorboard_step}")

    epochs = train_config.num_epochs
    best_val_loss = float('inf')  # 最佳验证损失

    # 9. 训练循环, 每个epoch训练后用验证集评估模型
    print(f"\n================= 开始训练 {epochs} 个epoch... =================")
    for epoch in range(start_epoch, epochs):
        # 检查是否需要调整学习率
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            tUtils.adjust_learning_rate(optimizer, 0.8)
        
        model.train()

        batch_time = tUtils.ExpoAverageMeter()
        train_loss = tUtils.ExpoAverageMeter()
        start = time.time()
        
        for i_batch, data_pack in enumerate(train_dataloader):            
            # 处理一个批次的数据，计算损失
            loss, outputs, diff = process_batch(data_pack, model, device)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪, 防止梯度爆炸
            try:
                nn.utils.clip_grad_norm_(model.parameters(), optimizer_config.gradient_clip)
            except AttributeError:
                pass
            optimizer.step()
            
            train_loss.update(loss.item())
            batch_time.update(time.time() - start)
            start = time.time()
            
            # 更新全局步数 + 记录tensorboard日志
            if writer:
                writer.add_scalar('Loss/train_batch', loss.item(), tensorboard_step)
            # 打印训练信息
            if tensorboard_step % train_config.print_frequency == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{i_batch}/{len(train_dataloader)}], Loss: {train_loss.val:.4f}({train_loss.avg:.4f}), Batch Time: {batch_time.val:.3f}({batch_time.avg:.3f})')
            if tensorboard_step % train_config.visualize_frequency == 0:
                visualize_outputs(outputs, diff, 
                                epoch, tensorboard_step, 
                                save_path=f"{log_dir}\\epoch_{epoch+1}_step_{tensorboard_step}.png")
            tensorboard_step += 1
        
        # 计算epoch平均损失（使用ExpoAverageMeter的指数加权平均）
        epoch_loss = train_loss.avg
        print(f'\nEpoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss:.4f}')
        if writer:
            epoch_step = (epoch + 1) * len(train_dataloader)
            writer.add_scalar('Loss/train_epoch', epoch_loss, epoch_step)
        
        # 在每个epoch后进行验证
        model.eval()
        val_loss = tUtils.ExpoAverageMeter()
        with torch.no_grad():
            for data_pack in val_dataloader:
                # 处理一个批次的数据，计算损失
                loss, _, _ = process_batch(data_pack, model, device)
                val_loss.update(loss.item())
        
        val_epoch_loss = val_loss.avg
        print(f'Epoch [{epoch+1}/{epochs}], Val Loss: {val_epoch_loss:.4f}\n')
        if writer:
            writer.add_scalar('Loss/val_epoch', val_epoch_loss, epoch_step)
        
        # 更新最佳模型
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            epochs_since_improvement = 0
            best_model_filename = f"{checkpoint_dir}\\best_model.pth"
            torch.save(model.state_dict(), best_model_filename)
            print(f"Best model saved: {best_model_filename} (Val Loss: {best_val_loss:.4f})")
        else:
            epochs_since_improvement += 1
        print(f"Epochs since improvement: {epochs_since_improvement}")
        
        # 保存checkpoint（每个epoch都保存）
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            'config': config,
            'timestamp': run_timestamp
        }
        checkpoint_filename = f"{checkpoint_dir}\\latest_checkpoint.pth"
        torch.save(checkpoint, checkpoint_filename)
        print(f"Checkpoint saved: {checkpoint_filename}")
        
        # 保存模型(仅作备份)
        if (epoch + 1) % train_config.save_frequency == 0:
            best_model_filename = f"{checkpoint_dir}\\epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), best_model_filename)
            print(f"Backup model saved: {best_model_filename}")
        print("\n")
    # 关闭TensorBoard writer
    if writer:
        writer.close()
    
    print("\n================= 训练完成! =================")




def valid_pipeline(config_path: str, model_checkpoint_path: str) -> None:
    """
    验证变化检测模型的主函数。

    Parameters:
    ----------
        config_path: str 配置文件路径
        model_checkpoint_path: str 模型checkpoint文件路径
    """
    print("\n================= 开始验证模型... =================")
    # 1. 加载配置
    config = get_yaml(config_path)
    print(f"配置文件 {config_path} 加载完成")
    
    # 2. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and config.device.use_gpu else "cpu")
    print(f"使用设备: {device}")
    
    # 3. 加载验证集
    val_config = config.validation
    val_data_config = argparse.Namespace(**vars(config.data))
    val_data_config.split = getattr(val_config, "split", "val")
    val_data_config.pack = getattr(val_config, "pack", config.data.pack)
    val_dataset = get_dataset(type="training", data_config=val_data_config)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=getattr(val_config, "batch_size", config.data.batch_size),
        shuffle=getattr(val_config, "shuffle", False),
    )

    # 4. 加载模型
    train_config = config.training
    model_type = train_config.model_name
    model = create_model(
        model_type=model_type,
        in_ch=len(config.data.factor_list)+6,  # 增加LU通道
        out_ch=1,
        model_config=getattr(config.model, model_type, {}),
    ).to(device)
    print(f"模型 {model_type} 初始化完成")

    # 5. 加载checkpoint
    checkpoint_path = model_checkpoint_path
    print(f"\n================= 开始加载checkpoint: {checkpoint_path}... =================")
    model_state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(model_state_dict)
    print(f"Checkpoint {checkpoint_path} 加载完成")

    # 6. 验证模型
    model.eval()
    val_loss = tUtils.ExpoAverageMeter()
    val_metric = tUtils.MetricsMeter()
    print(f"\n================= 开始验证 {len(val_dataloader)} 个batch... =================")
    
    with torch.no_grad():
        for i_batch, data_pack in enumerate(val_dataloader):
            # 处理一个批次的数据，计算损失
            loss, outputs, diff = process_batch(data_pack, model, device)
            # 计算指标
            outputs = F.sigmoid(outputs)
            val_metric.update(outputs, diff)

            val_loss.update(loss.item())
    
    val_epoch_loss = val_loss.avg
    print(f'\nVal Loss: {val_epoch_loss:.4f}')
    # 打印指标
    print(val_metric)
    # 保存json
    metrics = val_metric.get_metrics()
    metrics['model_name'] = train_config.model_name
    metrics['data'] = val_config.pack
    metrics['path'] = checkpoint_path
    json_path = os.path.join(os.path.dirname(checkpoint_path), f"val_metrics_{val_config.pack}.json")
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"指标已保存: {json_path}")

    print("\n================= 验证完成! =================")
    


def process_batch(data_pack: List[torch.Tensor], model: nn.Module, device: torch.device):
    """
    处理一个批次的数据，计算损失。
    PS: 这里的损失函数是 BCEWithLogitsLoss，不需要手动激活输出，outputs是模型的原始输出。
    若需使用返回的outputs，则需要手动激活outputs张量。

    Parameters:
    ----------
        data_pack: list[torch.Tensor] 一个批次的数据，
            包含驱动因子、srcLU、tgtLU、idx_label、changed_flag
        model: nn.Module 训练的模型
        device: torch.device 设备（CPU或GPU）

    Returns:
    -------
        loss: torch.Tensor 该批次的损失值
        outputs: torch.Tensor 模型输出, 形状为 (batch_size, 1, 128, 128)
        diff: torch.Tensor 目标ground truth, 形状为 (batch_size, 1, 128, 128)
    """
    # data_pack格式: [srcFactors, srcLU, diff, idx_label, changed_flag]
    # 1. src Factors (batch_size, factor_num, 128, 128)
    factors_src = torch.tensor(data_pack[0], dtype=torch.float32).to(device)
    # 2. src LU (batch_size, 6, 128, 128)
    lu_src = torch.tensor(data_pack[1], dtype=torch.float32).to(device)
    # 3. diff (batch_size, 1, 128, 128)
    diff = torch.tensor(data_pack[2], dtype=torch.float32).to(device)
    
    inputs = torch.cat([factors_src, lu_src], dim=1)
    diff = remove_lines_forbatch(diff.cpu()).to(device)

    outputs = model(inputs)
    loss = tUtils.cal_loss(outputs, diff)
    return loss, outputs, diff

def visualize_outputs(outputs: torch.Tensor, diff: torch.Tensor, epoch: int, step: int, save_path: str = None):
    """
    可视化模型输出和目标差异。
    PS: outputs 是模型的原始输出，这里需要sigmoid激活。(计算loss时自动经过sigmoid，这里需要手动激活)

    Parameters:
    ----------
        outputs: torch.Tensor 模型输出，形状为 (batch_size, 1, 128, 128)
        diff: torch.Tensor 目标差异，形状为 (batch_size, 1, 128, 128)
        epoch: int 当前训练epoch
        step: int 当前训练步数 tensorboard_step
        save_path: str 保存可视化图片的路径
    """
    # 转换为CPU上的numpy数组
    outputs = F.sigmoid(outputs)
    outputs = outputs.detach().cpu()
    diff = diff.detach().cpu()
    
    # 构建tensor dict
    tensor_dict = {}
    batch_size = min(4, outputs.shape[0])  # 确保不超过实际batch size
    
    for i in range(batch_size):
        tensor_dict[f'output{i+1}'] = outputs[i]
        tensor_dict[f'gt{i+1}'] = diff[i]
    
    # 可视化前4个样本
    return visualize_tensor_list(
        tensor_dict=tensor_dict,
        main_title=f'Epoch {epoch+1}, Step {step}',
        save_path = save_path,
        color_map='gray',
        figsize = (10, batch_size*6),
        layout=(batch_size, 2),
    )