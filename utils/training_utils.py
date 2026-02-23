# 训练模型相关工具
import os
import json
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix
from utils.performance import calculate_OA, calculate_kappa

def get_optimizer(name, parameters, learning_rate, weight_decay,
                  beta_1, amsgrad=False, epsilon=1e-7, maximize=False):
    print(f"\n================= 开始初始化优化器【{name}】... =================")
    if name == "adam":
        optimizer = optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay,
                               betas=(beta_1, 0.999), amsgrad=amsgrad, eps=epsilon,
                               maximize=maximize)
    elif name == "adamw":
        optimizer = optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay,
                                betas=(beta_1, 0.999), amsgrad=amsgrad, eps=epsilon,
                                maximize=maximize)
    elif name == "rmsprop":
        optimizer = optim.RMSprop(parameters, lr=learning_rate, weight_decay=weight_decay,
                                  maximize=maximize)
    elif name == "sgd":
        optimizer = optim.SGD(parameters, lr=learning_rate, momentum=0.9, maximize=maximize)
    else:
        raise NotImplementedError(f"Optimizer {name} not supported.")
    return optimizer

def adjust_learning_rate(optimizer, shrink_factor):
    """
    调整学习率，将当前学习率乘以shrink_factor

    Parameters:
    -----------
    - optimizer: 要调整学习率的优化器
    - shrink_factor: 学习率减少的因子（0<shrink_factor<1）
    """
    print(f"\n减少学习率，因子为：{shrink_factor}")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print(f"新学习率为：{optimizer.param_groups[0]['lr']}\n")

class ExpoAverageMeter(object):
    # Exponential Weighted Average Meter
    def __init__(self, beta=0.9):
        self.reset()

    def reset(self):
        self.beta = 0.9
        self.val = 0
        self.avg = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.avg = self.beta * self.avg + (1 - self.beta) * self.val

def cal_loss(output: torch.Tensor, target: torch.Tensor, focal_weight=0.5) -> torch.Tensor:
    """
    计算Focal损失+DICE损失, Focal损失权重为focal_weight
    PS: 输入的output未经过sigmoid激活, 计算Focal损失时使用的binary_cross_entropy_with_logits内部包含sigmoid处理

    Parameters:
    -----------
    - output: 模型输出，形状为 (batch_size, 1, 128, 128)
    - target: 目标值，形状为 (batch_size, 1, 128, 128)
    - focal_weight: Focal损失的权重，默认0.5

    Returns:
    --------
    - loss: 计算得到的Focal损失+DICE损失
    """
    focal = focal_loss_with_logits(output, target)
    prediction = F.sigmoid(output)
    dice = dice_loss(prediction, target)

    loss = focal * focal_weight + dice * (1 - focal_weight)

    return loss

# def cal_loss(output: torch.Tensor, target: torch.Tensor, bce_weight=0.5) -> torch.Tensor:
#     """
#     计算交叉熵损失(BCE)+DICE损失, bce权重为bce_weight
#     PS: 输入的output未经过sigmoid激活, 计算BCE时使用的binary_cross_entropy_with_logits内部包含sigmoid处理

#     Parameters:
#     -----------
#     - output: 模型输出，形状为 (batch_size, 1, 128, 128)
#     - target: 目标值，形状为 (batch_size, 1, 128, 128)
#     - bce_weight: BCE损失的权重，默认0.5

#     Returns:
#     --------
#     - loss: 计算得到的交叉熵损失+DICE损失
#     """
#     bce = F.binary_cross_entropy_with_logits(output, target)
#     prediction = F.sigmoid(output)
#     dice = dice_loss(prediction, target)

#     loss = bce * bce_weight + dice * (1 - bce_weight)

#     return loss


def focal_loss_with_logits(logits: torch.Tensor, target: torch.Tensor,
    alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
    """
    计算Focal Loss

    Parameters:
    -----------
    - logits: 模型输出，形状为 (batch_size, 1, 128, 128)
    - target: 目标值，形状为 (batch_size, 1, 128, 128)
    - alpha: Focal Loss的alpha参数，默认0.75
    - gamma: Focal Loss的gamma参数，默认2.0
    - reduction: 损失 reduction 模式，默认"mean"

    Returns:
    --------
    - loss: 计算得到的Focal Loss, reduction模式为"mean"时返回标量, 否则返回[batch_size,]
    """
    target = target.float()
    # 计算Focal Loss
    bce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = torch.exp(-bce_loss)
    focal_weight =  (1 - p_t) ** gamma 
    
    if alpha is not None:
        alpha_weight = alpha * target + (1 - alpha) * (1 - target)
        focal_loss = alpha_weight * focal_weight * bce_loss
    else:
        focal_loss = focal_weight * bce_loss
    
    # 每个样本单独聚合
    focal_loss = focal_loss.mean(dim=(1, 2, 3))  # → [B]
    if reduction == "mean":
        return focal_loss.mean()  # → 标量
    elif reduction == "sum":
        return focal_loss.sum()  # → 标量
    else:
        return focal_loss  # → [B]

def dice_loss(prediction: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    计算DICE损失

    Parameters:
    -----------
    - prediction: 模型输出，形状为 (batch_size, 1, 128, 128)
    - target: 目标值，形状为 (batch_size, 1, 128, 128)
    - smooth: 平滑项，默认1

    Returns:
    --------
    - loss: 计算得到的DICE损失
    """
    prediction = prediction.float()
    target = target.float()

    intersection = (prediction * target).sum(dim=(2, 3))  # [B,1]
    union = prediction.sum(dim=(2, 3)) + target.sum(dim=(2, 3))  # [B,1]

    dice = (2.0 * intersection + smooth) / (union + smooth)

    dice = dice.squeeze(1)  # → [B]

    return 1 - dice.mean()  # → 标量


# def dice_loss2(prediction: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
#     """
#     计算DICE损失

#     Parameters:
#     -----------
#     - prediction: 模型输出，形状为 (batch_size, 1, 128, 128)
#     - target: 目标值，形状为 (batch_size, 1, 128, 128)
#     - smooth: 平滑项，默认1

#     Returns:
#     --------
#     - loss: 计算得到的DICE损失
#     """
#     i_flat = prediction.view(-1)
#     t_flat = target.view(-1)
#     intersection = (i_flat * t_flat).sum()

#     return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))

class MetricsMeter(object):
    """
    累积预测和真实标签张量，并在每次更新时计算总体指标
    """
    def __init__(self, device='cpu'):
        self.device = device
        self.reset()

    def reset(self):
        """重置所有累积的张量和指标"""
        self.all_pred = None
        self.all_target = None
        self.count = 0
        self.current_metrics = {}

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        更新累积的张量并计算当前总体指标
        
        参数:
            pred: 预测张量，形状为[batch_size, 1, H, W]
            target: 真实标签张量，形状为[batch_size, 1, H, W]
        """
        # 确保张量在正确的设备上
        pred = pred.to(self.device)
        target = target.to(self.device)
        
        # 累积张量
        if self.all_pred is None:
            self.all_pred = pred
            self.all_target = target
        else:
            self.all_pred = torch.cat([self.all_pred, pred], dim=0)
            self.all_target = torch.cat([self.all_target, target], dim=0)
        
        self.count += pred.size(0)
        
        # 计算当前总体指标
        self._calculate_metrics()
        
        return self.current_metrics

    def _calculate_metrics(self):
        """计算所有累积数据的总体指标"""
        if self.all_pred is None:
            return
            
        # 将预测值转换为0/1（假设是二分类）
        pred_binary = (self.all_pred > 0.5).float()
        target_binary = self.all_target
        
        # 计算混淆矩阵分量
        TP = ((pred_binary == 1) & (target_binary == 1)).float().sum().item()
        FP = ((pred_binary == 1) & (target_binary == 0)).float().sum().item()
        TN = ((pred_binary == 0) & (target_binary == 0)).float().sum().item()
        FN = ((pred_binary == 0) & (target_binary == 1)).float().sum().item()
        
        total_pixels = TP + FP + TN + FN
        
        # 计算各项指标
        self.current_metrics = {
            'accuracy': (TP + TN) / (total_pixels + 1e-6),
            'precision': TP / (TP + FP + 1e-6),
            'recall': TP / (TP + FN + 1e-6),
            'f1_score': 2 * TP / (2 * TP + FP + FN + 1e-6),
            'iou': TP / (TP + FP + FN + 1e-6),
            'tp': TP,
            'fp': FP,
            'tn': TN,
            'fn': FN,
            'total_samples': self.count,
            'total_pixels': total_pixels
        }

    def get_metrics(self):
        """获取当前累积的所有指标"""
        return self.current_metrics

    def __str__(self):
        """格式化输出当前指标"""
        if not self.current_metrics:
            return "MetricsMeter: No data accumulated"
        
        metrics = self.current_metrics
        return (
            f"MetricsMeter (samples: {metrics['total_samples']}, pixels: {metrics['total_pixels']:.0f}):\n"
            f"  Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}\n"
            f"  Recall: {metrics['recall']:.4f}, F1-Score: {metrics['f1_score']:.4f}\n"
            f"  IoU: {metrics['iou']:.4f}, TP/FP/TN/FN: {metrics['tp']:.0f}/{metrics['fp']:.0f}/{metrics['tn']:.0f}/{metrics['fn']:.0f}"
        )
