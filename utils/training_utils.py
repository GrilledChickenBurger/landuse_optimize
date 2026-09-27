# 训练模型相关工具
import os
import json
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

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


def cal_loss_bce(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    原始 BCE 损失 (SegFormer/FCN/ViT 的多分类 CrossEntropy 在二分类 sigmoid 下的等价形式).

    segformer: 论文 (Xie et al., NeurIPS 2021) 明确"不使用 OHEM、辅助损失或类别平衡损失",
    官方实现为未加权 CrossEntropyLoss, 故此处不传 weight / pos_weight.
    其 ignore_index=255 用于 ADE20K 未标注像素与 padding, 本项目标签全为有效 0/1,
    无对应语义, 故该机制未引入 (等价于不生效).

    vit: 论文 (Dosovitskiy et al., ICLR 2021, arXiv:2010.11929) §4.1 为 softmax 交叉熵;
    官方实现 google-research/vision_transformer 的 vit_jax/train.py (cross_entropy_loss)
    即 -mean(Σ log_softmax(logits) * labels). 已核对 train.py / input_pipeline.py /
    configs 无 label smoothing、mixup 或类别权重, 属未加权的纯 CE.
    原论文该损失作用在 cls token 的分类头上 (一张图一个类别), 本项目将 ViT 改造为
    逐像素分割后, C=2 的 softmax CE 与 BCEWithLogits 数学恒等 (同 segformer 的适配),
    故仍用未加权 BCE.

    swin: 论文 (Liu et al., ICCV 2021, arXiv:2103.14030) 同为分类任务; 官方实现
    microsoft/Swin-Transformer 的 main.py (第 115-121 行) 按配置三选一:
      AUG.MIXUP > 0             → SoftTargetCrossEntropy()            (默认 MIXUP=0.8)
      MODEL.LABEL_SMOOTHING > 0 → LabelSmoothingCrossEntropy(0.1)     (默认 0.1)
      否则                      → nn.CrossEntropyLoss()
    其 config.py 默认 MIXUP=0.8 / CUTMIX=1.0 / LABEL_SMOOTHING=0.1, 即默认走 mixup 软标签 CE.
    本项目不引入 mixup / cutmix: 变化检测标签为稀疏 0/1 (正类约 1.7%), 样本间线性混叠会
    稀释正类信号, 且需改动数据流. 故 Swin 只取官方"关闭 mixup 后的 label smoothing 分支",
    由 cal_loss_bce_ls 承载 (timm LabelSmoothingCrossEntropy 的 C=2 等价形式);
    本函数 (未加权 BCE) 不再用于 swin, 仅供 segformer / fcn / vit 使用.

    Parameters:
    -----------
    - output: 模型输出(logits), 形状 (batch_size, 1, 128, 128)
    - target: 目标值, 形状 (batch_size, 1, 128, 128)
    """
    return F.binary_cross_entropy_with_logits(output, target.float())


def cal_loss_bce_ls(output: torch.Tensor, target: torch.Tensor,
                    smoothing: float = 0.1, num_classes: int = 2) -> torch.Tensor:
    """
    Label Smoothing 版 BCE (Swin 官方默认损失在 C=2 逐像素口径下的等价形式).

    swin: 官方实现 microsoft/Swin-Transformer 的 main.py (第 115-121 行) 在
    AUG.MIXUP <= 0 且 MODEL.LABEL_SMOOTHING > 0 时使用 timm 的
    LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING); 其 config.py
    默认 MODEL.LABEL_SMOOTHING = 0.1 (官方默认 MIXUP=0.8 走 SoftTargetCrossEntropy,
    该分支未引入, 理由见 cal_loss_bce 的 swin 段).

    timm 该实现的定义为硬标签 CE 与"各类别均匀分布 CE"的凸组合:
        loss = (1 - eps) * CE(y_hard) + eps * mean_c(-log p_c)

    C=2 适配: 令 on = 1 - eps + eps/C, off = eps/C, 则上式与
    BCEWithLogits(logits, y_smooth) 数学恒等 (y_smooth = y ? on : off), 因为
        (1-eps)(-log p_y) + eps/C * Σ_c(-log p_c) = -on * log p_y - off * log p_{1-y}
    binary_cross_entropy_with_logits 的公式对 [0,1] 软标签同样成立, 故直接传平滑后的
    目标即可, 无需手写 log_softmax. eps=0.1 时 on=0.95, off=0.05.

    Parameters:
    -----------
    - output: 模型输出(logits), 形状 (batch_size, 1, 128, 128)
    - target: 目标值, 形状 (batch_size, 1, 128, 128), 硬标签 0/1
    - smoothing: 平滑系数 eps, 默认 0.1 (官方 config.py 的 MODEL.LABEL_SMOOTHING 默认值)
    - num_classes: 类别数, 默认 2 (前景/背景), 用于把 eps 折算为 off = eps/C
    """
    y = target.float()
    off_value = smoothing / num_classes
    on_value = 1.0 - smoothing + off_value
    y_smooth = y * on_value + (1.0 - y) * off_value
    return F.binary_cross_entropy_with_logits(output, y_smooth)


def cal_loss_bce_dice(output: torch.Tensor, target: torch.Tensor,
                      bce_weight: float = 0.5, dice_smooth: float = 1e-5) -> torch.Tensor:
    """
    TransUNet 官方实现损失: 0.5 * CE + 0.5 * Dice.

    论文 (Chen et al., 2021, arXiv:2102.04306) 正文未描述训练损失(全文无 cross-entropy
    表述, "dice" 仅作评价指标), 故对齐其官方代码 Beckschen/TransUNet:
        trainer.py:  loss = 0.5 * loss_ce + 0.5 * loss_dice
        utils.py :  DiceLoss 先 softmax(dim=1) + target one-hot, smooth = 1e-5,
                    逐类算 1-(2*Σ(s*t)+s)/(Σt²+Σs²+s), 再按类别宏平均 (loss/n_classes);
                    且是对"整个 batch"求和, 不是按样本分别求.

    单通道适配: C=2 → 前景 (p, y) 与背景 (1-p, 1-y) 两类, sigmoid 概率与 2 类
    softmax 等价; CE 用 BCEWithLogits 等价替换(数学恒等).

    Parameters:
    -----------
    - output: 模型输出(logits), 形状 (batch_size, 1, 128, 128)
    - target: 目标值, 形状 (batch_size, 1, 128, 128)
    - bce_weight: BCE(CE) 损失权重, 默认 0.5
    - dice_smooth: Dice 平滑项, 默认 1e-5 (官方 _dice_loss 内的写死值)
    """
    y = target.float()
    p = torch.sigmoid(output).float()
    ce = F.binary_cross_entropy_with_logits(output, y)

    def _dice(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # 官方 _dice_loss: 整 batch 求和 (非按样本)
        inter = (a * b).sum()
        return 1.0 - (2.0 * inter + dice_smooth) / ((b * b).sum() + (a * a).sum() + dice_smooth)

    dice = 0.5 * (_dice(p, y) + _dice(1.0 - p, 1.0 - y))  # n_classes=2 → 类别宏平均
    return bce_weight * ce + (1.0 - bce_weight) * dice


def cal_loss_dice(output: torch.Tensor, target: torch.Tensor,
                  smooth: float = 0.01) -> torch.Tensor:
    """
    Attention U-Net 原版损失: Sørensen-Dice loss.

    论文 (Oktay et al., MIDL 2018) 原文: "The models are trained using the
    Sorensen-Dice loss [20] defined over all semantic classes, which is
    experimentally shown to be less sensitive to class imbalance."
    论文未给公式与平滑项; 以下按作者官方代码 Attention-Gated-Networks
    (models/layers/loss.py 的 SoftDiceLoss) 补齐: smooth = 0.01, 分子分母各加,
    逐图逐类算 Dice 后对 batch×类别取平均, 最后 1 - 均值.

    单通道适配: 输出为 1 通道 logits, sigmoid 后为前景概率 p, 背景概率 1-p,
    等价于 C=2 的 softmax 情形, 故按"前景/背景"两类各算一次 Dice.

    Parameters:
    -----------
    - output: 模型输出(logits), 形状 (batch_size, 1, 128, 128)
    - target: 目标值, 形状 (batch_size, 1, 128, 128)
    - smooth: 平滑项, 默认 0.01 (官方 SoftDiceLoss 原值)
    """
    p = torch.sigmoid(output).float()
    y = target.float()
    q, m = 1.0 - p, 1.0 - y

    inter_fg = (p * y).sum(dim=(1, 2, 3)) + smooth
    union_fg = p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3)) + smooth
    inter_bg = (q * m).sum(dim=(1, 2, 3)) + smooth
    union_bg = q.sum(dim=(1, 2, 3)) + m.sum(dim=(1, 2, 3)) + smooth

    # 官方: score = Σ_{b,c}(2*inter/union), loss = 1 - score/(B*C); 此处 C=2
    score = 0.5 * (2.0 * inter_fg / union_fg + 2.0 * inter_bg / union_bg)
    return 1.0 - score.mean()


# ---------------------------------------------------------------------------
# U-Net 原版损失 (Ronneberger et al., MICCAI 2015, 第 3 节)
#   式(1): E = -Σ_x w(x) * log p_{l(x)}(x)          (逐像素 softmax + 交叉熵, 带权重图)
#   式(2): w(x) = w_c(x) + w_0 * exp(-(d1(x)+d2(x))**2 / (2*sigma**2))
# ---------------------------------------------------------------------------
UNET_W0 = 10.0        # 论文原句: "In our experiments we set w_0 = 10 and sigma ≈ 5 pixels."
UNET_SIGMA = 5.0
# 式(2) 中类别频率平衡项 w_c = 1/f_c 所需的前景像素频率.
# 论文仅写 "the weight map to balance the class frequencies", 未给公式与数值;
# 此处取验证集实测的像素级变化占比 (tp+fn)/total = 102892/5947392 ≈ 0.0173 作为估计
# (与训练集同源同分布); 如需精确可换成训练集实测值.
UNET_FG_PIXEL_FREQ = 0.0173

# 权重图按"单张样本"缓存: 数据集中的样本是加载时一次性预生成的静态张量, 且小斑块移除
# (remove_lines_forbatch) 是确定性的, 故同一张图每 epoch 得到的权重图完全一致.
# 按样本缓存可跨 epoch 复用 (若按 batch 缓存则因 shuffle 永远命中不了).
_UNET_WM_CACHE = {}
_UNET_WM_CACHE_MAX = 4096


def _unet_weight_map_single(mask2d: np.ndarray) -> np.ndarray:
    """对单张 0/1 标签图按式(2)计算权重图, 返回 (1, H, W) float32."""
    h, w = mask2d.shape
    fg_w = 1.0 / UNET_FG_PIXEL_FREQ
    bg_w = 1.0 / (1.0 - UNET_FG_PIXEL_FREQ)
    mask = mask2d > 0.5
    wmap = np.where(mask, fg_w, bg_w).astype(np.float32)  # w_c 项

    struct = ndimage.generate_binary_structure(2, 1)
    labeled, n = ndimage.label(mask, structure=struct)
    if n > 0:
        # 各连通斑块的轮廓像素 (含所属斑块 id)
        border = mask & ~ndimage.binary_erosion(mask, structure=struct)
        pts = np.argwhere(border)                # (M, 2) 边界像素坐标 (y, x)
        pt_ids = labeled[border]
        yy, xx = np.mgrid[0:h, 0:w]
        query = np.stack([yy.ravel(), xx.ravel()], axis=1)

        k = min(16, len(pts))
        dist, idx = cKDTree(pts).query(query, k=k, workers=-1)
        dist = np.asarray(dist).reshape(-1, k)
        ids = pt_ids[np.asarray(idx).reshape(-1, k)]

        d1 = dist[:, 0]                          # 到最近斑块轮廓的距离
        if k == 1:
            d2 = d1
        else:
            # 首个"与最近邻不同斑块"的近邻距离 → d2; 不存在(仅一个斑块)时退化 d2 = d1
            diff = ids != ids[:, :1]
            first = np.argmax(diff, axis=1)
            d2 = np.where(diff.any(axis=1), dist[np.arange(len(d1)), first], d1)

        boost = UNET_W0 * np.exp(-((d1 + d2) ** 2) / (2.0 * UNET_SIGMA ** 2))
        wmap = wmap + boost.astype(np.float32).reshape(h, w)

    return wmap.reshape(1, h, w)


def _unet_weight_map(target: torch.Tensor) -> torch.Tensor:
    """按式(2)为整个 batch 计算权重图, 返回与 target 同形状的张量.

    论文的 "cell" 在本项目映射为"变化连通斑块":
      d1 = 到最近变化斑块轮廓的像素距离
      d2 = 到最近"其他"斑块轮廓的像素距离 (图中仅一个斑块时退化为 d2 = d1)
    整图无变化(n=0)时退化为常量图 w = w_c.
    """
    arr = target.detach().cpu().numpy()
    out = np.empty(arr.shape, dtype=np.float32)
    for b in range(arr.shape[0]):
        single = np.ascontiguousarray(arr[b])
        key = (single.shape, single.tobytes())
        cached = _UNET_WM_CACHE.get(key)
        if cached is None:
            cached = _unet_weight_map_single(single[0])
            if len(_UNET_WM_CACHE) >= _UNET_WM_CACHE_MAX:
                _UNET_WM_CACHE.clear()
            _UNET_WM_CACHE[key] = cached
        out[b] = cached
    return torch.from_numpy(out).to(target.device)


def cal_loss_weighted_ce(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    U-Net 原版损失: 逐像素 softmax + 加权交叉熵 (论文第 3 节式(1)(2)).

    C=2 时 softmax+交叉熵与 BCEWithLogits 数学恒等, 故单通道 logits 可直接代入.

    归一化说明: 论文式(1) 为逐像素"求和"(其 batch=1); 本项目 unet 的 batch_size=8
    且配 grad_clip=1.0, 直接求和会使量级失控(每步梯度都被裁剪), 故改用逐像素加权平均
    Σ(w*l)/Σw —— 与论文原式仅差一个归一化常数, 权重图的相对语义完整保留.

    Parameters:
    -----------
    - output: 模型输出(logits), 形状 (batch_size, 1, 128, 128)
    - target: 目标值, 形状 (batch_size, 1, 128, 128)
    """
    target = target.float()
    weight = _unet_weight_map(target)
    loss = F.binary_cross_entropy_with_logits(output, target, weight=weight,
                                              reduction='none')
    return loss.sum() / weight.sum()



# 模型 → 原始 loss 类型映射 (逐模型对齐原始文献 / 官方实现)
ORIGINAL_LOSS_MAP = {
    'unet':      'weighted_ce',  # U-Net: 逐像素 softmax + 加权交叉熵 + weight map (w0=10, sigma=5)
    'attunet':   'dice',         # Attention U-Net: Sørensen-Dice loss (smooth=0.01)
    'fcn':       'bce',
    'segformer': 'bce',          # SegFormer: 未加权 CrossEntropy ≡ BCE
    'transunet': 'bce_dice',     # TransUNet 官方实现: 0.5*CE + 0.5*Dice (smooth=1e-5)
    'vit':       'bce',          # ViT: 官方未加权 softmax CE (无 label smoothing/mixup), C=2 等价 BCE
    'swin':      'bce_ls',       # Swin: 官方关闭 mixup 时的 label smoothing 分支 (timm LabelSmoothingCE, eps=0.1)
    # 其余新增模型: 论文原版损失无法直接用于"单网络逐像素二分类"口径,
    # 暂回退到像素级 BCE + Dice (cgan 的对抗损失需双网络, 本项目只移植生成器;
    # convlstm 原论文为回归任务, 没有对应的分割损失)
    'convlstm':  'bce_dice',
    'cgan':      'bce_dice',
}

_ORIGINAL_LOSS_FNS = {
    'bce':         cal_loss_bce,
    'bce_ls':      cal_loss_bce_ls,
    'dice':        cal_loss_dice,
    'weighted_ce': cal_loss_weighted_ce,
    'bce_dice':    cal_loss_bce_dice,
}


def get_original_loss_fn(model_type: str):
    """
    按模型返回其原始损失函数; 未匹配时回退到 bce_dice.
    """
    kind = ORIGINAL_LOSS_MAP.get(model_type, 'bce_dice')
    return _ORIGINAL_LOSS_FNS[kind]


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
    def __init__(self, device='cpu', threshold: float = 0.5):
        """
        Parameters:
        -----------
            device: str, 计算设备, 默认'cpu'
            threshold: float, sigmoid 输出二值化阈值, 默认0.5
        """
        self.device = device
        self.threshold = threshold
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
            
        # 将预测值转换为0/1（假设是二分类），阈值可配置
        pred_binary = (self.all_pred > self.threshold).float()
        target_binary = self.all_target
        
        # 计算混淆矩阵分量
        TP = ((pred_binary == 1) & (target_binary == 1)).float().sum().item()
        FP = ((pred_binary == 1) & (target_binary == 0)).float().sum().item()
        TN = ((pred_binary == 0) & (target_binary == 0)).float().sum().item()
        FN = ((pred_binary == 0) & (target_binary == 1)).float().sum().item()
        
        total_pixels = TP + FP + TN + FN
        
        # 计算各项指标
        self.current_metrics = {
            'accuracy(OA)': (TP + TN) / (total_pixels + 1e-6),
            'precision(UA)': TP / (TP + FP + 1e-6),
            'recall(PA)': TP / (TP + FN + 1e-6),
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
            f"  Accuracy(OA): {metrics['accuracy(OA)']:.4f}, Precision(UA): {metrics['precision(UA)']:.4f}\n"
            f"  Recall(PA): {metrics['recall(PA)']:.4f}, F1-Score: {metrics['f1_score']:.4f}\n"
            f"  IoU: {metrics['iou']:.4f}, TP/FP/TN/FN: {metrics['tp']:.0f}/{metrics['fp']:.0f}/{metrics['tn']:.0f}/{metrics['fn']:.0f}"
        )
