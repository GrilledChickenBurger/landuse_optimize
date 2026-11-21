# 方案可行性检查 (硬约束 + 类别约束), 供生成端在评估前判定并丢弃不可行解.
import torch
from typing import Tuple, Optional

# 建设用地 5 类标签: cls_restrict 要求原 51/53 像素只能改为这些类别
_FIVE_CLASS_LABELS = (51, 52, 53)


def count_violations(plan: torch.Tensor,
                     ori_plan: torch.Tensor,
                     hard_restrict: torch.Tensor,
                     cls_restrict: Optional[torch.Tensor] = None) -> Tuple[int, int]:
    """
    统计方案相对硬约束与类别约束的违规像素数.\n
    Parameters:
    ----------
        plan: tensor, [1,1,H,W], 待检查的土地利用方案
        ori_plan: tensor, [1,1,H,W], 原始土地利用方案 (基准)
        hard_restrict: tensor, [1,1,H,W], 0-1, 1 表示可修改区域, 0 表示不可修改
        cls_restrict: tensor, [1,1,H,W], 0-1, 可选. 1 表示该像素原类型为 51/53, 只能改为 5 类

    Returns:
    --------
        (hard_viol, cls_viol): 两类违规像素数
            hard_viol: 修改了 hard_restrict==0 (不可修改) 的像素数
            cls_viol:  将原 51/53 像素改成非 {51,52,53} 的像素数 (cls_restrict 为 None 时恒为 0)
    """
    # 相对基准方案发生变化的位置
    changed = (plan != ori_plan)
    # 1. 硬约束: 变化落在不可修改区域 (hard_restrict==0)
    hard_viol = int(torch.count_nonzero(changed & ~hard_restrict.bool()).item())
    # 2. 类别约束: 原 51/53 像素被改成非 5 类
    cls_viol = 0
    if cls_restrict is not None:
        five = torch.tensor(_FIVE_CLASS_LABELS, device=plan.device, dtype=plan.dtype)
        is_five = torch.isin(plan, five)
        cls_viol = int(torch.count_nonzero(
            cls_restrict.bool() & changed & ~is_five).item())
    return hard_viol, cls_viol


def is_feasible(plan: torch.Tensor,
                ori_plan: torch.Tensor,
                hard_restrict: torch.Tensor,
                cls_restrict: Optional[torch.Tensor] = None) -> bool:
    """
    方案的可行性判定: 硬约束与类别约束均无违规时为 True.\n
    Parameters:
    ----------
        plan / ori_plan / hard_restrict / cls_restrict: 含义同 count_violations
    """
    hard_viol, cls_viol = count_violations(plan, ori_plan, hard_restrict, cls_restrict)
    return hard_viol == 0 and cls_viol == 0
