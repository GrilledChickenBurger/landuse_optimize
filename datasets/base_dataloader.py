import os
from typing import Dict, List, Tuple, Union, Any
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 从其他文件导入数据加载函数
# 这里使用导入语句，但具体导入路径需要根据实际情况调整
# 假设这两个函数从AAAA_LU_design.utilities.data模块导入
from .training_dataloader import load_training_data
from .optimize_dataloader import load_optimize_data
from libs.const import ALL_FACTORS
from utils.dataset_utils import get_year_data

def get_dataset(type:str, data_config:Dict[str, Any]) -> data.Dataset:
    """
    获取数据集

    Parameters:
    ----------
    type: str
        数据集类型，如 "optimize"、"training" 等
    data_config: Dict
        数据集配置，包含数据路径、训练/验证集划分等信息
    """
    data_config_dict = vars(data_config)
    shape = data_config_dict["shape"]
    transform = transforms.Compose(
        [
            transforms.Resize(shape[-2]),
            transforms.CenterCrop(shape[-2:]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            #  transforms.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0)),
            #  transforms.RandomRotation(degrees=30),
        ]
    )

    if type == "optimize":
        return OptimizeDataset(
            root=data_config_dict["root"],
            shape=data_config_dict["shape"],
            year=data_config_dict["year"],
            factor_list=data_config_dict["factor_list"],
            ES_list=data_config_dict["es_list"],
            use_parallel=data_config_dict.get("use_parallel", False),
            max_workers=data_config_dict.get("max_workers", 1),
        )
    elif type == "training":
        return TrainingDataset(
            root=data_config_dict["root"],
            split=data_config_dict.get("split", "train"),
            shape=data_config_dict["shape"],
            condition_list=data_config_dict["factor_list"],
            only_changed=data_config_dict["only_changed"],
            pack=data_config_dict.get("pack", [-1, -1]),
            transform=transform if data_config_dict.get("transform", True) else None,
            normalize=data_config_dict.get("normalize", False),
        )
    elif type == "statistic":
        return StatisticDataset(
            root=data_config_dict["root"],
            shape=data_config_dict["shape"],
            year=data_config_dict["year"],
            factor_list=data_config_dict["factor_list"],
        )
    else:
        raise ValueError(f"未知数据集类型: {type}")


class BaseDataset(data.Dataset):
    """
    基础数据集类，提供数据集的基本功能

    Parameters:
    ----------
    root: str
        数据集根目录
    shape: list
        数据形状，如 [1, 128, 128]
    factor_list: list
        驱动因子列表
    """
    def __init__(self, root: str, shape: List, factor_list: List, *args):
        """
        初始化基础数据集
        
        Parameters:
        ----------
        root: 数据集根目录
        shape: 数据形状
        factor_list: 驱动因子列表
        *args: 其他参数，传递给数据加载函数
        """
        self.root = root
        self.shape = shape
        self.factor_list = factor_list
        # 保存其他参数，供子类使用
        self.args = args
        # 初始化xf_pack_list为None，子类必须在自己的__init__中设置这个属性
        self.xf_pack_list = None

    def __getitem__(self, i):
        if self.xf_pack_list is None:
            raise NotImplementedError("子类必须初始化xf_pack_list属性")
        return self.xf_pack_list[i]

    def __len__(self) -> int:
        if self.xf_pack_list is None:
            raise NotImplementedError("子类必须初始化xf_pack_list属性")
        return len(self.xf_pack_list)


class TrainingDataset(BaseDataset):
    """
    训练数据集类（对应原LUDataset）
    
    Parameters:
    ----------
    root: str
        数据集根目录
    split: str
        数据集划分，如'train'或'valid'
    shape: list
        数据形状
    condition_list: list
        条件因子列表
    only_changed: bool
        是否只加载发生变化的图像
    pack: list
        打包对应年份间隔的数据
    transform: transforms.Compose
        数据增强方法
    normalize: bool
        是否归一化数据
    """
    def __init__(self, root: str, split: str, shape: List, condition_list: List,
                 only_changed: bool, pack: List, transform=None, normalize: bool=False):
        # 调用父类初始化，但不在这里加载数据
        super().__init__(root, shape, condition_list)
        # 验证条件列表
        if condition_list:
            assert set(condition_list).issubset(ALL_FACTORS), \
                "condition_list 包含无效的条件因子"
        
        # 保存参数
        self.split = split
        self.only_changed = only_changed
        self.pack = pack
        self.transform = transform
        
        # 加载数据
        self.xf_pack_list = load_training_data(
            root=root,
            split=split,
            shape=shape,
            condition_list=condition_list,
            only_changed=only_changed,
            pack=pack,
            transform=transform,
            normalize=normalize,
        )


class OptimizeDataset(BaseDataset):
    """
    优化数据集类（对应原SADataset）
    
    Parameters:
    ----------
    root: str
        数据集根目录
    shape: list
        数据形状
    year: str
        年份，如'10', '15', '20'
    factor_list: list
        条件因子列表
    ES_list: list
        生态系统服务列表
    use_parallel: bool
        是否并行加载数据
    max_workers: int
        并行加载数据的线程数
    """
    def __init__(self, root: str, shape: List, year: str, factor_list: List, ES_list: List, use_parallel: bool = False, max_workers: int = 2):
        # 调用父类初始化，但不在这里加载数据
        super().__init__(root, shape, factor_list)
        
        # 验证条件列表
        assert set(factor_list).issubset(ALL_FACTORS), \
            "factor_list 包含无效的条件因子"
        
        # 保存参数
        self.year = year
        self.ES_list = ES_list if ES_list is not None else []
        
        # 加载数据
        self.xf_pack_list = load_optimize_data(
            root=root,
            shape=shape,
            condition_list=factor_list,
            ES_list=ES_list,
            year=year,
            use_parallel=use_parallel,
            max_workers=max_workers,
        )


class StatisticDataset(BaseDataset):
    """
    统计数据集类 — 仅加载驱动因子和源年 LU，用于计算驱动因子分布参数。

    与 OptimizeDataset 相比，跳过 ES 数据和目标年 LU 的磁盘读取，约节省 30% I/O。

    Parameters:
    ----------
    root: str
        数据集根目录
    shape: list
        数据形状，如 [1, 128, 128]
    year: str
        年份，如 '10', '15', '20'
    factor_list: list
        驱动因子列表
    """
    def __init__(self, root: str, shape: List, year: str, factor_list: List):
        super().__init__(root, shape, factor_list)

        assert set(factor_list).issubset(ALL_FACTORS), \
            "factor_list 包含无效的驱动因子"

        root_ori = os.path.join(root, f"valid_{year}")
        # 只读因子 + LU，跳过 RA / ES / tgt-year LU
        cond_list = factor_list + ["LU"]
        xf_ori = get_year_data(root_ori, cond_list, shape, max_workers=1, use_parallel=False)

        factor_num = len(factor_list)
        self.xf_pack_list = [
            {
                "factors": xf_ori[idx, :factor_num],   # (F, H, W)
                "src_lu": xf_ori[idx, factor_num:],    # (1, H, W)
            }
            for idx in range(xf_ori.shape[0])
        ]
