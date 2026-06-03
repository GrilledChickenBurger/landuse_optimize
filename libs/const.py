import os

# 每个像素代表的面积转换为公顷 (1公顷 = 10,000平方米)
PIXEL_AREA_IN_HECTARES = (30 * 30) / 10000

ALL_LU_LABELS_SIMPLIFIED = [1, 2, 3, 4, 5, 6]
ALL_LU_LABELS = [11, 12, 21, 22, 23, 24, 31, 32, 33, 41, 42, 43, 45, 46, 51, 52, 53, 65, 66]
ALL_LU_LABELS_DICT = {
    # 耕地
    1: "耕地",
    11: "水田",
    12: "旱地",
    # 林地
    2: "林地",
    21: "有林地",
    22: "灌木林",
    23: "疏林地",
    24: "其他林地",
    # 草地
    3: "草地",
    31: "高覆盖度草地",
    32: "中覆盖度草地",
    33: "低覆盖度草地",
    # 水域
    4: "水域",
    41: "河渠",
    42: "湖泊",
    43: "水库坑塘",
    44: "永久性冰川雪地",
    45: "滩涂",
    46: "滩地",
    # 城乡、工矿、居民用地
    5: "建设用地",
    51: "城镇用地",
    52: "农村居民点",
    53: "其他建设用地",
    # 未利用地
    6: "未利用地",
    61: "沙地",
    62: "戈壁",
    63: "盐碱地",
    64: "沼泽地",
    65: "裸土地",
    66: "裸岩石质地",
    67: "其他",
}
ALL_ES_TYPES = ["food", "carbon", "water", "culture"]
ALL_ES_TYPES_DICT = {
    "food": "粮食生产",
    "carbon": "碳储量",
    "water": "水源涵养",
    "culture": "文化服务",
}
ALL_SCENARIOS = ["default", "enterprise", "farmer", "tourist"]
ALL_SCENARIOS_DICT = {
    "default": "base",
    "enterprise": "enterprise",
    "farmer": "farmer",
    "tourist": "tourist",
}
# 水源涵养计算：不同生态系统的地表径流系数 (按 LU 二级类精确赋值)
# 数值单位为小数 (原始表为百分比, 除以 100 转换)
# 未在原表列出的二级类按其一级类兜底:
#   水域 (4x) -> 0   建设用地 (5x) -> 1   未利用地 (6x) -> 1
WATER_SUPPLY_PARAM = {
    # 耕地
    11: 0.3470,   # 水田
    12: 0.4969,   # 旱地
    # 林地
    21: 0.02038,  # 有林地
    22: 0.0420,   # 灌木林
    23: 0.1920,   # 疏林地
    24: 0.1222,   # 其他林地
    # 草地
    31: 0.0478,   # 高覆盖度草地
    32: 0.08785,  # 中覆盖度草地
    33: 0.1827,   # 低覆盖度草地
    # 水域 (按一级类 4 -> 0 兜底)
    41: 0.0, 42: 0.0, 43: 0.0, 44: 0.0, 45: 0.0, 46: 0.0,
    # 建设用地 (按一级类 5 -> 1 兜底)
    51: 1.0, 52: 1.0, 53: 1.0,
    # 未利用地 (按一级类 6 -> 1 兜底)
    61: 1.0, 62: 1.0, 63: 1.0, 64: 1.0, 65: 1.0, 66: 1.0, 67: 1.0,
}

# 粮食生产计算：修正系数
FOOD_SUPPLY_PARAM = {
    'PODU': 0.3317,
    'GD': 0.1887,
    'WATER': 0.1592,
    'DL': 0.1297,
    'CITY': 0.1907,
}

# 像素数阈值，小于该值判断为未发生变化，不进入模拟退火过程
PIXEL_NUM_THRESHOLD_ISCHANGE = 30
# 像素数阈值，小于该值将从diff中移除
PIXEL_NUM_THRESHOLD_REMOVE = 20

# 保存原始默认值，用于实验后恢复（避免跨实验状态泄漏）
_DEFAULT_PIXEL_NUM_THRESHOLD_ISCHANGE = PIXEL_NUM_THRESHOLD_ISCHANGE
_DEFAULT_WATER_SUPPLY_PARAM = WATER_SUPPLY_PARAM.copy()


def apply_const_overrides(score_cfg) -> None:
    """
    根据 score_cfg.const 覆写本模块的模块级常量, 仅在 ablation pipeline 入口调用一次.
    其它 pipeline 不调用, 常量保持原值, 行为不变.

    注意：本函数修改全局状态。若同进程顺序运行多个不同 score_cfg 的实验，
    前面的覆写会泄漏到后面。实验结束后建议调用 restore_const_defaults() 恢复默认值。
    """
    global PIXEL_NUM_THRESHOLD_ISCHANGE, WATER_SUPPLY_PARAM
    c = score_cfg.const
    PIXEL_NUM_THRESHOLD_ISCHANGE = c.pixel_num_threshold_ischange
    # water_supply_param 在 yaml 中为 dict-like, 经 get_namespace 后会成为 Namespace
    if hasattr(c.water_supply_param, '__dict__'):
        # 将 Namespace 还原为 {int: float} dict, 保持 const.WATER_SUPPLY_PARAM 原有结构
        WATER_SUPPLY_PARAM = {int(k): v for k, v in vars(c.water_supply_param).items()}
    else:
        WATER_SUPPLY_PARAM = {int(k): v for k, v in dict(c.water_supply_param).items()}


def restore_const_defaults() -> None:
    """恢复 apply_const_overrides 修改前的模块级常量默认值，防止跨实验状态泄漏。"""
    global PIXEL_NUM_THRESHOLD_ISCHANGE, WATER_SUPPLY_PARAM
    PIXEL_NUM_THRESHOLD_ISCHANGE = _DEFAULT_PIXEL_NUM_THRESHOLD_ISCHANGE
    WATER_SUPPLY_PARAM = _DEFAULT_WATER_SUPPLY_PARAM.copy()

ALL_PACK = [[10, 15], [15, 20], [10, 20], [-1, -1]]
# 每个pack组合对应的变换增强重复次数
ALL_PACK_TRANSFORM_REPEAT_DICT = {
    (10, 15): 6,
    (15, 20): 2,
    (10, 20): 2,
    (-1, -1): 1,
}
ALL_YEARS = ["10", "15", "20"]
ALL_FACTORS = ["DEM", "DL", "NLMY", "GDP", "RK"]

COLOR_MAP = {
    0: (0,0,0),  # non-diff
    1: (1,1,1),  # diff
    False: (0,0,0),  # non-diff
    True: (1,1,1),  # diff
    # 耕地
    11: (254/255,250/255,207/255),  # 水田
    12: (253/255,216/255,65/255),      # 旱地
    # 林地
    21: (41/255,138/255,49/255),      # 有林地
    22: (157/255,205/255,74/255),    # 灌木林
    23: (172/255,229/255,138/255),     # 疏林地
    24: (125/255,226/255,150/255),     # 其他林地
    # 草地
    31: (156/255,151/255,41/255),      # 高覆盖度草地
    32: (189/255,183/255,115/255),    # 中覆盖度草地
    33: (220/255,185/255,137/255),    # 低覆盖度草地
    # 水域
    41: (1/255,0,247/255),       # 河渠
    42: (1/255,0,247/255),       # 湖泊
    43: (1/255,0,247/255),       # 水库坑塘
    44: (229/255,230/255,248/255),   # 永久性冰川雪地
    45: (68/255,104/255,220/255),   # 滩涂
    46: (38/255,125/255,248/255),     # 滩地
    # 城乡、工矿、居民用地
    51: (251/255,103/255,78/255),       # 城镇用地
    52: (252/255,142/255,45/255),       # 农村居民点
    53: (236/255,131/255,129/255),       # 其他建设用地
    # 未利用地
    61: (179/255,152/255,186/255),   # 沙地
    62: (129/255,124/255,182/255),   # 戈壁
    63: (152/255,51/255,197/255),   # 盐碱地
    64: (147/255,113/255,214/255),   # 沼泽地
    65: (183/255,89/255,206/255),   # 裸土地
    66: (184/255,88/255,208/255),   # 裸岩石质地
    67: (253/255,192/255,204/255),      # 其他
}
# SIMPLIFIED_COLOR_MAP={
#     1: (254/255,250/255,207/255),  # 耕地
#     2: (167/255,211/255,152/255),      # 林地
#     3: (208/255,226/255,192/255),      # 草地
#     4: (192/255,226/255,251/255),       # 水域
#     5: (250/255,201/255,207/255),       # 城乡、工矿、居民用地
#     6: (179/255,152/255,186/255),   # 未利用地
# }

SIMPLIFIED_COLOR_MAP={
    1: (199/255,217/255,169/255),  # 耕地
    2: (162/255,190/255,146/255),      # 林地
    3: (175/255,199/255,191/255),      # 草地
    4: (196/255,218/255,230/255),       # 水域
    5: (153/255,195/255,217/255),       # 城乡、工矿、居民用地
    6: (159/255,188/255,194/255),   # 未利用地
}

ES_COLOR_MAP={
    "food": (255/255,225/255,0/255),
    "carbon": (8/255,140/255,0/255),
    "water": (34/255,121/255,176/255),
    "culture": (254/255,192/255,193/255),
}

SCENARIO_COLOR_MAP = {
    "default": (175/255,199/255,191/255),
    "enterprise": (196/255,218/255,230/255),
    "farmer": (153/255,195/255,217/255),
    "tourist": (159/255,188/255,194/255),
}
SA_GA_COLOR_MAP = {"SA": "#5B9BD5", "GA": "#ED7D31"}

FACTOR_10_PARAMS_DICT={
    "DEM": {
        "miu": 1.0814407780597586,
        "epsilon": 1.788031763345373
    },
    "DL": {
        "miu": 2.522156872482891,
        "epsilon": 1.187615531096033
    },
    "GDP": {
        "miu": 7.693943775452838,
        "epsilon": 0.8684736515245268
    },
    "RK": {
        "miu": 1.0299328146645907,
        "epsilon": 0.7987507497713701
    }
}
FACTOR_15_PARAMS_DICT={
    "DEM": {
        "miu": 1.0814430603217478,
        "epsilon": 1.788033743629245
    },
    "DL": {
        "miu": 2.3430028285064752,
        "epsilon": 1.0569192558795943
    },
    "GDP": {
        "miu": 8.01455241977147,
        "epsilon": 0.8878913056045725
    },
    "RK": {
        "miu": 0.9983773746635167,
        "epsilon": 0.8456885410728755
    }
}
FACTOR_20_PARAMS_DICT={
    "DEM": {
        "miu": 1.0814407780597586,
        "epsilon": 1.788031763345373
    },
    "DL": {
        "miu": 1.6398164592274322,
        "epsilon": 0.8869015324486094
    },
    "GDP": {
        "miu": 8.468035998360396,
        "epsilon": 0.5520970106850966
    },
    "RK": {
        "miu": 1.029140928244634,
        "epsilon": 0.8556354944383957
    }
}

# 标准化中所有需要进行对数变换的因子
ALL_LOG_FACTORS = ["GDP", "RK", "DL"]

# 每个年份不同因子的归一化参数字典
FACTOR_PARAMS_DICT={
    "10": FACTOR_10_PARAMS_DICT,
    "15": FACTOR_15_PARAMS_DICT,
    "20": FACTOR_20_PARAMS_DICT,
}

# 原始数据根目录, 保存着tif文件配套的tfw等文件
# 若需要frg模型评估方案, 则待评估方案需要从此处复制tfw文件
# 可通过环境变量 LU_OPTIM_DATA_ROOT 覆盖
RAW_DATA_ROOT = os.environ.get('LU_OPTIM_DATA_ROOT',
                               'D:\\_programming\\_LAB_ASSIGMENTS\\AAAAA栅格分割')

# 默认驱动因子列表 (pred_diff 计算中 normalization 使用的条件因子)
DEFAULT_CONDITION_LIST = ["DEM", "DL", "GDP", "RK"]

FACTORS_NAME_DICT = {
    'RA': '耕地红线(不可变更)',
    'DEM': '海拔',
    'DL': '距道路距离',
    'NLMY': '农林牧渔产值',
    'GDP': '人均生产总值',
    'RK': '人口密度',
    'RAIN': '年均降水',
    'NJW': '年均气温',
    'ZSL': '年均蒸散量',
    'PODU': '坡度',
    'WATER': '距水源距离',
    'CITY': '距城市距离',
    'GD': '距农田距离',
}