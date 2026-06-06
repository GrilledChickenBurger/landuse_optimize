# 临时脚本: 用 attunet 模型预测所有 optimize 样本的变化区域并可视化
# 使用方法: 在项目根目录直接 python scripts/temp_pred_visualize.py
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.base_dataloader import get_dataset
from utils.tools import get_yaml, ensure_folder, visualize_tensor_list
from libs.const import COLOR_MAP, DEFAULT_CONDITION_LIST
import utils.runner_utils as rUtils
from pipelines.experiment import process_batch


MODEL_PATH = r"checkpoints\attunet_best_260212.pth"
OUTPUT_FOLDER = "_temp/pred_visualize"


def main():
    cfg = get_yaml("configs/optimizerConfig.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device.use_gpu else "cpu")
    print(f"Using device: {device}")

    ensure_folder(OUTPUT_FOLDER)

    dataset = get_dataset(type="optimize", data_config=cfg.data)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"Total samples: {len(dataloader)}")

    model_type = cfg.model.type
    type_config = getattr(cfg.model, model_type, None)
    factors_ch = len(cfg.data.factor_list)
    ann_model = rUtils.load_ANN_model(
        model_type, MODEL_PATH, factors_ch + 6, 1, device, type_config)

    for i, data_pack in enumerate(dataloader):
        index = data_pack[5][0]
        data_dict = process_batch(data_pack, cfg)

        year = index.split('_')[0]
        pred_diff = rUtils.get_pred_diff(
            ann_model, data_dict['ori_plan'], data_dict['factors'], device,
            year=year, condition_list=DEFAULT_CONDITION_LIST,
            normalize=True, sigmoid_threshold=0.5)

        real_diff = (data_dict['ori_plan'] != data_dict['final_plan']).float()

        save_path = os.path.join(OUTPUT_FOLDER, f"{index}.png")
        visualize_tensor_list(
            {
                '2010 LU': data_dict['ori_plan'],
                '2015 LU': data_dict['final_plan'],
                'Predicted Change': pred_diff,
                'Real Change': real_diff,
            },
            main_title=index,
            save_path=save_path,
            color_map=COLOR_MAP,
            layout=(1, 4),
            figsize=(20, 5),
        )

        if (i + 1) % 50 == 0:
            print(f"  saved {i+1} / {len(dataloader)}")

    print(f"Done. Images saved to {OUTPUT_FOLDER}/")


if __name__ == '__main__':
    main()
