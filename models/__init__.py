"""
预测LU变化的ANN模型
"""
from typing import Optional
from .unet import AttU_Net, U_Net
from .fcn import FCN
from .segformer import Segformer
from .transunet import TransUNet
from .convlstm import ConvLSTM_Seg
from .cgan import CGAN
from .vit import ViT
from .swin import Swin

def create_model(model_type:str, in_ch:int, out_ch:int, model_config:Optional[dict]=None):
    print(f"\n================= 开始初始化模型【{model_type}】... =================")
    model_config = model_config or {}
    # 处理 argparse.Namespace 对象，转换为字典
    if hasattr(model_config, '__dict__'):
        model_config = vars(model_config)

    if model_type == 'attunet':
        return AttU_Net(in_ch, out_ch, **model_config)
    elif model_type == 'fcn':
        return FCN(in_ch, out_ch, **model_config)
    elif model_type == 'unet':
        return U_Net(in_ch, out_ch, **model_config)
    elif model_type == 'segformer':
        return Segformer(channels=in_ch, num_classes=out_ch, **model_config)
    elif model_type == 'transunet':
        return TransUNet(in_channels=in_ch,
                          class_num=out_ch,
                          **model_config)
    elif model_type == 'convlstm':
        return ConvLSTM_Seg(in_ch, out_ch, **model_config)
    elif model_type == 'cgan':
        return CGAN(in_ch, out_ch, **model_config)
    elif model_type == 'vit':
        return ViT(in_ch, out_ch, **model_config)
    elif model_type == 'swin':
        return Swin(in_ch, out_ch, **model_config)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")