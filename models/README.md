# Models Module

## 这个模块做什么

models 是 Stage-1 的**ANN 变化检测模型库**——给定驱动因子 + 历史 LU，预测每个像素未来是否变化。所有模型都是二分类语义分割（输入 10 通道、输出 1 通道概率图），通过统一的工厂函数 `create_model()` 创建。

Stage-1 训练得到的模型在 Stage-2 优化阶段被**复用**：`utils/runner_utils.get_pred_diff()` 加载它生成 `pred_diff`，作为 SA/GA 目标函数中 change_score 项的先验。模型的好坏直接决定 pred_diff 的质量，进而影响整个优化系统的搜索方向。

## 在项目中的位置

```
驱动因子 + 历史LU (10 通道)
       ↓
   [models 中的网络]  ← 训练阶段：pipelines/train.py
       ↓
   pred_diff (1 通道概率图)
       ↓
   ├─ 验证：pipelines/train.py → MetricsMeter (OA/UA/PA/F1/IoU)
   └─ 优化：utils/runner_utils.get_pred_diff() → calculator/cal_change_score
```

## 文件一览

| 文件           | 内容                                                                               |
| -------------- | ---------------------------------------------------------------------------------- |
| `__init__.py`  | 工厂函数 `create_model()`，根据 `model_type` 字符串分发到对应类                    |
| `fcn.py`       | `FCN`：VGG16 风格编码器 + 双线性上采样解码器                                       |
| `unet.py`      | `U_Net` 和 `AttU_Net`（注意力门控 U-Net）；文件中还有 R2U_Net、NestedUNet 等未启用 |
| `segformer.py` | `Segformer`：MiT 编码器 + 轻量 MLP 解码器                                          |
| `transunet.py` | `TransUNet`：ResNet 风格编码器 + ViT 瓶颈 + U-Net 解码器                           |
| `convlstm.py`  | `ConvLSTM_Seg`：4 级 ConvLSTM 编码器 + U-Net 风格解码器                            |
| `cgan.py`      | `CGAN`：pix2pix 的 U-Net 生成器（只含生成器，不含判别器）                          |
| `vit.py`       | `ViT`：ViT 编码器（cls token 丢弃）+ 轻量上采样分割头                              |
| `swin.py`      | `Swin`：Swin Transformer 4-stage 编码器 + 多尺度轻量解码器                         |

## 工厂函数 `create_model`

```python
create_model(model_type: str, in_ch: int, out_ch: int, model_config: dict = None)
```

- `model_type` ∈ {`attunet`, `unet`, `fcn`, `segformer`, `transunet`, `convlstm`, `cgan`, `vit`, `swin`}
- `in_ch` = `len(factor_list) + 6`（4 个驱动因子 + 6 类简化 LU 的 one-hot）= 10
- `out_ch` = 1（二分类变化掩码，输出为 raw logits，损失函数内部做 sigmoid）
- `model_config` 来自 `configs/trainConfig.yaml` 的 `model.<type>` 节；`segformer`、`transunet`、`convlstm`、`cgan`、`vit`、`swin` 均在 YAML 中配置，`attunet`/`unet`/`fcn` 使用代码默认值

工厂会把 `argparse.Namespace` 自动转成 `dict` 以兼容 YAML 加载方式。

## FCN

VGG16 风格的全卷积网络，是最简单的基线模型。

**整体结构**：5 个卷积块下采样到 4×4 → 三次双线性上采样回到 128×128，途中两次跳跃融合（element-wise add，FCN-8s 风格）。

**编码器**（每个 `conv3x3_block`：Conv-BN-ReLU 重复 2~3 次 + MaxPool2(2)）

| 块     | 重复次数 | 通道      | 输出尺寸 |
| ------ | -------- | --------- | -------- |
| block1 | 2        | 10 → 64   | 64×64    |
| block2 | 2        | 64 → 128  | 32×32    |
| block3 | 3        | 128 → 256 | 16×16    |
| block4 | 3        | 256 → 512 | 8×8      |
| block5 | 3        | 512 → 512 | 4×4      |

**解码器**（每个 `upsample` = 1×1 conv + `F.interpolate(bilinear)`）

| 步骤      | 通道      | 缩放 | 跳跃融合       |
| --------- | --------- | ---- | -------------- |
| upsample1 | 512 → 512 | ×2   | `+ block4`     |
| upsample2 | 512 → 256 | ×2   | `+ block3`     |
| upsample3 | 256 → 1   | ×8   | 直接到 128×128 |

**特点**：跳跃连接用**加法**而非 concat；上采样用双线性插值而非转置卷积；无 dropout；约 **~15M 参数**。

## U-Net

经典的 Ronneberger-2015 U-Net，编码器-解码器对称结构 + concat 跳跃连接。

**通道进度**：`filters = [64, 128, 256, 512, 1024]`

**编码器**（每个 `conv_block`：(Conv3×3-BN-ReLU)×2）

| 阶段 | 操作                     | 通道       | 输出尺寸 |
| ---- | ------------------------ | ---------- | -------- |
| e1   | Conv1                    | 10 → 64    | 128×128  |
| e2   | MaxPool2 + Conv2         | 64 → 128   | 64×64    |
| e3   | MaxPool2 + Conv3         | 128 → 256  | 32×32    |
| e4   | MaxPool2 + Conv4         | 256 → 512  | 16×16    |
| e5   | MaxPool2 + Conv5（瓶颈） | 512 → 1024 | 8×8      |

**解码器**（每个 `up_conv` = `nn.Upsample(nearest, ×2)` + Conv3×3-BN-ReLU，concat 跳跃后再 `conv_block`）

| 阶段 | up_conv    | concat 跳跃 | conv_block | 输出尺寸 |
| ---- | ---------- | ----------- | ---------- | -------- |
| d5   | 1024 → 512 | e4 (512)    | 1024 → 512 | 16×16    |
| d4   | 512 → 256  | e3 (256)    | 512 → 256  | 32×32    |
| d3   | 256 → 128  | e2 (128)    | 256 → 128  | 64×64    |
| d2   | 128 → 64   | e1 (64)     | 128 → 64   | 128×128  |

**最终输出头**：`Conv2d(64, 1, kernel_size=1)`

**特点**：跳跃连接用 **concat**（保留所有信息）；上采样用 nearest 插值 + 卷积（比转置卷积少棋盘效应）；无 dropout；约 **~31M 参数**。

## AttU-Net（注意力 U-Net）

在 U-Net 的每个跳跃连接前插入**注意力门**（Attention Gate, Oktay et al. 2018），编码器特征经过门控后再与解码器特征拼接，让网络学会"关注"重要区域、抑制无关响应。

**骨架**：与 U-Net 完全相同（同样的 `[64,128,256,512,1024]` 通道、同样的 `conv_block` 和 `up_conv`）。

**注意力门 `Attention_block(F_g, F_l, F_int)` 结构**：

| 分支 | 层                                               | 作用                             |
| ---- | ------------------------------------------------ | -------------------------------- |
| W_g  | Conv2d(F_l → F_int, 1×1) + BN                    | 投影 gating 信号 g（解码器特征） |
| W_x  | Conv2d(F_g → F_int, 1×1) + BN                    | 投影 skip 特征 x（编码器特征）   |
| psi  | Conv2d(F_int → 1, 1×1) + BN + Sigmoid            | 产生每像素注意力分数             |
| 合并 | `relu(W_g(g) + W_x(x))` → `psi(...)` → `x * psi` | 门控后的 skip 输出               |

**四个注意力门的维度**：

| 门   | F_g | F_l | F_int |
| ---- | --- | --- | ----- |
| Att5 | 512 | 512 | 256   |
| Att4 | 256 | 256 | 128   |
| Att3 | 128 | 128 | 64    |
| Att2 | 64  | 64  | 32    |

**特点**：相比 U-Net 仅增加 4 组 1×1 卷积三元组，参数量略增（约 **~34M**）；注意力机制对小尺度变化检测有帮助。

## SegFormer

来自 NVIDIA 2021 论文的纯 Transformer 分割模型，**MiT 分层编码器 + 轻量 MLP 解码器**。

### 编码器（MiT，固定 4 个 stage）

每个 stage 由 **Overlapping Patch Embedding**（`nn.Unfold` + 1×1 conv）+ N 个 Transformer 块组成。Transformer 块结构：`x = Attn(PreNorm(x)) + x; x = FF(PreNorm(x)) + x`。

| Stage | patch (k,s,p) | dim | heads | ff_exp | red_ratio | layers | 输出尺寸 |
| ----- | ------------- | --- | ----- | ------ | --------- | ------ | -------- |
| 1     | (7, 4, 3)     | 32  | 1     | 4      | 8         | 2      | 32×32    |
| 2     | (3, 2, 1)     | 64  | 2     | 4      | 4         | 2      | 16×16    |
| 3     | (3, 2, 1)     | 160 | 5     | 4      | 2         | 2      | 8×8      |
| 4     | (3, 2, 1)     | 256 | 8     | 4      | 1         | 2      | 4×4      |

**关键模块**：

- **`EfficientSelfAttention`**：Q 在原分辨率上 1×1 conv 得到；K/V 先做 `reduction_ratio` 步长的卷积**降低空间分辨率**，使注意力复杂度从 O(N²) 降到 O(N·N/R)
- **`MixFeedForward`**：`Conv1×1 → DsConv3×3 → GELU → Conv1×1`，其中 DsConv 把 3×3 卷积带回到 FFN，增强局部建模

### 参数含义

| 参数              | 含义                                     | YAML 值              |
| ----------------- | ---------------------------------------- | -------------------- |
| `dims`            | 各 stage 通道宽度                        | `[32, 64, 160, 256]` |
| `heads`           | 各 stage 注意力头数                      | `[1, 2, 5, 8]`       |
| `ff_expansion`    | FFN 隐藏层扩展倍率（hidden = dim × exp） | `[4, 4, 4, 4]`       |
| `reduction_ratio` | K/V 空间降维倍率                         | `[8, 4, 2, 1]`       |
| `num_layers`      | 每 stage 的 Transformer 块数             | 2                    |
| `decoder_dim`     | MLP 解码器中间通道宽度                   | 128                  |

### 解码器（`to_fused` + 融合）

1. 每个 stage 输出 → `Conv2d(dim_i → 128, 1×1)` → `Upsample(scale_factor=2^i)`（统一到 H/4 分辨率，即 32×32）
2. 4 个 stage 的特征沿通道 concat → `Conv2d(4×128 → 128, 1×1)` → `Conv2d(128 → 1, 1×1)`
3. 最后 `F.interpolate(..., size=128, bilinear)` 恢复到输入分辨率

**特点**：纯 Transformer 编码、极轻量解码；LayerNorm + GELU、**无 dropout**；约 **~3-4M 参数**（项目中最轻量）。

注意：代码默认 `ff_expansion=(8,8,4,4)`、`decoder_dim=256`，YAML 覆写为 `[4,4,4,4]` 和 `128` 以进一步减小模型。

## TransUNet

CNN-Transformer 混合模型：ResNet 风格编码器提取多尺度特征 → ViT 在瓶颈处建模全局依赖 → U-Net 风格解码器逐步恢复分辨率。

### 编码器

| 步骤 | 操作 | 通道 | 输出尺寸 | 保存为 |
|---|---|---|---|---|
| stem | Conv2d(k=7, s=2, pad=3) + BN + ReLU | 10 → 32 | 64×64 | x1 |
| encoder1 | EncoderBottleneck(stride=2) | 32 → 64 | 32×32 | x2 |
| encoder2 | EncoderBottleneck(stride=2) | 64 → 128 | 16×16 | x3 |
| encoder3 | EncoderBottleneck(stride=2) | 128 → 256 | 8×8 | 送入 ViT |

**`EncoderBottleneck`（ResNet 残差块）**：1×1(in→width) → BN+ReLU → 3×3(width→width, stride=2) → BN+ReLU → 1×1(width→out) → BN；`+` downsample(1×1 stride conv + BN)；ReLU。

### ViT 瓶颈

输入 8×8×256 特征图，以 **patch_dim=1**（每像素一个 token，共 64 个 token）送入 Vision Transformer。

- 添加可学习 `cls_token` 和位置编码
- 4 个 `TransformerEncoderBlock`，每个：MultiHeadAttention(heads=4) + Dropout(0.1) → 残差 → LayerNorm → MLP(256→512→256) + Dropout(0.1) → 残差 → LayerNorm
- 输出去掉 cls_token 后恢复为 8×8×256
- 后接 Conv3×3-BN-ReLU（256→128）

### 解码器

每个 `DecoderBottleneck` = Upsample(bilinear, ×2) → concat 跳跃 → Conv3×3-BN-ReLU ×2

| 步骤 | up_conv | concat 跳跃 | 输出通道 | 输出尺寸 |
|---|---|---|---|---|
| decoder1 | 256→64 | x3 (128) | 64 | 16×16 |
| decoder2 | 128→32 | x2 (64) | 32 | 32×32 |
| decoder3 | 64→16 | x1 (32) | 16 | 64×64 |
| decoder4 | 16→4 | 无 | 4 | 128×128 |

**最终输出头**：`Conv2d(4, 1, kernel_size=1)`

### 参数含义

| 参数 | 含义 | YAML 值 |
|---|---|---|
| `img_dim` | 输入 H/W | 128 |
| `out_channels` | stem 基础通道，所有编码/解码宽度以此为基准缩放 | 32 |
| `head_num` | ViT 中多头注意力的头数 | 4 |
| `mlp_dim` | ViT MLP 隐藏层维度 | 512 |
| `block_num` | ViT Transformer 块数 | 4 |
| `patch_dim` | 仅用于计算 `vit_img_dim = img_dim // patch_dim = 8`，ViT 内部实际 patch_dim=1 | 16 |

**特点**：ViT 位于最窄处（8×8），三个编码器跳跃绕过 ViT 直连解码器；CNN 部分用 BN+ReLU，ViT 部分用 LN+GELU+Dropout(0.1)；约 **~10-13M 参数**。

## ConvLSTM（`convlstm.py`）

论文 Convolutional LSTM Network（Shi et al., NeurIPS 2015, arXiv:1506.04214）。PyTorch 官方仓库没有 ConvLSTM，论文官方代码是 Torch/Lua 版，故采用社区引用最多的 PyTorch 实现 [ndrplz/ConvLSTM_pytorch](https://github.com/ndrplz/ConvLSTM_pytorch)，`ConvLSTMCell` / `ConvLSTM` 与上游逐行一致。

**`ConvLSTMCell`**（门控在卷积特征图上逐元素运算，而非全连接）

```
combined = cat([x_t, h_{t-1}], dim=1)            # (C+H) 通道
i, f, o, g = split(Conv2d(combined))             # 4H 通道; i/f/o 过 sigmoid, g 过 tanh
c_t = f ⊙ c_{t-1} + i ⊙ g
h_t = o ⊙ tanh(c_t)
```

**`ConvLSTM`**：多层堆叠、逐时间步递推，输入 `(B,T,C,H,W)`（`batch_first=True`）或 `(T,B,C,H,W)`，返回 `(每层输出序列, 每层末态)`。

**`ConvLSTM_Seg`**（上游只有序列层，编解码网络为本项目新增）

| 阶段 | 模块 | 通道 | 分辨率 |
| ---- | ---- | ---- | ------ |
| enc1 | ConvLSTM | 10 → 64 | 128×128 |
| enc2 | MaxPool + ConvLSTM | 64 → 128 | 64×64 |
| enc3 | MaxPool + ConvLSTM | 128 → 256 | 32×32 |
| enc4 | MaxPool + ConvLSTM | 256 → 512 | 16×16 |
| bottleneck | MaxPool + ConvLSTM | 512 → 512 | 8×8 |

- 每级取**最后时刻**的隐状态 `h_T` 作为该级空间跳跃特征（T=1 时就是该级输出）
- 解码器：`up_conv`（nearest×2 + Conv3×3-BN-ReLU）+ concat 跳跃 + `conv_block`，通道 512→256→128→64，输出头 `Conv2d(64, 1, 1)`
- 输入兼容 `(B,C,H,W)`（自动按 T=1 处理）与 `(B,T,C,H,W)`；本项目默认 T=1，此时 ConvLSTM 退化为带输入/遗忘/输出门的门控卷积
- 参数量约 **~42.4M**

## 条件 GAN / pix2pix 生成器（`cgan.py`）

论文 Image-to-Image Translation with Conditional Adversarial Networks（Isola et al., CVPR 2017, arXiv:1611.07004），实现对齐 [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) 官方 `models/networks.py` 的 `UnetGenerator` / `UnetSkipConnectionBlock`。

**只移植生成器，不含判别器**：`create_model()` 返回单网络、训练循环是 `model(x) -> logits` 模式，生成器本身的 U-Net 结构即可作为逐像素预测网络单独训练并与其余模型公平对比。

**`UnetSkipConnectionBlock`**（自内向外递归构建，每级都是"下采样 → 子模块 → 上采样 + 同分辨率 concat"）

| 位置 | 下采样 | 上采样 | 归一化 |
| ---- | ------ | ------ | ------ |
| outermost | `Conv4×4 s2 p1` | `ConvTranspose4×4 s2 p1` + ReLU | 无 |
| innermost | `LeakyReLU(0.2)` + `Conv4×4 s2 p1` | `ConvTranspose4×4 s2 p1` + ReLU | 有 |
| 中间层 | `LeakyReLU(0.2)` + `Conv4×4 s2 p1` | `ConvTranspose4×4 s2 p1` + ReLU | 有 |

- 与官方的两处差异：① outermost 去掉官方 `nn.Tanh()`，默认输出 raw logits（本项目用 BCE/Focal-Dice），`output_activation="tanh"` 可还原官方行为；② 输入是 128×128，`num_downs` 由官方 256 版的 8 改为 **7**（128/2⁷ = 1，瓶颈 1×1）
- 默认 `ngf=64`、`norm="batch"`（官方 `define_G` 默认值，CycleGAN 用 instance）、`use_dropout=False`
- 参数量约 **~41.8M**

## ViT（`vit.py`）

论文 An Image is Worth 16x16 Words（Dosovitskiy et al., ICLR 2021, arXiv:2010.11929）。官方实现 [google-research/vision_transformer](https://github.com/google-research/vision_transformer) 只有 JAX/Flax 版且只做分类，本文件按官方结构用 PyTorch 重写并补分割头。

| 官方模块（Flax） | 本文件实现 |
| ---------------- | ---------- |
| `embedding`（Conv s2d + emb 合并） | `PatchEmbed`：`Conv2d(in_ch, dim, k=patch, stride=patch)` + flatten |
| `cls`（零初始化） + `AddPositionEmbs`（正态 std=0.02） | `cls_token` / `pos_embed` 两个 `nn.Parameter`，相加后 dropout |
| `Encoder1DBlock`（PreNorm） | `Encoder1DBlock`：`x = x + Attn(LN(x))`；`x = x + MLP(LN(x))` |
| `MlpBlock` | `MlpBlock`：Linear → GELU → Dropout → Linear → Dropout |
| `MultiHeadDotProductAttention`（`broadcast_dropout=False`） | `Attention`：手写多头点积 + 注意力 dropout + 输出投影 dropout |
| `encoder_norm` | 末尾 `nn.LayerNorm` |

**分割头（本项目新增，官方无）**：丢掉 cls token → token 网格 `(B, dim, g, g)` → `Conv1×1` 降维到 `decoder_dim` → `Conv3×3-BN-GELU` 细化 → 双线性上采样到 128×128 → `Conv1×1` 输出 1 通道。

- 官方 ViT-Base 是 img 224 / patch 16（14×14 token）；本项目输入 128×128 且需要密集预测，默认 **patch=8**（16×16 = 256 token）。其余超参对齐官方 **ViT-S_16** 配置：`dim=384 / depth=12 / heads=6 / dropout=0.0 / attn_dropout=0.0`（官方 ViT-S_16 的 `dropout_rate` 与 `attention_dropout_rate` 均为 0.0，0.1 只出现在官方 testing 调试配置里），可在 YAML 覆写
- 参数量约 **~22.3M**

## Swin Transformer（`swin.py`）

论文 Swin Transformer: Hierarchical Vision Transformer using Shifted Windows（Liu et al., ICCV 2021, arXiv:2103.14030），移植自官方 PyTorch 实现 [microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer)。

**逐一对齐官方的模块**：`PatchEmbed`、`PatchMerging`、`WindowAttention`（相对位置偏置表 + `relative_position_index` + shift mask）、`SwinTransformerBlock`（W-MSA / SW-MSA 交替、`torch.roll` 循环位移）、`BasicLayer`、以及 `_init_weights`（`nn.Linear` 用 `trunc_normal_(std=.02)`，`nn.LayerNorm` 置 1/0）。官方依赖 timm 的 `DropPath` / `to_2tuple` / `trunc_normal_`，已在本文件内自行实现，**不引入 timm 依赖**。

| Stage | patch (k,s) | dim | heads | depth | 分辨率 |
| ----- | ----------- | --- | ----- | ----- | ------ |
| 1 | (4,4) | 96 | 3 | 2 | 32×32 |
| 2 | PatchMerging | 192 | 6 | 2 | 16×16 |
| 3 | PatchMerging | 384 | 12 | 6 | 8×8 |
| 4 | PatchMerging | 768 | 24 | 2 | 4×4 |

**与官方的三处差异**：

1. **窗口补齐**：官方 `BasicLayer.forward` 不做 padding，要求 H/W 能被 7 整除；本项目 stage 分辨率为 32/16/8/4，前三个都不满足，故每个 `BasicLayer` 内部先补到窗口整数倍、跑完再裁掉（注意力 mask 也按补齐后的分辨率计算）。最后一个 stage 因 `min(H,W) <= window` 自动关闭 shift，与官方对 224 输入第 4 stage 的行为一致
2. **解码器**：官方分割版是 UperNet；此处换成轻量解码器——各 stage 输出先过 `LayerNorm`（官方只在最后一个 stage 后加），再 `Conv1×1` 统一到 `decoder_dim`、按 2^i 上采样到 32×32、concat 后 `Conv1×1` 输出并双线性上采样回 128×128
3. `BasicLayer` 拆成 `forward_features()`（只跑 Transformer 块）与 `forward()`（再下采样），以便解码器取到下采样前的多尺度特征

参数量约 **~28.2M**（Swin-T 量级）。

## 九模型对比

| 模型 | 参数量 | 编码器 | 解码器 | 跳跃连接 | 特点 |
|---|---|---|---|---|---|
| FCN | ~15M | VGG16 5-block | 双线性上采样 ×3 | add (FCN-8s) | 最简基线，无 concat |
| U-Net | ~31M | 5-stage conv | nearest 上采样 + conv | concat | 经典架构，参数较多 |
| AttU-Net | ~34.9M | 同 U-Net | 同 U-Net | concat + 注意力门 | 在跳跃上加门控，抑制无关区域 |
| SegFormer | ~3-4M | MiT 4-stage Transformer | MLP 1×1 conv | 无编码器-解码器跳跃 | 最轻量，全局注意力 |
| TransUNet | ~10-13M | ResNet 3-stage + ViT 瓶颈 | U-Net concat 解码 | concat（绕过 ViT） | CNN 局部 + Transformer 全局 |
| ConvLSTM | ~42.4M | 4-stage ConvLSTM | nearest 上采样 + conv | concat | 门控卷积编码，可扩展为多期序列输入 |
| cGAN（pix2pix 生成器） | ~41.8M | Conv4×4 s2 + LeakyReLU | ConvTranspose4×4 + ReLU | concat（逐级） | 官方 pix2pix 生成器，判别器未启用 |
| ViT | ~22.3M | PatchEmbed + 12×Transformer | 3×3 细化 + 双线性上采样 | 无 | 全局自注意力，patch=8，超参对齐官方 ViT-S_16 |
| Swin | ~28.2M | 4-stage 窗口注意力 | 多尺度 1×1 融合 + 上采样 | 无（stage 间 PatchMerging） | 窗口注意力，复杂度对分辨率线性 |

## 训练超参数

### 通用设置

| 参数 | 值 |
|---|---|
| 输入尺寸 | 10 × 128 × 128（4 驱动因子 + 6 LU one-hot） |
| 输出 | 1 × 128 × 128（raw logits） |
| 批大小 | 8（训练）/ 1（验证） |
| 训练轮数 | 100 |
| 保存频率 | 每 20 轮存 checkpoint |
| 可视化频率 | 每 200 步 |
| TensorBoard | 启用 |
| 训练数据包 | `[[10, 15]]`（2010→2015 年份对） |

### 损失函数

**0.5 × Focal Loss + 0.5 × Dice Loss**（联合损失，非 BCE）：

- **Focal Loss**：`BCEWithLogits` + `(1-p_t)^γ` 加权，γ=2.0，α=0.75（正类权重），沿 (B,H,W) 均值
- **Dice Loss**：sigmoid 激活后计算 Dice 系数，smooth=1.0，返回 `1 - mean(dice)`

### 优化器

| 设置 | 默认（attunet/unet/fcn/transunet/convlstm/cgan） | segformer / vit / swin |
|---|---|---|
| 优化器 | Adam | AdamW |
| 学习率 | 1e-4 | 5e-5 |
| weight_decay | 1e-5 | 1e-4 |
| beta_1 | 0.9 | 0.9 |
| 梯度裁剪 | 1.0 | 1.0 |

### 学习率调度

无 PyTorch `lr_scheduler`，使用自定义 on-plateau 策略：

- 跟踪 `epochs_since_improvement`（验证 loss 不下降的轮数）
- 每 8 轮无改进，LR 乘以 0.8

### 验证指标

训练时仅记录 `val_loss`；独立验证管道（`run_valid_pipeline`）通过 `MetricsMeter` 计算：

| 指标 | 公式 |
|---|---|
| OA（准确率） | (TP+TN) / total |
| UA（精确率） | TP / (TP+FP) |
| PA（召回率） | TP / (TP+FN) |
| F1 | 2 × PA × UA / (PA+UA) |
| IoU | TP / (TP+FP+FN) |

阈值：sigmoid 输出 > 0.5 视为正类。

### Checkpoint 输出

路径模板：`checkpoints/<model_name>/<run_timestamp>/`

| 文件 | 内容 | 保存时机 |
|---|---|---|
| `best_model.pth` | `model.state_dict()` | val_loss 创新低时覆盖 |
| `latest_checkpoint.pth` | `{epoch, model_state_dict, optimizer_state_dict, loss, config, timestamp}` | 每轮覆盖 |
| `epoch_<n>.pth` | state_dict | 每 20 轮 |

## 与其他模块的协作

| 消费模块 | 用途 | 调用方式 |
|---|---|---|
| `pipelines/train.py` | Stage-1 训练 | `create_model(model_name, in_ch=10, out_ch=1, model_config=...)` |
| `utils/runner_utils.py` | Stage-2 生成 pred_diff | `load_ANN_model()` → `get_pred_diff()` |
| `configs/trainConfig.yaml` | 模型与训练配置 | `model.<type>` 节传入 `create_model` 的 `model_config` |

## 已知问题

- `unet.py` 中定义了 `R2U_Net`、`R2AttU_Net`、`NestedUNet`、`Unet_dict` 但未在 `__init__.py` 中暴露，属于未启用的实验代码
- SegFormer 代码默认 `ff_expansion=(8,8,4,4)` 和 `decoder_dim=256`，但 YAML 覆写为 `[4,4,4,4]` 和 `128`——如果直接 `Segformer(in_ch=10, num_classes=1)` 不传 config，得到的是更大的模型
- TransUNet 的解码器通道计算硬编码假设 `out_channels=32`，改为其他值可能导致维度不匹配
- `cgan.py` 只含 pix2pix 生成器，未实现判别器与对抗损失；`utils/training_utils.py` 的 `ORIGINAL_LOSS_MAP` 中 `convlstm` / `cgan` 仍回退到 `bce_dice`（cgan 的原始对抗损失需要双网络训练循环；convlstm 原论文是回归任务，没有对应的分割损失）。`vit` 映射为 `bce`（官方未加权 softmax CE 的 C=2 等价形式）；`swin` 映射为 `bce_ls`（官方默认走 mixup 软标签 CE，此处对齐其关闭 mixup 时的 `LabelSmoothingCrossEntropy(0.1)` 分支，mixup/cutmix 未引入）
- `vit.py` / `swin.py` 的默认骨干超参已对齐官方（ViT 对齐 `ViT-S_16`：dim=384/depth=12/heads=6/dropout=0.0；Swin 对齐 `swin_tiny_patch4_window7` 的全部默认值），但输入通道数与尺寸（10 通道 128×128 vs 官方 3 通道 224×224）以及分割头/解码器为本项目新增，与官方分类模型不可直接对齐参数量；官方均带 ImageNet 预训练权重，本项目的这些模型是从零训练
- `convlstm.py` 的 `ConvLSTM_Seg` 编解码网络为本项目新增（上游只有序列层）；本项目样本为单期，故默认 T=1，多期序列输入需要 dataloader 侧提供 `(B,T,C,H,W)`
