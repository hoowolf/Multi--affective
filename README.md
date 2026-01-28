## Multi-affective

本项目为多模态情感分类基线系统，输入为文本与图像两种模态，输出为三分类标签（positive / neutral / negative）。包含数据检查、预处理、训练与产出可复现的实验记录。

## Setup

本项目基于 Python 与 PyTorch。安装依赖：

```bash
pip install -r requirements.txt
```

## Repository structure

```text
|-- multi_affective/          # 公共代码（数据集、模型、训练工具等）
|   |-- __init__.py           # 包入口
|   |-- data.py               # TextDataset/ImageDataset/MultiDataset
|   |-- labels.py             # 标签映射（label2id/id2label）
|   |-- models.py             # 文本/图像/多模态模型
|   |-- training.py           # train/eval、绘图与保存
|   |-- config.py             # 读取预处理配置、图像 transform
|   |-- reproducibility.py    # 随机种子与复现实验相关工具
|   |-- io_utils.py           # JSON/CSV 读写等 I/O 工具
|   |-- text_preprocess.py    # 文本清洗工具
|   |-- image_preprocess.py   # 图像预处理配置与 transform 构建
|   |-- stats.py              # 统计工具（如分位数）
|-- check_data.py             # 数据正确性检查（缺失/重复/标签分布）
|-- preprocess.py             # 数据预处理（清洗、配置、分层划分）
|-- train.py                  # 训练入口（Text-only / Image-only / Multimodal）
|-- run_train.py              # Python 调用 train.py 的封装（便于批量实验）
|-- eval.py                   # 预留：评估入口（可扩展）
|-- predict.py                # 加载 checkpoint 对测试集生成预测结果
|-- requirements.txt          # 环境依赖
|-- README.md                 # 项目说明与复现指南
|-- .gitignore                # Git 忽略规则（不提交数据与大文件）
```

## Run pipeline

### 1. 数据检查

```bash
python check_data.py --data-dir ./datasets --output-dir ./outputs --save-json
```

### 2. 数据预处理与划分

```bash
python preprocess.py --data-dir ./datasets --output-dir ./outputs --seed 42 --val-ratio 0.1
```

该步骤会生成：
- `outputs/preprocess/`：文本/图像预处理配置
- `outputs/splits/`：训练/验证划分文件（如 `train_split.txt`、`val_split.txt`）

### 3. 模型训练

一键训练三种模式（消融 + 多模态）：

```bash
python train.py --mode all --run-name baseline --deterministic --seed 42 --epochs 5 --batch-size 16
```

仅训练单模态：

```bash
python train.py --mode text  --run-name text_only  --deterministic --seed 42
python train.py --mode image --run-name image_only --deterministic --seed 42
```

多模态融合架构对比（控制变量，只改 `--multimodal-arch`）：

```bash
python train.py --mode multimodal --run-name mm_gated  --multimodal-arch gated  --deterministic --seed 42
python train.py --mode multimodal --run-name mm_concat --multimodal-arch concat --deterministic --seed 42
python train.py --mode multimodal --run-name mm_late   --multimodal-arch late   --deterministic --seed 42
```

Flooding 正则化（论文中“洪水填充”策略）：

```bash
python train.py --mode multimodal --run-name mm_flood --multimodal-arch gated --flood-level 0.1 --deterministic --seed 42
```

## train.py 参数说明

下面给出常用参数含义（更完整列表可运行 `python train.py --help` 查看）：

- **运行模式与路径**
  - `--mode {text,image,multimodal,all}`：训练模式（`all` 会依次训练三种模式）
  - `--data-dir`：数据根目录（默认 `./datasets`，样本文件在 `data/` 下）
  - `--train-index / --val-index`：训练/验证划分文件路径
  - `--preprocess-dir`：预处理配置目录（读取 `text_config.json`/`image_config.json`）
  - `--output-dir`：输出根目录（默认 `./outputs`）
  - `--run-name`：本次实验名字（输出为 `outputs/<run_name>/`；不填则用时间戳）

- **复现与设备**
  - `--seed`：随机种子
  - `--deterministic`：开启更强的确定性设置（便于复现）
  - `--device {auto,cpu,cuda}`：训练设备选择
  - `--num-workers`：DataLoader 线程数

- **模型与融合**
  - `--text-model`：文本编码器权重路径/名称（默认使用预处理配置或 `./models/bert-base-uncased`）
  - `--image-encoder`：图像编码器（目前支持 `resnet18`）
  - `--dropout`：分类头 dropout
  - `--fusion-dim`：多模态特征投影维度（`multimodal` 模式下使用）
  - `--multimodal-arch {gated,concat,late}`：多模态融合架构
    - `gated`：门控融合（特征级自适应加权）
    - `concat`：早期融合（特征拼接 + MLP）
    - `late`：后期融合（各自分类后融合 logits）

- **数据增强**
  - `--text-aug {baseline,weak,strong}`：文本增强强度（仅训练集启用）
  - `--image-aug {baseline,weak,strong}`：图像增强强度（仅训练集启用）
  - `--no-pretrained-image`：不加载 ImageNet 预训练的 ResNet 权重

- **优化与学习率**
  - `--epochs`：训练轮数
  - `--batch-size`：batch 大小
  - `--lr-encoder / --lr-head`：编码器与分类头的学习率（分组训练）
  - `--weight-decay`：AdamW 权重衰减
  - `--lr-scheduler {none,cosine,cosine_warmup,step,plateau}`：学习率调度策略
  - `--warmup-epochs`：warmup 轮数（`cosine_warmup` 有效）
  - `--eta-min`：cosine 最小学习率
  - `--step-size / --gamma`：StepLR 的步长与衰减系数；`gamma` 也用于 plateau 降学习率
  - `--plateau-patience`：ReduceLROnPlateau 的耐心值

- **损失与正则化**
  - `--class-weights / --no-class-weights`：是否按训练集类别分布计算类别权重（用于交叉熵）
  - `--freeze-encoders`：冻结编码器，只训练分类头
  - `--early-stop-patience`：早停耐心值（按验证集 `val_acc` 判断）
  - `--flood-level`：Flooding 水位（`0` 关闭；如 `0.1`/`0.01`）

## Outputs

训练完成后，查看 `outputs/<run_name>/`：
- `text/`：Text-only 训练日志与可视化
- `image/`：Image-only 训练日志与可视化
- `multimodal/<arch>/`：多模态训练日志与可视化（`<arch>` 为 `gated/concat/late`）
- `summary.json` / `summary.csv`：本次运行的汇总结果

每个子目录通常包含：
- `run_config.json`：本次运行的完整配置（包含 `multimodal_arch`、`flood_level` 等）
- `history.json` / `history.csv`：逐 epoch 训练/验证指标
- `curves.png`：loss/acc 曲线
- `confusion_matrix.png`：验证集混淆矩阵（按 best checkpoint 保存）
- `best.pt`：最佳模型参数（按验证集准确率 `val_acc` 选择）

## predict.py 使用

`predict.py` 用于加载 `train.py` 生成的 checkpoint，在测试集上输出预测标签文件（两列：`guid,tag`）。

```bash
python predict.py \
  --checkpoint ./outputs/<run_name>/<subdir>/best.pt \
  --data-dir ./datasets \
  --test-file ./datasets/test_without_label.txt \
  --output-file ./outputs/test_predictions.txt \
  --device auto \
  --batch-size 64
```

其中 `<subdir>` 的取值示例：
- Text-only：`text`
- Image-only：`image`
- Multimodal：`multimodal/gated` 或 `multimodal/concat` 或 `multimodal/late`

## Attribution

本项目主要依赖以下开源工具与实现：
- Hugging Face Transformers（文本编码器）
- TorchVision（图像编码器与增强）

参考论文：
- Do We Need Zero Training Loss After Achieving Zero Training Error?[https://arxiv.org/pdf/2002.08709]
