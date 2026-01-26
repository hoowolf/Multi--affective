## Multi-affective

### 目录结构
- `docs/`：实验说明与计划
- `multi_affective/`：公共代码（随机种子、复现配置等）
- `check_data.py`：数据正确性检查（缺失/重复/标签分布）
- `preprocess.py`：数据预处理（标签清洗、文本/图像配置、分层划分）
- `train.py`：模型训练（Text-only / Image-only / Multimodal）
- `requirements.txt`：环境依赖

### 环境安装
```bash
pip install -r requirements.txt
```

### 复现流程
```bash
python check_data.py --data-dir .\datasets --output-dir .\outputs --save-json
python preprocess.py --data-dir .\datasets --output-dir .\outputs --seed 42 --val-ratio 0.1
python train.py --mode all --run-name baseline --deterministic --seed 42 --epochs 5 --batch-size 16
```

训练完成后，查看 `outputs/<run_name>/`：
- `text/`、`image/`、`multimodal/`：三种模式各自的日志与图
- `curves.png`：训练过程曲线
- `confusion_matrix.png`：验证集混淆矩阵（best checkpoint）
- `best.pt`：最优模型参数（按验证集 Macro-F1 选择）
