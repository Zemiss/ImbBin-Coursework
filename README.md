# APS Failure Imbalanced Binary Classification

本仓库用于完成机器学习期末大作业的任务一：空气压力系统故障检测。

## 目录结构

```text
.
├── train.py                  # 作业要求的训练入口
├── test.py                   # 作业要求的评测入口
├── src/aps_failure/          # 公共建模与指标代码
├── tests/                    # 单元测试
├── docs/                     # 实验报告 Markdown
├── models/                   # 模型权重文件
├── scripts/                  # 可选辅助脚本目录
├── requirements.txt          # Python 依赖
└── .gitignore
```

## 运行环境

推荐使用 Python 3.12。本机验证命令使用：

```bash
conda run -n py312 python -m unittest tests/test_aps_model.py
```

## 训练

```bash
python train.py --train_data data/ml_train.csv --model_path models/model.joblib
```

## 评测

```bash
python test.py --test_data <eval_data.csv> --model_path models/model.joblib
```

评测数据包含 `class` 标签列时，脚本会打印 APS 维修代价、Precision、Recall、F1、Micro-F1、Macro-F1 和混淆矩阵。

## 提交注意

不要提交或打包 `data/` 目录、训练数据、测试数据或由数据复制得到的中间文件。
