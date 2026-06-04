# APS Failure Imbalanced Binary Classification

本仓库用于完成机器学习期末大作业任务一：空气压力系统故障检测。

## 目录结构

```text
.
|-- main/
|   |-- train.py              # 训练入口
|   |-- test.py               # 测试/评测入口
|   |-- report.md             # 中文实验报告
|   |-- requirements.txt      # 依赖列表
|   |-- models/
|   |   `-- model.joblib      # 模型权重
|   `-- src/
|       |-- modeling.py       # 公共建模和指标代码
|       `-- __init__.py
|-- data/                     # 本地数据目录，不提交
|-- README.md
`-- .gitignore
```

## 环境

推荐使用 Python 3.12。本机使用 `conda` 环境 `py312`。

如果还没有安装依赖，在项目根目录运行：

```powershell
conda run -n py312 python -m pip install -r main/requirements.txt
```

## 训练命令

在项目根目录运行：

```powershell
conda run -n py312 python main/train.py --train_data data/ml_train.csv --model_path main/models/model.joblib
```

如果已经进入 `main` 目录：

```powershell
conda run -n py312 python train.py --train_data ..\data\ml_train.csv --model_path models\model.joblib
```

## 测试命令

在项目根目录运行：

```powershell
conda run -n py312 python main/test.py --test_data data/ml_train.csv --model_path main/models/model.joblib
```

如果已经进入 `main` 目录：

```powershell
conda run -n py312 python test.py --test_data ..\data\ml_train.csv --model_path models\model.joblib
```

`test.py` 支持通过 `--test_data` 接收评测数据相对路径，并通过 `--model_path` 接收权重文件路径。评测数据包含 `class` 标签列时，会打印 APS 维修代价、Precision、Recall、F1、Micro-F1、Macro-F1 和混淆矩阵。

## 提交注意

提交作业时不要包含 `data/` 目录、训练数据、测试数据或由数据复制得到的中间文件。
