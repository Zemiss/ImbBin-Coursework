# 机器学习课程大作业：空气压力系统故障检测

> APS 卡车气压系统不平衡二分类任务

[![Python](https://img.shields.io/badge/Python-3.12-green?style=flat-square)](https://www.python.org/)
![Task](https://img.shields.io/badge/Task-APS%20Binary%20Classification-orange)


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

## 训练

python -m main.train

```
python -m main.train --train_data data\ml_train.csv --model_path main\models\model.joblib
```

## 测试

python -m main.test

```
python -m main.test --test_data data\ml_train.csv --model_path main\models\model.joblib
```

`test.py` 支持通过 `--test_data` 接收评测数据相对路径，并通过 `--model_path` 接收权重文件路径。评测数据包含 `class` 标签列时，会打印 APS 维修代价、Precision、Recall、F1、Micro-F1、Macro-F1 和混淆矩阵。

