# 机器学习课程大作业：ImbBin

[![Python](https://img.shields.io/badge/Python-3.12-green?style=flat-square)](https://www.python.org/)
![Task](https://img.shields.io/badge/Task-APS%20Binary%20Classification-orange)

> APS 卡车气压系统不平衡二分类任务。项目目标不只单纯提高准确率，还要降低维修代价。

## 目录

- [机器学习课程大作业：ImbBin](#机器学习课程大作业imbbin)
  - [目录](#目录)
  - [项目结构](#项目结构)
  - [环境配置](#环境配置)
  - [配置系统](#配置系统)
  - [模型结构](#模型结构)
  - [训练配置](#训练配置)
  - [输出文件](#输出文件)
  - [训练](#训练)
  - [测试](#测试)
  - [实验结果](#实验结果)

## 项目结构

```text
.
|-- main/
|   |-- train.py              # 训练入口
|   |-- test.py               # 测试/评测入口
|   |-- report.md             # 中文实验报告
|   |-- requirements.txt      # 依赖列表
|   |-- configs/
|   |   `-- default.yaml      # 默认训练和测试配置
|   |-- models/
|   |   `-- model.joblib      # 训练得到的模型权重
|   `-- src/
|       |-- config.py         # YAML 配置读取和命令行参数合并
|       |-- modeling.py       # 公共建模、阈值选择和指标计算代码
|       `-- __init__.py
|-- data/                     # 本地数据目录，不提交
|-- README.md
`-- .gitignore
```

## 环境配置

本项目使用 Python 3.12，主要依赖包括 `pandas`、`numpy`、`scikit-learn`、`joblib` 和 `PyYAML`。

```powershell
conda create -n ImbBin python=3.12 -y
pip install -r requirements.txt
```

数据文件需要放在本地 `data/` 目录下，例如：

```text
data/ml_train.csv
```

## 配置系统

默认配置文件位于 `main/configs/default.yaml`：

```yaml
train:
  train_data: "../data/ml_train.csv"
  model_path: "models/model.joblib"
  valid_size: 0.2
  random_state: 42

test:
  test_data: "../data/ml_train.csv"
  model_path: "models/model.joblib"
  prediction_path: null
```

`train.py` 和 `test.py` 会先读取 YAML 配置，再用命令行参数覆盖配置值。配置文件和命令行中的相对路径都会按 `main/` 目录解析，因此默认配置中的 `../data/ml_train.csv` 指向仓库根目录下的 `data/ml_train.csv`。

如果只想临时覆盖某个参数，也可以直接传命令行参数：

```bash
python -m main.train --train_data ../data/ml_train.csv --model_path models/model.joblib
python -m main.test --test_data ../data/ml_train.csv --model_path models/model.joblib
```

## 模型结构

模型使用 `scikit-learn` 的 `HistGradientBoostingClassifier`，属于梯度提升树模型，适合处理表格型传感器数据，并且可以直接处理数据中的缺失值。

完整预测流程为：

1. 读取 CSV，并将 `na` 识别为缺失值；
2. 将标签列 `class` 中的 `pos` 转为 1、`neg` 转为 0；
3. 使用梯度提升树输出正类概率；
4. 使用验证集上选出的阈值将概率转为 `pos` / `neg`；
5. 根据 `10 * FP + 500 * FN` 计算 APS 维修代价。

## 训练配置

训练阶段使用的流程参数如下：

| 参数 | 取值 | 说明 |
| --- | ---: | --- |
| `valid_size` | 0.2 | 将 20% 训练数据划为验证集，用于选择分类阈值 |
| `random_state` | 42 | 固定数据划分和模型随机性，保证结果可复现 |
| `stratify` | `y` | 按标签做分层划分，保持正负样本比例 |
| 阈值搜索范围 | `0.001-0.1` 和 `0.11-0.9` | 在验证集上搜索 APS Cost 最低的分类阈值 |
| 优化目标 | `10 * FP + 500 * FN` | 以题目定义的维修代价作为阈值选择标准 |

模型训练使用的 `HistGradientBoostingClassifier` 参数如下：

| 参数 | 取值 | 说明 |
| --- | ---: | --- |
| `max_iter` | 500 | 最大 boosting 迭代轮数 |
| `learning_rate` | 0.04 | 每轮弱学习器的学习率 |
| `max_leaf_nodes` | 31 | 每棵树的最大叶子节点数 |
| `l2_regularization` | 0.03 | L2 正则化强度，用于抑制过拟合 |
| `random_state` | 42 | 模型随机种子 |

训练时先在 80% 训练子集上拟合验证模型，并在 20% 验证集上选择最优阈值；确定阈值后，再使用全部训练数据重新训练最终模型。

## 输出文件

训练完成后会生成或更新以下文件：

| 路径 | 说明 |
| --- | --- |
| `main/models/model.joblib` | 保存最终模型、分类阈值、特征列顺序、验证集指标和随机种子 |
| `main/report.md` | 中文实验报告，包含实验目的、原理、步骤、代码说明、结果和分析 |

当测试数据不包含 `class` 标签列，并且指定了 `--prediction_path` 时，`test.py` 会额外输出预测结果 CSV。该文件包含两列：

| 列名 | 说明 |
| --- | --- |
| `score` | 模型输出的正类概率 |
| `prediction` | 根据阈值得到的预测类别，取值为 `pos` 或 `neg` |

## 训练

**在仓库根目录运行：**

```bash
python -m main.train
```

训练流程会完成以下步骤：

- 读取 `data/ml_train.csv`，将标签列 `class` 中的 `pos` 转为 1、`neg` 转为 0；
- 按配置中的 `valid_size=0.2` 做分层验证集划分；
- 使用 `HistGradientBoostingClassifier` 训练验证模型；
- 在验证集上搜索使 APS 维修代价最低的分类阈值；
- 使用全量训练数据重新训练最终模型；
- 将模型、阈值、特征列顺序和验证指标保存到 `main/models/model.joblib`。

训练完成后，终端会打印保存路径、选出的阈值、验证集混淆矩阵以及 Cost、Precision、Recall、F1、Micro-F1、Macro-F1。

## 测试

**在仓库根目录运行默认评测：**

```bash
python -m main.test
```

如果评测数据包含 `class` 标签列，脚本会加载 `main/models/model.joblib` 并输出：

- APS 维修代价；
- Precision、Recall、F1；
- Micro-F1、Macro-F1；
- 混淆矩阵 `[TN FP FN TP]`；
- `scikit-learn` 分类报告。

如果评测数据不包含标签列，脚本会输出每条样本的正类概率和预测类别。可以通过 `--prediction_path` 保存预测结果：

```bash
python -m main.test --test_data ../data/eval.csv --prediction_path predictions.csv
```

## 实验结果

本项目以 APS 维修代价作为主要优化目标：

```text
Cost = 10 * FP + 500 * FN
```

在分层验证集上，模型选出的最佳分类阈值为 `0.0110`，验证结果如下：

| 指标 | 数值 |
| --- | ---: |
| 混淆矩阵 `[TN FP FN TP]` | `[11468, 332, 4, 196]` |
| APS Cost | 5320 |
| Precision | 0.3712 |
| Recall | 0.9800 |
| F1 | 0.5385 |
| Micro-F1 | 0.9720 |
| Macro-F1 | 0.7620 |

在同一份 `data/ml_train.csv` 上评测时，本项目模型与 baseline 的对比如下：

| 模型 | 阈值 | 混淆矩阵 `[TN FP FN TP]` | Cost | Precision | Recall | F1 | Micro-F1 | Macro-F1 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.49 | `[57386, 1614, 44, 956]` | 38140 | 0.3720 | 0.9560 | 0.5356 | 0.9724 | 0.7607 |
| 本项目模型 | 0.0110 | `[57606, 1394, 6, 994]` | 16940 | 0.4162 | 0.9940 | 0.5868 | 0.9767 | 0.7874 |


