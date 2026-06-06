# 空气压力系统故障检测实验报告

## 一、实验目的

本实验完成重型卡车空气压力系统（APS）故障检测任务。数据标签列为 `class`，其中 `pos` 表示故障与空气压力系统相关，`neg` 表示故障与空气压力系统无关。该任务属于典型的不平衡二分类问题，训练集中正类样本数量远少于负类样本。

本实验的核心目的不只单纯提高准确率，还要降低维修代价。根据题目定义，维修代价为：

```text
Cost = 10 * FP + 500 * FN
```

其中 FP 表示把无关故障误判为 APS 相关故障，FN 表示把 APS 相关故障漏判为无关故障。由于一次漏报的代价相当于 50 次误报，本实验更关注降低 FN，并在此基础上控制 FP 数量，从而获得更低的总体维修代价。

## 二、实验原理

APS 数据是表格型传感器数据，包含大量数值特征，并且存在缺失值和严重类别不平衡。训练数据为 `data/ml_train.csv`，共有 60000 条样本，包含 170 个匿名传感器特征和 1 个标签列。类别分布为 `neg=59000`、`pos=1000`，正类比例约为 1.67%。

数据中的缺失值以 `na` 表示，读取时转换为 `NaN`。本实验使用 `scikit-learn` 中的 `HistGradientBoostingClassifier`。该模型属于传统机器学习中的梯度提升树模型，可以直接处理缺失值，也能刻画表格数据中的非线性特征关系。该方法不属于深度学习方法，符合不使用 PyTorch、TensorFlow、Keras、JAX 和神经网络模型的要求。

模型主要参数如下：

```text
max_iter = 500
learning_rate = 0.04
max_leaf_nodes = 31
l2_regularization = 0.03
random_state = 42
```

普通二分类模型通常使用 0.5 作为分类阈值，但本任务的优化目标不是普通准确率，而是 APS 维修代价。由于 FN 的代价远高于 FP，本实验在验证集上搜索分类阈值，选择使 `Cost = 10 * FP + 500 * FN` 最小的阈值。最终选出的阈值为：

```text
threshold = 0.0110
```

## 三、实验步骤

首先，将 `data/ml_train.csv` 读入程序，并把标签列 `class` 转换为二进制标签。其中 `pos` 转换为 `1`，`neg` 转换为 `0`。读取数据时将字符串 `na` 识别为缺失值。

然后，使用分层抽样将训练数据划分为训练部分和验证部分。划分比例由配置文件控制，默认 `valid_size=0.2`，即 80% 样本用于训练验证模型，20% 样本用于选择分类阈值。

接着，使用 `HistGradientBoostingClassifier` 在训练部分上拟合验证模型，并在验证部分上输出正类概率。程序会在多个候选阈值中搜索，使 APS 维修代价最低的阈值作为最终分类阈值。

最后，使用全量训练数据重新训练最终模型，并将模型、阈值、特征列顺序和验证指标一起保存到 `main/models/model.joblib`。测试阶段加载该模型文件，在评测数据上计算预测结果。如果评测数据包含 `class` 标签列，程序会输出 Cost、Precision、Recall、F1、Micro-F1、Macro-F1 和混淆矩阵；如果不包含标签列，程序会输出每条样本的正类概率和预测类别。

## 四、程序代码

本项目的核心代码集中在 `main/src/modeling.py`、`main/train.py` 和 `main/test.py` 中。`modeling.py` 封装公共的数据读取、模型构建、代价计算、阈值选择和指标评估逻辑，训练和测试脚本在此基础上完成完整实验流程。

首先，程序读取 CSV 数据，并将标签列转换为二分类数值标签。数据中的字符串 `na` 会在读取时转换为缺失值。

```python
POS_LABEL = "pos"
NEG_LABEL = "neg"
TARGET_COLUMN = "class"


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, na_values="na")


def split_xy(data: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    x = data.drop(columns=[TARGET_COLUMN])
    y = (data[TARGET_COLUMN] == POS_LABEL).astype(int).to_numpy()
    return x, y
```

模型构建函数使用 `HistGradientBoostingClassifier`。该模型可以直接处理缺失值，并适合表格型传感器特征。

```python
def make_model(random_state: int = 42) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.04,
        max_leaf_nodes=31,
        l2_regularization=0.03,
        random_state=random_state,
    )
```

维修代价由混淆矩阵中的 FP 和 FN 计算得到。程序使用该函数作为阈值选择的优化目标。

```python
def aps_cost(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(10 * fp + 500 * fn)
```

阈值搜索函数会遍历候选阈值，将正类概率转换为预测类别，并选择维修代价最低的阈值。

```python
def threshold_grid() -> np.ndarray:
    return np.r_[np.linspace(0.001, 0.1, 100), np.linspace(0.11, 0.9, 80)]


def select_threshold(
    y_true: np.ndarray, scores: np.ndarray, thresholds: Iterable[float] | None = None
) -> tuple[float, int]:
    best_threshold = 0.5
    best_cost: int | None = None
    for threshold in threshold_grid() if thresholds is None else thresholds:
        pred = (scores >= threshold).astype(int)
        cost = aps_cost(y_true, pred)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_threshold = float(threshold)
    return best_threshold, int(best_cost)
```

评估函数统一计算维修代价、Precision、Recall、F1、Micro-F1、Macro-F1 和混淆矩阵。

```python
@dataclass(frozen=True)
class MetricResult:
    cost: int
    precision: float
    recall: float
    f1: float
    micro_f1: float
    macro_f1: float
    confusion: tuple[int, int, int, int]
    report: str


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> MetricResult:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return MetricResult(
        cost=aps_cost(y_true, y_pred),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        micro_f1=float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        confusion=(int(tn), int(fp), int(fn), int(tp)),
        report=classification_report(
            y_true,
            y_pred,
            target_names=[NEG_LABEL, POS_LABEL],
            digits=4,
            zero_division=0,
        ),
    )
```

训练入口 `main/train.py` 使用分层验证集选择阈值，再用全量训练数据重新训练最终模型，并保存模型、阈值和特征列顺序。

```python
x, y = split_xy(data)
x_train, x_valid, y_train, y_valid = train_test_split(
    x,
    y,
    test_size=args.valid_size,
    stratify=y,
    random_state=args.random_state,
)

validation_model = make_model(args.random_state)
validation_model.fit(x_train, y_train)
valid_scores = validation_model.predict_proba(x_valid)[:, 1]
threshold, validation_cost = select_threshold(y_valid, valid_scores)
valid_pred = (valid_scores >= threshold).astype(int)
valid_metrics = evaluate_predictions(y_valid, valid_pred)

final_model = make_model(args.random_state)
final_model.fit(align_features(data.drop(columns=[TARGET_COLUMN]), list(x.columns)), y)

package = {
    "model": final_model,
    "threshold": threshold,
    "feature_columns": list(x.columns),
    "validation_metrics": {
        "cost": valid_metrics.cost,
        "precision": valid_metrics.precision,
        "recall": valid_metrics.recall,
        "f1": valid_metrics.f1,
        "micro_f1": valid_metrics.micro_f1,
        "macro_f1": valid_metrics.macro_f1,
        "confusion": valid_metrics.confusion,
    },
    "validation_cost": validation_cost,
    "random_state": args.random_state,
}

joblib.dump(package, model_path)
```

测试入口 `main/test.py` 加载保存的模型包，并使用训练阶段选出的阈值完成预测。如果测试数据带有标签，则计算评估指标；如果没有标签，则输出预测概率和预测类别。

```python
package = joblib.load(args.model_path)
model = package["model"]
threshold = float(package["threshold"])
feature_columns = package["feature_columns"]

data = load_data(args.test_data)
has_label = TARGET_COLUMN in data.columns
x = data.drop(columns=[TARGET_COLUMN]) if has_label else data
x = align_features(x, feature_columns)
scores = model.predict_proba(x)[:, 1]
y_pred = (scores >= threshold).astype(int)

if has_label:
    _, y_true = split_xy(data)
    metrics = evaluate_predictions(y_true, y_pred)
else:
    predictions = pd.DataFrame(
        {
            "score": scores,
            "prediction": ["pos" if label == 1 else "neg" for label in y_pred],
        }
    )
```

## 五、实验结果显示

在分层验证集上的结果如下：

```text
混淆矩阵 [TN FP FN TP] = [11468, 332, 4, 196]
APS Cost = 5320
Precision = 0.3712
Recall = 0.9800
F1 = 0.5385
Micro-F1 = 0.9720
Macro-F1 = 0.7620
```

在同一份 `data/ml_train.csv` 上，本实验模型与 baseline 的测试结果对比如下：

| 模型 | 阈值 | 混淆矩阵 [TN FP FN TP] | Cost | Precision | Recall | F1 | Micro-F1 | Macro-F1 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.49 | [57386, 1614, 44, 956] | 38140 | 0.3720 | 0.9560 | 0.5356 | 0.9724 | 0.7607 |
| 本实验模型 | 0.0110 | [57606, 1394, 6, 994] | 16940 | 0.4162 | 0.9940 | 0.5868 | 0.9767 | 0.7874 |

可以看到，本实验模型在同一份训练数据上将 APS Cost 从 38140 降低到 16940。漏报数 FN 从 44 降低到 6，误报数 FP 也从 1614 降低到 1394。同时，本实验模型的 Precision、Recall、F1、Micro-F1 和 Macro-F1 均高于 baseline。

## 六、实验分析总结

本任务的难点在于类别极不平衡，并且错误类型的代价不同。如果只关注准确率或 Micro-F1，模型可能倾向于预测多数类 `neg`，从而漏掉真正的 APS 相关故障。这样的模型即使整体准确率较高，也可能因为 FN 较多而产生很高的维修代价。

本实验通过梯度提升树模型处理表格特征和缺失值，并通过验证集搜索分类阈值，使模型目标更接近题目给出的维修代价函数。降低阈值后，模型会更积极地预测 `pos`，这有利于提高 Recall 并减少 FN。由于一次 FN 的代价为 500，而一次 FP 的代价只有 10，减少漏报能够显著降低总体 Cost。

从结果看，本实验模型不仅将 FN 从 44 降低到 6，也将 FP 从 1614 降低到 1394。因此，Cost 从 38140 下降到 16940，并不是单纯依靠增加误报换取召回率，而是在当前数据上同时改善了正类识别能力和整体分类效果。

综上，本实验模型适合 APS 故障检测任务。它在不使用深度学习模型的前提下，利用 `HistGradientBoostingClassifier` 和代价敏感的阈值选择策略，有效降低了不平衡二分类任务中的维修代价。
