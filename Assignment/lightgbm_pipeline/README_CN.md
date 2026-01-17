# Home Credit Default Risk - LightGBM 机器学习流水线

[English](README.md) | [中文](README_CN.md)

本项目实现了 [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) Kaggle 竞赛的完整机器学习流水线。

## 项目结构 (Project Structure)

```
lightgbm_pipeline/
├── README.md                 # 英文说明文档
├── README_CN.md              # 本文件 (中文说明文档)
├── notebooks/
│   ├── lightgbm_pipeline_notebook.ipynb          # 主 Notebook (支持训练/推理模式)
│   └── lightgbm_pipeline_notebook_executed.ipynb # 已运行的 Notebook (包含输出结果)
├── scripts/
│   └── lightgbm_with_simple_features.py          # 原始 Python 脚本
└── models/                                       # 保存的模型 (训练后生成)
    ├── lgbm_fold_0.txt ~ lgbm_fold_9.txt        # 10 折模型文件 (LightGBM 原生格式)
    └── feature_list.pkl                          # 推理用的特征列表

**注意**: 所有输出文件 (提交结果、图表、日志) 都会自动添加时间戳后缀 (例如 `_20260117_232817`) 防止覆盖。
```

## 使用方法 (Usage)

### 配置 (Configuration)

在 `lightgbm_pipeline_notebook.ipynb` 的第一个代码单元格中修改配置：

```python
TRAIN_MODE = True   # True: 训练新模型, False: 加载已保存模型进行推理
DEBUG = False       # True: 使用 10000 行数据快速测试
```

### 训练模式 (`TRAIN_MODE = True`)

1. 加载所有数据并执行特征工程
2. 训练 10-Fold LightGBM 模型
3. 将模型保存到 `models/` 目录
4. 生成可视化图表和 Kaggle 提交文件
5. **耗时**: 约 30-40 分钟 (全量数据)

### 推理模式 (`TRAIN_MODE = False`)

1. 加载所有数据并执行特征工程
2. 从 `models/` 目录加载预训练模型
3. **执行 OOF 评估**: 重现验证集划分，计算 AUC 和 OOF 得分
4. **生成可视化**: 绘制 ROC 曲线和特征重要性图表
5. **生成预测结果**: 生成测试集的提交文件
6. **耗时**: 约 5-10 分钟

## 结果 (Results)

| 指标 (Metric) | 值 (Value) |
|--------|-------|
| 10-Fold 平均 AUC | 0.7917 |
| OOF AUC | 0.7917 |
| 最佳折 (Fold 5) | 0.7966 |
| 最差折 (Fold 6) | 0.7852 |

## 数据要求 (Data Requirements)

请将竞赛数据放置在 `./home-credit-default-risk/` 目录下：
- application_train.csv
- application_test.csv
- bureau.csv
- bureau_balance.csv
- previous_application.csv
- POS_CASH_balance.csv
- installments_payments.csv
- credit_card_balance.csv

## 依赖 (Dependencies)

- Python 3.8+
- lightgbm
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib
