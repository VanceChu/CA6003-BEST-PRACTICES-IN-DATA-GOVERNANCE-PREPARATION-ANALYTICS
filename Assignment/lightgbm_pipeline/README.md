# LightGBM Pipeline for Home Credit Default Risk

[English](README.md) | [中文](README_CN.md)

This project implements a complete machine learning pipeline for the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) Kaggle competition.

## Project Structure

```
lightgbm_pipeline/
├── README.md                 # This file
├── notebooks/
│   ├── lightgbm_pipeline_notebook.ipynb          # Main notebook (supports train/inference modes)
│   └── lightgbm_pipeline_notebook_executed.ipynb # Executed notebook with outputs
├── scripts/
│   └── lightgbm_with_simple_features.py          # Original Python script
└── models/                                       # Saved models (generated after training)
    ├── lgbm_fold_0.txt ~ lgbm_fold_9.txt        # 10 fold models (LightGBM native format)
    └── feature_list.pkl                          # Feature list for inference

**Note**: All output files (predictions, visualizations, logs) are automatically suffixed with a timestamp (e.g., `_20260117_232817`) to prevent overwriting.
```

## Usage

### Configuration

In the first cell of `lightgbm_pipeline_notebook.ipynb`:

```python
TRAIN_MODE = True   # True: train new models, False: load saved models
DEBUG = False       # True: use 10000 rows for quick testing
```

### Training Mode (`TRAIN_MODE = True`)

1. Loads all data and performs feature engineering
2. Trains 10-fold LightGBM models
3. Saves models to `models/` directory
4. Generates visualizations and submission file
5. **Time**: ~30-40 minutes (full data)

### Inference Mode (`TRAIN_MODE = False`)

1. Loads all data and performs feature engineering
2. Loads pre-trained models from `models/` directory
3. **Performs OOF Evaluation**: Recreates validation splits to calculate AUC and OOF scores
4. **Generates Visualizations**: Creates ROC curves and feature importance plots
5. **Generates Predictions**: Creates submission file for test set
6. **Time**: ~5-10 minutes

## Results

| Metric | Value |
|--------|-------|
| 10-Fold Mean AUC | 0.7917 |
| OOF AUC | 0.7917 |
| Best Fold (Fold 5) | 0.7966 |
| Worst Fold (Fold 6) | 0.7852 |

## Data Requirements

Place the competition data in `./home-credit-default-risk/`:
- application_train.csv
- application_test.csv
- bureau.csv
- bureau_balance.csv
- previous_application.csv
- POS_CASH_balance.csv
- installments_payments.csv
- credit_card_balance.csv

## Dependencies

- Python 3.8+
- lightgbm
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib
