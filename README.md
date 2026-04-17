# Walmart Sales Demand Forecasting

An end-to-end machine learning project that predicts weekly retail sales for
Walmart stores and departments using the
[Kaggle Walmart Recruiting – Store Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting)
dataset.

---

## Project Structure

```
├── data/                          # Raw CSV files (not committed)
│   ├── train.csv
│   ├── test.csv
│   ├── stores.csv
│   └── features.csv
├── models/                        # Saved model artefacts (created at runtime)
├── notebooks/
│   └── EDA_and_Forecasting.ipynb  # Full walkthrough notebook
├── tests/
│   └── test_forecasting.py        # Unit & smoke tests (no Kaggle data needed)
├── walmart_forecasting/
│   ├── __init__.py
│   ├── data_loader.py             # Load & merge raw CSVs
│   ├── evaluation.py              # WMAE, MAE, RMSE, MAPE
│   ├── feature_engineering.py    # Lag, rolling, calendar & holiday features
│   └── models.py                  # LinearRegression, RandomForest, XGBoost
├── main.py                        # CLI entry point
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get the data

Download the four CSV files from Kaggle and place them in `data/`:

```
data/train.csv
data/test.csv
data/stores.csv
data/features.csv
```

### 3. Train a model

```bash
# Train XGBoost (default) and evaluate on the last 20 % of weeks
python main.py train --model xgboost --data-dir data/ --output-dir models/

# Other available models
python main.py train --model random_forest
python main.py train --model linear_regression
```

### 4. Generate predictions

```bash
python main.py predict --model-path models/xgboost.pkl --data-dir data/ --output submission.csv
```

### 5. Run the notebook

```bash
jupyter notebook notebooks/EDA_and_Forecasting.ipynb
```

---

## Models

| Model              | Description                                              |
|--------------------|----------------------------------------------------------|
| `linear_regression`| Baseline with StandardScaler                             |
| `random_forest`    | Ensemble of decision trees, good interpretability        |
| `xgboost`          | Gradient-boosted trees, typically best WMAE              |

---

## Features

| Category         | Features                                                               |
|------------------|------------------------------------------------------------------------|
| Store metadata   | Store ID, Dept ID, Store Type, Store Size                              |
| External         | Temperature, Fuel Price, CPI, Unemployment, MarkDown1–5                |
| Calendar         | Year, Month, Week-of-year, Quarter, Day-of-week, Week-of-month         |
| Holiday flags    | IsHoliday, Super Bowl, Labor Day, Thanksgiving, Christmas              |
| Lag (weekly)     | 1, 2, 4, 8, 12, 26, 52 weeks                                           |
| Rolling stats    | Mean & Std over 4, 8, 13, 26-week windows (shift-1, no leakage)        |

---

## Evaluation Metric

The primary metric is the competition's **Weighted Mean Absolute Error (WMAE)**:

```
WMAE = Σ(w_i × |y_i − ŷ_i|) / Σ(w_i)
```

where `w_i = 5` for holiday weeks and `w_i = 1` otherwise.

---

## Running Tests

```bash
pytest tests/ -v
```

All tests use synthetic data — no Kaggle files required.
