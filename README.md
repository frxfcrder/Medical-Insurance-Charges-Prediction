# Medical Insurance Charges Prediction

Exploratory data analysis and regression models that predict medical insurance
charges from customer attributes. The analysis lives in a notebook; the
modeling logic is extracted into an importable `src/` package with tests.

## Dataset

- **Source:** [Kaggle — insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance) (auto-downloaded with `kagglehub`)
- **Size:** 1,338 customer records × 7 columns (6 features + target), no missing values
- **Target:** `charges` — continuous, right-skewed
- **Features:** `age`, `sex`, `bmi`, `children`, `smoker`, `region`

## Project Structure

```
health-insurance-cost-prediction/
├── data/
├── models/
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── tests/
│   ├── conftest.py
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_pipeline.py
├── requirements.txt
├── README.md
└── .gitignore
```

`data/` holds the downloaded CSV and `models/` the saved artifacts; both are
git-ignored (`.gitkeep` keeps the folders in the repo).

## How to Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux activate with `source .venv/bin/activate`.

**Notebook (EDA + results):**

```bash
jupyter notebook notebooks/01_eda.ipynb
```

**As a library:**

```python
from src.data_loader import load_dataset, split_features_target
from src.preprocessing import clean_dataframe
from src.train import split_data, train_all
from src.evaluate import evaluate_models, format_metrics_table
from src.predict import save_model, load_model, predict_charges

df = clean_dataframe(load_dataset())
X, y = split_features_target(df)
X_train, X_test, y_train, y_test = split_data(X, y)

models = train_all(X_train, y_train)
print(format_metrics_table(evaluate_models(models, X_test, y_test, y_train=y_train)))

save_model(models["Random Forest (Log-Transform)"])
print(predict_charges(load_model(), X_test.head()))
```

**Tests:**

```bash
pytest
```

## Results

Held-out test set (80/20 split, `random_state=42`):

| Model                             |        MAE |       RMSE |        R² |
| :-------------------------------- | ---------: | ---------: | --------: |
| Baseline                          |  $9,593.34 |  $12,465.61 |   -0.0009 |
| Linear Regression                 |  $4,181.19 |   $5,796.28 |    0.7836 |
| Random Forest                     |  $2,560.71 |   $4,591.09 |    0.8642 |
| **Random Forest (Log-Transform)** | **$2,079.26** | **$4,373.23** | **0.8768** |
| XGBoost                           |  $2,449.76 |   $4,369.64 |    0.8770 |

- Best MAE: Random Forest (Log-Transform) — $2,079
- Best RMSE / R²: XGBoost — $4,370 / 0.8770
- Top charge drivers: `smoker`, `bmi`, `age`
