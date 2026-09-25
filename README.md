# Insurance Charges Prediction

EDA and charges prediction on an insurance dataset, built as a Jupyter/Colab notebook.

## Dataset

- Source: [Kaggle — insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance) (auto-downloaded via `kagglehub`)
- 1,338 customer records, 7 columns (6 features + target), no missing values
- Target: `charges` (continuous, right-skewed)

## Table of Contents

- Data Loading and Initial Inspection
- Summary Statistics
- Check for Missing Values
- Exploratory Data Analysis
- Categorical Feature Distributions
- Numerical Feature Distributions
- Feature Correlation
- Impact of Categorical Features on Charges
- Impact of Numerical Features on Charges
- Model Building and Evaluation
- Baseline Model
- Linear Regression Model
- Random Forest Regressor Model
- Random Forest with Log Transformation
- Model Comparison
- Conclusion

## Results

| Model                             |        MAE |       RMSE |        R² |
| :-------------------------------- | ---------: | ---------: | --------: |
| Baseline                          |  $9,593.34 |  $12,465.61 |   -0.0009 |
| Linear Regression                 |  $4,181.19 |   $5,796.28 |    0.7836 |
| Random Forest                     |  $2,563.57 |   $4,594.50 |    0.8640 |
| **Random Forest (Log-Transform)** | **$2,079.99** | **$4,375.56** | **0.8767** |
| XGBoost                           |  $2,446.47 |   $4,347.48 |    0.8783 |

- Best MAE: Random Forest (Log-Transform) ($2,080)
- Best RMSE: XGBoost ($4,347)
- Best R²: XGBoost (0.878)
- Top charge drivers: `smoker`, `bmi`, `age`

## How to Run

Open `Med_Insurance_Charges_Prediction.ipynb` in Jupyter/Colab and run all cells.