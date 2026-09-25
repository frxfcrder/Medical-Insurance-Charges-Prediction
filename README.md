# Insurance Charges Prediction

EDA and charges prediction on an insurance dataset, built as a Jupyter/Colab notebook.

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

| Model | MAE | RMSE | R² |
| :--- | ---: | ---: | ---: |
| Baseline | $9,593 | $12,466 | -0.001 |
| Linear Regression | $4,181 | $5,796 | 0.784 |
| Random Forest | $2,564 | $4,595 | 0.864 |
| **Random Forest (Log-Transform)** | **$2,080** | **$4,376** | **0.877** |

- Best model: Random Forest on log-transformed target
- Top charge drivers: `smoker`, `bmi`, `age`

## How to Run

Open `Insurance_Charges_Analysis.ipynb` in Jupyter/Colab and run all cells.