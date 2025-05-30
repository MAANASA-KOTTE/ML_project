**🚗 Mileage Prediction - Regression Analysis**
This project applies machine learning techniques to predict a car's Miles Per Gallon (MPG) using regression analysis. The dataset used comes from the StatLib library maintained at Carnegie Mellon University, which was originally utilized in the 1983 American Statistical Association Exposition.

**📊 Dataset Overview**
Source: StatLib, Carnegie Mellon University

Attributes:

mpg – Miles per gallon (target variable)

cylinders – Multi-valued discrete

displacement – Continuous

horsepower – Continuous

weight – Continuous

acceleration – Continuous

model_year – Multi-valued discrete

origin – Multi-valued discrete (categorical)

name – Car name (string, unique)

**🛠️ Libraries Used**
```
python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

```
**🔍 Data Preprocessing**

Dropped missing values (horsepower had some missing entries).

Dropped non-numeric and less useful features (origin, name) for regression.

Standardized the numerical features (displacement, horsepower, weight, acceleration).

**📈 Model Used**

Linear Regression

Trained using 70% of the dataset.

Coefficients and intercept learned.

Model predicts mpg using 4 input features.

**✅ Model Performance**

Mean Absolute Error (MAE): 3.33

Mean Absolute Percentage Error (MAPE): 14.71%

R² Score: 0.703 — The model explains 70.3% of the variance in MPG.

**📌 Visualizations**

Pairplots and regression plots for exploratory data analysis.

Strong negative correlation observed between MPG and features like displacement, horsepower, and weight.

**📁 How to Run**

Clone this repository or open it in Google Colab.

Ensure all required libraries are installed.

Run all cells in order for preprocessing, training, and evaluation.

📌 Note

This is a basic regression model and can be further improved by:

-Adding polynomial features

-Applying regularization (Ridge/Lasso)

-Hyperparameter tuning
