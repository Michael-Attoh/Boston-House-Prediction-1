# Michael's Portfolio

# [Boston House Price Prediciton](https://github.com/Michael-Attoh/Boston-House-Prediction-1/tree/main)

This project aims to predict house prices in Boston using machine learning techniques. The dataset used for this project is the well-known Boston Housing Dataset, which contains information collected by the U.S. Census Service concerning housing in the area of Boston, Massachusetts.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Results](#results)
- [Conclusion](#conclusion)

## Introduction

The goal of this project is to explore the Boston Housing Dataset and build regression models to predict the median value of owner-occupied homes. The project involves data preprocessing, feature engineering, model selection, and evaluation.

## Dataset

The Boston Housing Dataset contains 506 observations and 14 variables. Key features include crime rate, average number of rooms per dwelling, and proximity to employment centers.

## Project Workflow

1. **Data Preprocessing:** Handling missing values, data transformation (e.g., log transformation), and feature scaling.
2. **Exploratory Data Analysis (EDA):** Understanding data distribution, relationships between features, and outliers.
3. **Modeling:** Implemented regression models such as Linear Regression, Ordinary Least Squares (OLS), and Random Forest Regressor.
4. **Evaluation:** Assessed model performance using metrics like Root Mean Squared Error (RMSE).
5. **Future Work:** Planned tasks include Ridge Regression, Lasso Regression, and GridSearchCV for hyperparameter tuning.


## Modeling

We used the following regression techniques:

- **Ordinary Least Squares (OLS) Regression**
- **Linear Regression**
- **Random Forest Regressor**

## Evaluation

The models were evaluated based on:

- **Root Mean Squared Error (RMSE):** Measures the model’s prediction error, with lower RMSE values indicating better model performance.
- **R-squared (R²) and Adjusted R-squared:** These metrics were considered but not utilized in this analysis.


## Results

The best-performing model was Random Forest Regressor, achieving a RMSE of 3.23 on the test data.

## Conclusion

This project demonstrates the process of building and evaluating regression models for predicting housing prices using the Boston Housing dataset. The analysis involved developing and comparing three different models: Ordinary Least Squares (OLS) regression, Linear Regression, and Random Forest Regressor.

**Key Findings:**

- **Model Performance:** The Random Forest Regressor outperformed both the OLS and Linear Regression models, as evidenced by its lower Root Mean Squared Error (RMSE). This indicates that the Random Forest model provided more accurate predictions of housing prices.
- **Model Suitability:** While the Random Forest model demonstrated superior accuracy, it is also more complex and less interpretable than the linear models. The Linear Regression model, though less accurate, offers greater transparency and simplicity.
- **Recommendations:** For predictive accuracy, the Random Forest model is recommended. For clear interpretability, the Linear Regression model remains a viable option.

**Limitations:**

- Potential for overfitting in the Random Forest model, recommending further validation with additional data.
- Linear assumptions of the OLS model may not fully capture the underlying data complexities.

Overall, the project underscores the importance of selecting the appropriate model based on analysis needs—whether for accuracy or interpretability. Future work could involve further tuning of the Random Forest model and exploring additional data sources to enhance performance.

