# Michael's Portfolio

# [Boston House Price Prediction](https://github.com/Michael-Attoh/Boston-House-Prediction-1/tree/main)

This project aims to predict house prices in Boston using machine learning techniques. The dataset used for this project is the well-known Boston Housing Dataset("boston.csv"), which contains information collected by the U.S. Census Service concerning housing in the area of Boston, Massachusetts.

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

The Boston Housing Dataset("boston.csv") contains 506 observations and 14 variables. Key features include crime rate, average number of rooms per dwelling, and proximity to employment centers.
### Import libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns #for the regression plots
#%matplotlib inline ## Optional, as Colab does this by default
```
```python
boston = pd.read_csv('boston.csv')  # load data
```
<img width="838" alt="data" src="https://github.com/user-attachments/assets/d1acf194-4755-4867-89e4-ad86ff00a491">


## Project Workflow

1. **Data Preprocessing:** Handling missing values, data transformation (e.g., log transformation), and feature scaling.
2. **Exploratory Data Analysis (EDA):** Understanding data distribution, relationships between features, and outliers.
3. **Modeling:** Implemented regression models such as Linear Regression, Ordinary Least Squares (OLS), and Random Forest Regressor.
4. **Evaluation:** Assessed model performance using metrics like Root Mean Squared Error (RMSE).
5. **Future Work:** Planned tasks include Ridge Regression, Lasso Regression, and GridSearchCV for hyperparameter tuning.

## Data Preprocessing: 
Handling missing values, data transformation (e.g., log transformation), and feature scaling.
1. Data loading and inspection.
2. Handling missing values
3. handling outliers

### Missing values
```python
boston.isnull().sum()   #checking for missing values
```
<img width="129" alt="missing" src="https://github.com/user-attachments/assets/7dd0f50d-aef6-4e5d-a2a8-d87237d4ef89">


### Imputation with median 
```python
boston.fillna(boston.median(), inplace=True)
```
<img width="1302" alt="imput-describe" src="https://github.com/user-attachments/assets/faa86b66-b583-4485-a9e2-a9cefa4c552f">


## Exploratory Data Analysis (EDA):
1. What is the overall Price trend?
2. Explore the relationship between the target variable(Price) and the predictor variables.
3. What is the distribution of the selected variables?

### Relationship Between Rooms and Prices
```python
sns.regplot(x='RM', y='MEDV', data = boston, fit_reg=True) # this fits a linear regression line to data
plt.title('Relationship Between Rooms and Prices')
plt.show()
```
<img width="618" alt="eda1" src="https://github.com/user-attachments/assets/c3110210-55f6-4f14-a921-c909fda47116">

### Relationship Between Population and Prices
```python
sns.regplot(x='LSTAT', y='MEDV', data = boston, fit_reg=True) # this fits a linear regression line to data
plt.title('Relationship Between Population and Prices')
plt.show()
```
<img width="576" alt="eda2" src="https://github.com/user-attachments/assets/90d9da15-1ec7-4cb8-bbea-78e5d1e54c3f">

### Relationship Between Nitric Oxide and Prices
```python
sns.regplot(x='NOX', y='MEDV', data = boston, fit_reg=True) # this fits a linear regression line to data
plt.title('Relationship Between Nitric Oxide and Prices')
plt.show()
```
<img width="598" alt="eda3" src="https://github.com/user-attachments/assets/f82637f6-6d5e-4284-8e07-bc493efc618a">

### Relationship btw Distance to 5 Boston employment centers & Prices
```python
sns.regplot(x='DIS', y='MEDV', data = boston, fit_reg=True)
plt.title('Relationship btw Distance to 5 Boston employment centers & Prices' )
plt.show()
```
<img width="636" alt="eda4" src="https://github.com/user-attachments/assets/6c1fc8c4-b470-4c7d-98c0-d7e5ec3423ae">

### Relationship Between Pupil Teacher & Prices
```python
sns.regplot(x='PTRATIO', y='MEDV', data = boston, fit_reg = True)
plt.title('Relationship Between Pupil Teacher & Prices')
plt.show()
```
<img width="623" alt="eda5" src="https://github.com/user-attachments/assets/27e511a7-c452-4712-9e49-0398051529dd">

### Relationship Between Crime rate & Prices
```python
sns.regplot(x='CRIM', y='MEDV', data = boston, fit_reg= True)
plt.title('Relationship Between Crime rate & Prices')
plt.show()
```
<img width="627" alt="eda6" src="https://github.com/user-attachments/assets/2ef59c53-2ea2-49e9-9710-7f7a7c0f33e5">

   
## Modeling/Analysis
- variable selection
```python
boston.columns
boston_selected_var_df = boston.iloc[:,[0,4,5,7,10,12]]
boston_selected_var_df.head(10)
```
<img width="439" alt="selected1" src="https://github.com/user-attachments/assets/07034c77-060e-4d83-9093-413d3add585d">


## Data Transformation

1. Transform/normalize the variables using the 'log transformation'.
```python
# Adding a small constant to avoid log(0) and log of negative values
# Normalizing the skewness for CRIM, DIS, NOX
small_constant = 1e-6
boston_selected_var_df['CRIM'] = np.log(boston_selected_var_df.CRIM + small_constant)
boston_selected_var_df['DIS'] = np.log(boston_selected_var_df.DIS + small_constant)
boston_selected_var_df['NOX'] = np.log(boston_selected_var_df.NOX + small_constant)
boston_selected_var_df
```
<img width="508" alt="selected 2" src="https://github.com/user-attachments/assets/efc51e84-184b-46a3-b554-78b2354c6a46">

## Correlation Check

```python
boston_selected_var_df.corr()  #clearly there exits some correlation so we can use the variables as predictors
```
<img width="591" alt="corr" src="https://github.com/user-attachments/assets/02fded05-9ca5-42c9-ae46-fe93bd657387">


## Side/sample task
- perform an OLS analysis on the whole data.

####  Include the Target variable 'Price' or MEDV in the selected data
```python
boston_selected_var_df['PRICE'] = boston.MEDV

```
#### Create the OLS model
```python
import statsmodels.api as sm
from statsmodels.formula.api import ols #ols to estimate the unknown parameters in linear regression model
```
- #model
#### Predict Price using the corresponding variables from the data and fit it to OLS

```python
model = ols('PRICE ~ CRIM + NOX + RM + DIS + PTRATIO + LSTAT',boston_selected_var_df).fit()
```
#### check the job done by the model
```python
print(model.summary())
```
<img width="794" alt="modelsummary1" src="https://github.com/user-attachments/assets/4c9705ce-d070-4b68-bc1d-e1cacf9e48fe">


#### Interpretation
RM             4.4882 

- A unit increase in RM(room) will cause a Price increase of $4,4882 

NOX          -16.7569

- A unit increase in NOX(Nitric Oxide Conc) will cause a Price decrease of $16,7569

CRIM rate p = 0.958 > 0.05 

- Crime is not a significant indicator in this model. So we can eliminate it.

Durbin-Watson:   1.019 < 2

- Meaning the variables are not significantly autocorrelated. we can use them.

Key note:

R-squared:       0.712 
- R-squared tells how well the model is performing given the VARs = 71.2%

Adj. R-squared:  0.709 = 70.9%

- True indicator of the models performance, as it tells you whether an added variavle is significant or not.
- Adjusted R² provides a more accurate measure of model fit when multiple predictors are used, helping to prevent overfitting by penalizing the addition of unnecessary variables.

#### Check the Root Mean Square Error
```python
from sklearn.metrics import mean_squared_error

predicted_prices = model.fittedvalues
error = np.sqrt(mean_squared_error(boston_selected_var_df.PRICE,predicted_prices))
error

```
- #4.929596855776021. Meaning we have  +- $4,9295 margin of error.

# Back to Main Task/Project

We used the following regression techniques:
- **Linear Regression**
- **Random Forest Regressor**
## Linear Regression
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

x = boston_selected_var_df.drop('PRICE', axis = 1)
y = boston_selected_var_df['PRICE']

# Split and use for all models 
X_train,X_test,Y_train,Y_test = train_test_split(x,y, test_size = .3, random_state = 42)

LinReg = LinearRegression() # Linear regression model initialized
LinReg.fit(X_train, Y_train)

Y_pred = LinReg.predict(X_test)

plt.scatter(Y_test,Y_pred) #show the relationship

```
<img width="582" alt="positiverelation1" src="https://github.com/user-attachments/assets/e77152fd-554a-4d1b-99a8-5929b8bae4a5">

## Random Forest Regressor
```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42) #rf model initialized

rf.fit(X_train, Y_train)

yrf_pred = rf.predict(X_test)  # apply the trained model on the holdout data

```


## Evaluation

The models were evaluated based on:

- **Root Mean Squared Error (RMSE):** Measures the model’s prediction error, with lower RMSE values indicating better model performance.
### Linear Regression
```python
np.sqrt(mean_squared_error(Y_test,Y_pred))
```
#### root mean square error
meaning we have this much of error in our prediciton.
4.619269764844031. = +- $4,619269 margin of error
 #### interpretation 
 - On average, the predictions made by the linear regression model are about 4.62 units away from the actual values in the test data (Y_test)

- So, if you're predicting house prices, the model's predictions are off by an average of +- $4,620.

### Random Forest Regressor
```python
np.sqrt(mean_squared_error(Y_test, yrf_pred))
```
#### Interpretation:

- On average, the predictions made by RFregressor model are about 3.23 units away from the actual values in the test data (Y_test).

-  So, if you're predicting house prices, the model's predictions are off by an average of +- $3,230.


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


## Reference 
1. [Sklearn](https://scikit-learn.org/stable/)
2. Boston Housing Price Data
