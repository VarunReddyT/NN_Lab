# Build a regression model using Python & Scikit-learn
# a) Data Preparation
# b) Loading Data Set 
# c) EDA 
# d) Data Visualization (Histogram, Box Plot, Scatter Plot to analyze outliers)
# e) Split for training and testing and develop a regression model to predict the price of the house
# f) Evaluate model coefficients, calculate MSE, MAE using R square score to assess the model fit
# https://www.kaggle.com/datasets/camnugent/california-housing-prices

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score

data = pd.read_csv('housing.csv') 

print(data.info())
print(data.describe())
print(data.head())  

data = data.dropna()

plt.figure(figsize=(10, 5))
plt.hist(data['median_house_value'], bins=50, color='blue')
plt.title('Histogram of Median House Value')
plt.xlabel('Median House Value')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 5))
plt.boxplot(data['median_house_value'])
plt.title('Box Plot of Median House Value')
plt.ylabel('Median House Value')
plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(data['median_income'], data['median_house_value'], color='green')
plt.title('Scatter Plot of Median Income vs Median House Value')
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.show()

X = data[['median_income']]
Y = data['median_house_value']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean Squared Error:', mean_squared_error(Y_test, y_pred))
print('Mean Absolute Error:', mean_absolute_error(Y_test, y_pred))
print('R2 Score:', r2_score(Y_test, y_pred))
