# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Position_Salaries.csv")

# Want X to be a matrix, not a vector
# Want Y to be a vector
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

"""from sklearn.cross_validation import train_test_split
X_train, y_train, X_test, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)"""

# From Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#Fitting Polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

# Visualizing the Linear Regression results
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y, color='red')
plt.plot(X_grid,lin_reg.predict(X_grid), color='blue')
plt.title("Truth or Bluff (linear regressions)")
plt.xlabel("Position Level")
plt.ylabel("Salary")

# Visualizing the Polynomial Regression results
plt.plot(X,lin_reg_2.predict(X_poly),color='cyan')

# Predicting a new result with Linear Regression
lin_reg.predict(6.5)

# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))