#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 20:40:15 2018

@author: Chris
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

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Decision Tree regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y)

# Predicting a new result
y_pred = regressor.predict(6.5)

# Visualizing the Decision Tree Regression results
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y, color='red')
plt.plot(X_grid,regressor.predict(X_grid), color='blue')
plt.title("Truth or Bluff (linear regressions)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
