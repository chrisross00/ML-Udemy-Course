


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Position_Salaries.csv")
# Want X to be a matrix, not a vector
# Want Y to be a vector
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2:].values

"""from sklearn.cross_validation import train_test_split
X_train, y_train, X_test, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)"""

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
#y = sc_y.fit_transform(np.array(y).reshape(-1, 1))

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)

# Predicting a new result
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))
y_pred = regressor.predict(6.5)

y_pred = sc_y.inverse_transform(y_pred)

# Visualizing the Linear Regression results
plt.scatter(X,y, color='red')
plt.plot(X,regressor.predict(X), color='blue')
plt.title("Truth or Bluff (linear regressions)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
