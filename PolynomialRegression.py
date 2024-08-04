import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# importing data
dataset = pd.read_csv("Position_Salaries.csv")

# identify dependant and independent variables
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, -1].values

# training the linear regression model on whole data
lr = LinearRegression()
lr.fit(X, Y)
Y_pred = lr.predict(X)

# training the polynomial regression model
# degree is 'n' in the formula
poly_regressor = PolynomialFeatures(degree=4)
# the squares of the x
X_poly = poly_regressor.fit_transform(X)
lr_2 = LinearRegression()
lr_2.fit(X_poly, Y)

# Visualizing linear regression model
plt.scatter(X, Y, color='green')
plt.plot(X, lr.predict(X), color='blue')
plt.title("Linear regression Level vs Salary")
plt.xlabel("position level")
plt.ylabel("salary")
plt.show()

# Visualizing Polynomial regression model
plt.scatter(X, Y, color="black")
plt.plot(X, lr_2.predict(X_poly), color='red')
plt.title("Polynomial regression level vs salary")
plt.xlabel("position level")
plt.ylabel("salary")
plt.show()

# visualizing polynomial regression for a higher resolution and smoother curve

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, Y, color='black')
plt.plot(X_grid, lr_2.predict(poly_regressor.fit_transform(X_grid, Y)), color='red')
plt.title("Polynomial regression level vs salary")
plt.xlabel("position level")
plt.ylabel("salary")
plt.show()

# predicting a new salary with linear regression
print("predicted salary for position 6.5 according to linear regression is : ")
print(lr.predict([[6.5]]))
# predicting a new salary with polynomial regression
print("predicted salary for position 6.5 according to polynomial regression is : ")
print(lr_2.predict(poly_regressor.fit_transform([[6.5]])))