import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt
dataset = pd.read_csv("Position_Salaries.csv")
# here only the second index variable acts as features
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, -1].values

# no need to categorize the data in this example since the features are numeric
# no need to split the data into training and test set in SVR method.
# because we need to leverage the maximum data

# we did not feature scale the data in polynomial regression because every feature is multiplied by it's
# coefficient. in SVM we must feature scale the data
print("X before scaling: ")
print(X)
ss_X = StandardScaler()
X = ss_X.fit_transform(X)
print("X after scaling: ")
print(X)
# we should create another instance of StandardScaler class for dependent variable Y because the fit method of X
# will create mean and standard deviation of that particular column which we don't want to use for the salary column,
# so we must create another StandardScaler instance for variable Y
print("Y before scaling: ")
print(Y)
ss_Y = StandardScaler()
# the fit_transform method will take 2D array. since the Y is 1D array we need to convert it to 2D by reshaping it

Y = Y.reshape(len(Y), 1)
print("Y after reshaping: ")
print(Y)

Y = ss_Y.fit_transform(Y)
print("Y after scaling: ")
print(Y)
# create instance of SVR with Radial Basis Function RBF kernel which is recommended(model initiation)
svr = SVR(kernel="rbf")

# train the regressor on whole data
# the fit method here trains the data and finds the optimal hyperplane with in specified margin of tolerance
svr.fit(X, Y)
# now we need to predict the salary for the level 6.5, as in our example
# we must reverse scale the prediction to get the actual salary since the Y variable is also scaled
print(ss_Y.inverse_transform(svr.predict(ss_X.transform([[6.5]])).reshape(-1, 1)))

# visualizing the SVR results

plt.scatter(ss_X.inverse_transform(X), ss_Y.inverse_transform(Y), color='blue')
plt.plot(ss_X.inverse_transform(X), ss_Y.inverse_transform(svr.predict(X).reshape(-1, 1)), color='red')
plt.title("SVR predictions")
plt.xlabel("positions")
plt.ylabel("salary")
plt.show()

# visualize in high resolution
# inverse transform the X to get the x_grid with original scale, because it is scaled before
X_grid = np.arange(min(ss_X.inverse_transform(X)), max(ss_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
print(X_grid)
plt.scatter(ss_X.inverse_transform(X), ss_Y.inverse_transform(Y), color='blue')
plt.plot(X_grid, ss_Y.inverse_transform(svr.predict(ss_X.transform(X_grid)).reshape(-1, 1)), color='red')
plt.title("SVR predictions")
plt.xlabel("positions")
plt.ylabel("salary")
plt.show()


