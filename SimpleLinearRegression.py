# simple Linear Regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# importing data set
dataset = pd.read_csv("Salary_Data.csv")

# separate the feature and dependant variable

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Split the data into training and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create the object of LinearRegression class to predict and plot the results
lr = LinearRegression()
lr.fit(X_train, Y_train)
# Predict the test results
Y_predict = lr.predict(X_test)


# plot the chart for training set
plt.scatter(X_train, Y_train, color='green')
# plt.scatter(X_test, Y_test, color='red')
plt.plot(X_train, lr.predict(X_train), color='black')
plt.title("salary vs experience")
plt.xlabel("experience")
plt.ylabel("salary")
plt.show()

# plot the chart for test set

plt.scatter(X_test, Y_test, color='red')
plt.plot(X_train, lr.predict(X_train), color='black')
plt.title("salary vs experience")
plt.xlabel("experience")
plt.ylabel("salary")
plt.show()

# Making a single prediction (for example the salary of an employee with 12 years of experience)
print(lr.predict([[12]]))

# Getting the final linear regression equation with the values of the coefficients
print("b1: ", lr.coef_)
print("b0: ", lr.intercept_)