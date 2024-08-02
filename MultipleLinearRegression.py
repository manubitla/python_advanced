import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
# import data
dataset = pd.read_csv("50_Startups.csv")

# identify dependant and independent variable
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# fix the missing values if any

# encoding the categorical data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
print(X)
print("************************")
X = np.array(ct.fit_transform(X))
print(X)

# in multiple linear regression we don't need to feature scale because every independent variable
# is multiplied by corresponding coefficient which results in high values also

# Splitting the data set into training and test set

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# the below multiple linear regression class will by default avoids dummy variable trap
# we use same linearRegression model for multiple linear regression. here the class itself checks
# that there are more features, and it acts as a multiple linear regression model

mlr = LinearRegression()
mlr.fit(X_train, Y_train)
# predicted test values

Y_pred = mlr.predict(X_test)

# Predict the test set results
print("Predict the test set results")
np.set_printoptions(precision=2)
# reshape to get the values vertically instead of horizontally
print(np.concatenate((Y_pred.reshape(len(Y_pred), 1), (Y_test.reshape(len(Y_test), 1))),1))

# ************** EXCERCISE 1*****************

#Question 1: How do I use my multiple linear regression model to make a single prediction,
# for example, the profit of a startup with R&D Spend = 160000, Administration Spend = 130000,
# Marketing Spend = 300000 and State = California?