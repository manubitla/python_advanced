
# 1. importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# importing data set
dataset = pd.read_csv("Data.csv")
print(dataset)
# dataset = dataset.dropna()
#  getting features and dependant value separately

x_features = dataset.iloc[:, :-1].values
print(x_features)
y_dependant_var = dataset.iloc[:, -1].values
print(y_dependant_var)

print("features with missing values: ", x_features)
# Taking care of missing data
imputor = SimpleImputer(missing_values=np.nan, strategy="mean")
# learn the data and strategy
imputor.fit(x_features[:, 1:3])
# actually change the empty values with mean
x_features[:, 1:3] = imputor.transform(x_features[:, 1:3])
print("features after filling missing values: ", x_features)

# one hot encoding with pandas
# data = pd.DataFrame({'color': ['red', 'green', 'blue', 'green', 'red']})
# print(data)
# one_hot_encoded_data = pd.get_dummies(data, columns=['color'])
# print(one_hot_encoded_data)

# Encoding the categorical values/independent variables
# create object of column transformer class
# ColumnTransformer constructor will take two arguments 1. transformers(what kind of
# transformation we want to do and which indexes of the columns we want to transform
# second argument is remainder which specifies on which columns we are not applying one hot encoder
# transformers will take 3 values. first kind of transformation(encoding), second what kind of
# encoding we want to do(one hot encoding), third indexes of columns we want to encode
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x_features = np.array(ct.fit_transform(x_features))
print("**************************************************")
print(x_features)
le = LabelEncoder()
print(y_dependant_var)
y_dependant_var = le.fit_transform(y_dependant_var)
print(y_dependant_var)

# Splitting the data set into training set and test set
# train_test_split(independant, dependant, size, random)
X_train, X_test, Y_train, Y_test = train_test_split(x_features, y_dependant_var, test_size=0.2, random_state = 1)
print("**********************")
print(X_train)
print("**********************")
print(Y_train)
print("**********************")
print(X_test)
print("**********************")
print(Y_test)
print("**********************")
# Feature scaling
ss = StandardScaler()
X_train[:, 3:] = ss.fit_transform(X_train[:, 3:])
# should not apply fit here because test set will take mean and median on training set
X_test[:, 3:] = ss.transform(X_test[:, 3:])
print(X_train)
print("**********************")
print(X_test)

