import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
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

# in multiple linear regression we dont need to feature scale because every independent variable
# is multiplied by corresponding coefficient which results in high values also

# Splitting the data set into training and test set