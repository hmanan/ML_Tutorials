# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 16:16:52 2020

@author: Harnoor
"""
# Importing the libraries
 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset

dataset = pd.read_csv('data.csv')

# Independent variable (featues)
X = dataset.iloc[:, :-1].values


# Dependent variable
y = dataset.iloc[:, -1].values

# print(X)
# print(y)

# Taking care of missing data

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# print(X)

# Encoding the independent variable

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# print(X)

# Encoding the independent variable

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)
# print(y)

# Splitting Data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)  

# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)


# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])

X_test[:, 3:] = sc.fit_transform(X_test[:, 3:])

print(X_train)
print(X_test)
