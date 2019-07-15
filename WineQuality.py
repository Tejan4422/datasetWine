# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:55:01 2019

@author: Tejan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv('WineQualityRed.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,11].values

#splitting dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#fitting Linear model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#prediction
y_pred = regressor.predict(X_test)


#building optimal model using backward elimindation
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((1599, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0,1,2,3,4,5,6,7,9,10,11]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


X_opt = X[:, [0,1,2,3,5,6,7,9,10,11]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,2,3,5,6,7,9,10,11]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


X_opt = X[:, [0,2,5,6,7,9,10,11]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


X_opt = X[:, [0,2,5,6,7,9,10,11]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

