# -*- coding: utf-8 -*-
"""
Created on Sun May 17 18:43:34 2020

@author: user1
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('boston_house_prices.csv')
dataset.head()

X= dataset.drop('MEDV', axis=1)
#X1= dataset.iloc[:,:-1]
y= dataset.iloc[:,-1]

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
sc_X.fit(X)
X = sc_X.transform(X)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(include_bias = False)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
print(X_poly.shape)




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, random_state = 42)

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
# regressor = LinearRegression()
# regressor.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score
print("Model Score:{:.2f}".format(np.mean(cross_val_score(LinearRegression(),X_train, y_train,cv=10))))

from sklearn.model_selection import GridSearchCV
parameters={"alpha":np.logspace(-3, 3, 7)}
grid=GridSearchCV(Ridge(),
                  param_grid=parameters,
                  cv=10,
                  return_train_score=True)
grid.fit(X_train,y_train)
result=pd.DataFrame(grid.cv_results_)

result.plot("param_alpha", ["mean_train_score","mean_test_score"],logx=True)