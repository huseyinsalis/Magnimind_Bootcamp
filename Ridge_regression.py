# -*- coding: utf-8 -*-
"""
Created on Sun May 17 18:22:38 2020

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

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

from sklearn.linear_model import Ridge
regressor = Ridge()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

print("Traininng Accuracy:{:.2f}".format(regressor.score(X_train, y_train)))
print("Test Accuracy:{:.2f}".format(regressor.score(X_test, y_test)))

from sklearn.model_selection import GridSearchCV
parameters={"alpha":np.logspace(-3, 3, 7)}
grid=GridSearchCV(regressor,
                  param_grid=parameters,
                  cv=10,
                  return_train_score=True)
grid.fit(X_train,y_train)
result=pd.DataFrame(grid.cv_results_)

result.plot("param_alpha", ["mean_train_score","mean_test_score"],logx=True)
