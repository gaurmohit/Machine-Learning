# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 10:56:16 2018

@author: raja
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv('C:\\Users\\Raja\\Desktop\\ml\\Regression.csv', delim_whitespace =1)
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/4,random_state=0)


from sklearn.linear_model import LinearRegression 
regressor=LinearRegression()
regressor.fit(X_train, y_train)
#predicting the test
y_pred = regressor.predict(X_test)

#plotting the graph
plt.plot(X_test,y_pred,c='red')
plt.scatter(X,y)

