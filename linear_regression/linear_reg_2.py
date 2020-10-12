# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 11:50:03 2018

@author: Raja
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
dataset = pd.read_csv('Salary_Data.csv')
dataset.head()  #for load top 5 values
dataset.tail()  #for load last 5 values 
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

"""split the dataset into two parts
-->X_train,Y_train
-->X_test,Y_test"""

#splitting the dataset into training set and test set

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 1/3,random_state=0) 
# uper 3rd perameter is optional
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train) #fit gives arrays
#predicting the test value
y_pred = regressor.predict(x_test)

#ploting the graph now


plt.xlabel("this is x axis")
plt.ylabel("this is y axis")
plt.plot(x_test,y_pred,c='red')
plt.scatter(x,y)
plt.legend(['x-coordinate','y-coordinate'])



"""plt.scatter(x_train,y_train,c='red')
plt.plot(x_train,regressor.predict())
plt.title("salery vs experience")
plt.xlabel("year of experience")
plt.ylabel("salary")
plt.show()"""