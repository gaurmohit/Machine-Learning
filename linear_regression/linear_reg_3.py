#linear regression algorithm

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
data = pd.read_csv('home.csv')
data.head()
x = data.iloc[1:17:,1].values
y = data.iloc[1:17:,2].values
             
x=pd.to_numeric(x)
y=pd.to_numeric(y)
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 1/3,random_state=0) 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train.reshape(-1,1),y_train) 
y_pred = regressor.predict(x_test.reshape(-1,1))


plt.xlabel("this is x axis")
plt.ylabel("this is y axis")
plt.plot(x_test,y_pred,c='red')
plt.scatter(x,y)
plt.legend(['x-coordinate','y-coordinate'])
