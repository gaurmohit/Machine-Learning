#multilinear regression algorithm

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
data = pd.read_csv('50_Startups.csv')
data.head()

x = data.iloc[:,:-1].values

y = data.iloc[:,4].values

#encoding of the catagorical var by creating the object
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder = LabelEncoder()
x[:,3] = labelencoder.fit_transform(x[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

#avoiding the dummy var and drop them
x = x[:,1:]

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2) #for 20% 
# uper 3rd perameter is optional
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train) #fit gives arrays
#predicting the test value
y_pred = regressor.predict(x_test)

#building the optical model using the backword elimination
import statsmodels.formula.api as sm
x = np.append(arr = np.ones((50,1)).astype(int),values = x,axis = 1)
"""                 |   |       |      |     |                   |
               numpyarray   anfirstarray  datatype                 row wise
                        |              |                 0 for colomn wise 
                       add    float to int convertor
"""
x_opt=x[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
#OLS used for calc, in this case we are focusing in calc of p value
regressor_OLS.summary()

#1st largest value
x_opt=x[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
#OLS used for calc, in this case we are focusing in calc of p value
regressor_OLS.summary()

#2nd largest value
x_opt=x[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
#OLS used for calc, in this case we are focusing in calc of p value
regressor_OLS.summary()

#3nd largest value
x_opt=x[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
#OLS used for calc, in this case we are focusing in calc of p value
regressor_OLS.summary()

#4nd largest value
x_opt=x[:,[0,3]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
#OLS used for calc, in this case we are focusing in calc of p value
regressor_OLS.summary()
#we got final answer

plt.xlabel("this is x axis")
plt.ylabel("this is y axis")
plt.plot(x_test,y_pred)
plt.legend(['R&D','administrator','marketing','state'])

#plt.scatter(x_test,y_pred)
