# practice
"""
Created on Fri Jun 29 14:33:10 2018

@author: HPr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv('data.csv') 
data.head()
X=data.iloc[:,:-1].values
y=data.iloc[:,6].values

##splitting in to train and split
from sklearn.cross_validation import train_test_split     
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

##featuring scalering
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy')##,random_state=0)
classifier.fit(X_train,y_train)

##predicting the test results.
y_predict=classifier.predict([[4,3,2,1,2,3]])
