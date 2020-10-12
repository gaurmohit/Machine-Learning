# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 15:26:37 2018

@author: HPr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv('knn.csv') 
data.head()
X=data.iloc[:,0:4].values
y=data.iloc[:,-1].values

##splitting in to train and split
from sklearn.cross_validation import train_test_split     
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

##featuring scalering
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski' , p=2) ##considering thar we are given 5 points and p=2 is used for using eulicidian method
classifier.fit(X_train,y_train)



##predicting the test results.
y_predict=classifier.predict([[5,3,1,0]])



