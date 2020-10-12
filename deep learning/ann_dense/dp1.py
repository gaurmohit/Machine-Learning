# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 09:46:47 2018

@author: Raja
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv('Churn_Modelling.csv')
dataset.head()
#theano+tensorflow=keras

x= dataset.iloc[:,3:13].values
y= dataset.iloc[:,13].values
               
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
x[:,1]=labelencoder.fit_transform(x[:,1])
onehotencoder=OneHotEncoder(categorical_features= [1])

labelencoder2=LabelEncoder()
x[:,2]=labelencoder2.fit_transform(x[:,2])
onehotencoder=OneHotEncoder(categorical_features= [1])
x=onehotencoder.fit_transform(x).toarray()

#avoiding the dummy variable trap
x=x[:,1:]
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/2,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
 
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)


"""





#KNN
from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier(n_neighbors=5,metric ='minkowski',p=2)
#p=2 for ecucledean distance
clf.fit(x,y)
prediction=clf.predict(x_test)


#kmeans
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++')
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)

plt.title('the elbow model')
plt.xlabel('number of clusters')
plt.ylabel('WCSS')

kmeans=KMeans(n_clusters=5,init='k-means++',random_state=42)
y_means=kmeans.fit_predict(x)

plt.scatter(x[y_means==0,0],x[y_means==0,1],s=100,c='red',label='cluster1')
plt.scatter(x[y_means==1,0],x[y_means==1,1],s=100,c='red',label='cluster2')
plt.scatter(x[y_means==2,0],x[y_means==2,1],s=100,c='red',label='cluster3')
plt.scatter(x[y_means==3,0],x[y_means==3,1],s=100,c='red',label='cluster4')
plt.scatter(x[y_means==4,0],x[y_means==4,1],s=100,c='red',label='cluster5')
plt.scatter(s=300,c='yellow',)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)


#svmc
from sklearn.svm import SVC
classfier= SVC(kernel='linear',random_state=0)

classfier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)


#gaussian kernal

from sklearn.svm import SVC


df=SVC(kernel='rbf',random_state=0)
df.fit(x_train,y_train)
y_pred=df.predict(x_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

plt.plot(x_train,y_train)


#naivebyes
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)








#decision tree
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(x_train,y_train) 
y_pred=classifier.predict(x_test)   
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)


"""
import keras 
from keras.models import Sequential
from keras.layers import Dense


classifier=Sequential()
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))

#add a new hidden layer of logic 
#dense function add in the hidden layer


classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))

classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))


classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(x_train,y_train,batch_size=10,nb_epoch=5)


y_p=classifier.predict(x_test)

y_p.dtype=np.int32


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_p)
 











