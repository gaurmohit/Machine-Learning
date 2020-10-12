# -*- coding: utf-8 
"""
Created on Mon Jul  2 12:42:25 2018

@author: Raja
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv('C:\\Users\\Raja\\Desktop\\ml\\Mall_customers.csv')
dataset.head()

x = dataset.iloc[:,[3,4]].values

from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',random_state = 42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Elbow method')
plt.xlabel('no.of clusters')
plt.ylabel('wcss')
plt.show()

kmeans = KMeans(n_clusters = 5, init='k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(x)

plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100,c='red',label='cluster1')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=100,c='blue',label='cluster2')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,c='green',label='cluster3')
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=100,c='pink',label='cluster4')
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],s=100,c='yellow',label='cluster5')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],s=300,c='black',label='centroids')

plt.title('clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.xlabel('spending score (1-100)')
plt.legend()
plt.show()
