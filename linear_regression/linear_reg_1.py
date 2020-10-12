# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 10:35:12 2018

@author: Raja
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
#reading the data
data = pd.read_csv('C:\\Users\\Raja\\Desktop\\ml\\headbrain.csv')
data.head()  #for load top 5 values
data.tail()  #for load last 5 values
#collecting x and y
x = data['Head Size(cm^3)'].values
y = data['Brain Weight(grams)'].values
#mean of x and y
mean_x=np.mean(x)
mean_y=np.mean(y)
#total n. of values
m=len(x)
#using the formula for calc B0 and B1
numer = 0
denom = 0
for i in range(m):
    numer += (x[i]-mean_x)*(y[i]-mean_y)
    denom +=(x[i]-mean_x)**2
            
B1 = numer/denom    
B0 = mean_y - (B1 * mean_x)
#print coefficent now
print(B1,B0)
#Brain Weight(grams)=325.57___+0.2634____*Head Size(cm^3)
##now ploting the graph now

plt.scatter(x,y,c='red')
plt.xlabel("this is X axis")
plt.ylabel("this is Y axis")
y=B0+B1*x
plt.plot(x,y)

# x=np.linspace(min_x,max_x,1000)

