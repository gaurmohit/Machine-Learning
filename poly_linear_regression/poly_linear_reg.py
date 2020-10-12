#polynomial regression algorithm

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
data = pd.read_csv('Position_Salaries.csv')
data.head()

x = data.iloc[:,1:2].values
y = data.iloc[:,2].values


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)
             
#fitting polynomial regression to the data
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3) 
x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly, y)

#visualising the polynomial regression
#plt.scatter(x,y,c='purple')
#plt.plot(x,y,c='yellow')
 
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

plt.scatter(x,y,c='red')

plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)),c='blue')
plt.title("truth or bluff (polynomial regression)")
plt.xlabel("this is the position level")
plt.ylabel('salary')

lin_reg.predict(7)
y_pre = lin_reg_2.predict(poly_reg.fit_transform(7))

x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,c='red')
plt.plot(x_grid,lin_reg_2.predict(poly_reg.fit_transform(x_grid)),c='blue')

plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)),c='blue')
plt.title("truth or bluff (polynomial regression)")
plt.xlabel("this is the position level")
plt.ylabel('salary')


