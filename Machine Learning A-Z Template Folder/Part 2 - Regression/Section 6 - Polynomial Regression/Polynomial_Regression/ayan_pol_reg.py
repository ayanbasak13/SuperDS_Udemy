#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing the dataset
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,-1].values





#Fitting Simple Linear Regression Model to Training Set
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)



#Fitting Polynomial Linear Regression Model to Training Set
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)

linreg2 = LinearRegression()
linreg2.fit(X_poly,y)







#LINEAR REGRESSION
plt.scatter(X,y,color = 'red')
plt.plot(X,lin_reg.predict(X),color = 'blue')
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()



#POLYNOMIAL REGRESSION
X_grid = np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color = 'red')
plt.plot(X_grid,linreg2.predict(poly_reg.fit_transform(X_grid)),color = 'blue')
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()



#Predicting new result with Linear Regression
lin_reg.predict(6.5)



#Predicting new result with Polynomial Regression
linreg2.predict(poly_reg.fit_transform(6.5))