from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import numpy as np



#Loading the data in
diabetes = datasets.load_diabetes()

#Showing the featured names
print(diabetes.feature_names)

#Seperating the data into x and y
X = diabetes.data
Y = diabetes.target
#Showing the shape of the data
print(X.shape, Y.shape)

# Dividing the data into 4 parts the x and y train and test

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
# showing the new shapes

print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
#Make the model instance
model  = linear_model.LinearRegression

#Fit the model with teh train data
model.fit(X_train,Y_train)

#Create the predictions based on the x test 
Y_pred = model.predict(X_test)

#Pring out the variables of the line or regression line
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)

#Print out the amount of error as well as the how accurate the line was

print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_test, Y_pred))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_test, Y_pred))

r2_score(Y_test, Y_pred)
r2_score(Y_test, Y_pred).dtype
np.array(Y_test)



