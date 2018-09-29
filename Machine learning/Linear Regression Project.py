import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#Figure out whether to focus the company's efforts on their mobile app experience or their website.

customers = pd.read_csv("Ecommerce Customers")

#Training and Testing Data
y = customers['Yearly Amount Spent']#The True values that we want to predict from the dataframe.
X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']] #The features of the learning.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101) #Split the data into training and testing sets.

linearRegressionModel = LinearRegression()
linearRegressionModel.fit(X_train,y_train)#Train the model using the training set.

predictions = linearRegressionModel.predict( X_test) #evaluate its performance by predicting off the test values.

#Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error.
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

plt.scatter(y_test,predictions)#Create a scatterplot of the real test values versus the predicted values.
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()

coeff_df = pd.DataFrame(linearRegressionModel.coef_,X.columns,columns=['Coefficient'])#Show the coefficients of each feature.
print(coeff_df) 
