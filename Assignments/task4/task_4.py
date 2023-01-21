# For task 4 we have multivariate linear regression
# the dataset in data.csv is used to take multiple inputs
# to predict output of Species based on the other 6 features

import numpy as np
import pandas as pd

data = pd.read_csv('data.csv')
x = data.iloc[:, 1:].values # We now put : after 1 so that columns are in the form of a matrix
y = data.iloc[:, 0].values

# Since our output is in the form of a string we need to convert it to a number
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

# hypothesis function
def hypothesis(x, theta):
  return np.dot(x, theta)
# This is used to perform dot product of x and theta since they are both matrices

# Same as task 1
def cost_function(theta):
  m = len(x)

  total = 0
  for i in range(m):
    total += (hypothesis(theta, x[i]) - y[i]) ** 2
  
  return total / (2 * m)

# Define the gradient descent algorithm
def gradient_descent(theta, learning_rate, num_iterations):
  m = len(x)
  for i in range(num_iterations):
    # x.T is basically the transpose of x
    theta = theta - (learning_rate / m) * np.dot(x.T, hypothesis(x, theta) - y)
  return theta

# Also convert x and y into numpy arrays
x = np.array(x)
y = np.array(y)


# Initial value of theta
# This will be in form [0, 0, 0, 0, 0, 0] 
# So that we can perform dot product with x
theta = np.zeros(6) 

# Same as task 3
learning_rate = 0.0001
num_iterations = 100

theta = gradient_descent(theta, learning_rate, num_iterations)

print("theta", theta)
print("Final cost:", cost_function(theta))