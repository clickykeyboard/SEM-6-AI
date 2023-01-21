import pandas as pd

data = pd.read_csv('data.csv')
x = data.iloc[:, 0].values
y = data.iloc[:, 1].values

def hypothesis(x, theta0, theta1):
  return theta0 + theta1 * x

# From task 1
def cost_function(theta_0, theta_1):
  # 1/2m * summation of (hθ(x) - y)^2
  m = len(x)

  total = 0
  for i in range(m):
    total += (hypothesis(theta_0, theta_1, x[i]) - y[i])**2
  return total / (2 * m)

# Gradient descent Algorithm

# we will assume no of iterations to be aroudn 1000
# and learning rate to be 0.01

# let initial values for theta0 and theta1 be 0
theta0 = 0
theta1 = 0 

learning_rate = 0.01
for i in range(1000):
  t0_grad = 0
  t1_grad = 0
  m = len(x) # number of training examples
  for j in range(m):
    t0_grad += -(2/m) * (y[j] - (theta0 + theta1 * x[j]))
    t1_grad += -(2/m) * x[j] * (y[j] - (theta0 + theta1 * x[j]))

  theta0 = theta0 - (learning_rate * t0_grad)
  theta1 = theta1 - (learning_rate * t1_grad)

print("theta0", theta0)
print("theta1", theta1)
print("Final cost:", cost_function(theta0, theta1))

# we can now use theta0 and theta1 to predict the salary of a person with x years of experience