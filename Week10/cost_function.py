import numpy as np
import pandas as pd

dataset = pd.read_csv("dataset.csv")
x = dataset.iloc[:, 0].values
y = dataset.iloc[:, -1].values

# theta0 and theta1
parameters = np.array([1, 1])

# Hypothesis: h(x) = theta0 + theta1 * x
def hypothesis(x, parameters):
  return parameters[0] + parameters[1] * x

def cost_function(x, y, parameters):
  m = len(x)
  total_error = 0
  for i in range(m):
    total_error += (hypothesis(x[i], parameters) - y[i])**2
  
  return total_error / (2 * m)

costs = {}

def calculate_cost():
  learning_rate = 0.0001
  m = len(x)

  # Gradient descent (for adjusting parameters)
  for i in range(10000):
    theta0_deriv = 0
    theta1_deriv = 0
    for j in range(m):
      # Calculate partial derivatives
      theta0_deriv += (hypothesis(x[j], parameters) - y[j])
      theta1_deriv += (hypothesis(x[j], parameters) - y[j]) * x[j]
    
    # Update parameters
    parameters[0] = parameters[0] - learning_rate * theta0_deriv
    parameters[1] = parameters[1] - learning_rate * theta1_deriv

    costs[f"{parameters[0]},{parameters[1]}"] = cost_function(x, y, parameters)

calculate_cost()

# e.g 24253,9677
parameters = min(costs, key=costs.get).split(",")
parameters[0] = float(parameters[0])
parameters[1] = float(parameters[1])

print("theta0:", parameters[0])
print("theta1:", parameters[1])

x = float(input("Enter x value to predict y: "))
print("y:", hypothesis(x, parameters))