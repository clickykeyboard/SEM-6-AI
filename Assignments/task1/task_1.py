import pandas as pd

# Part 1
data = pd.read_csv('data.csv')
x = data.iloc[:, 0].values
y = data.iloc[:, 1].values

def hypothesis(theta_0, theta_1, x):
  # θ0 + θ1 * x
  return theta_0 + theta_1 * x

def cost_function(theta_0, theta_1):
  # 1/2m * Σ(hθ(x) - y)^2
  m = len(x)

  total = 0
  for i in range(m):
    total += (hypothesis(theta_0, theta_1, x[i]) - y[i]) ** 2
  
  return total / (2 * m)

theta_0 = float(input("Enter θ0: "))
theta_1 = float(input("Enter θ1: "))

print("Cost: ", cost_function(theta_0, theta_1))

# Part 2 (return parameters of best model)
parameters_group = []
best_parameters = {}
for i in range(2):
  print("Enter parameters for model", i + 1)
  theta_0 = float(input("Enter θ0: "))
  theta_1 = float(input("Enter θ1: "))

  parameters_group.append([theta_0, theta_1])

first_parameters = parameters_group[0]
second_parameters = parameters_group[1]

print(first_parameters, second_parameters)
first_cost = cost_function(first_parameters[0], first_parameters[1])
second_cost = cost_function(second_parameters[0], second_parameters[1])

if first_cost < second_cost:
  print("Best parameters are: ", first_parameters)
  best_parameters = first_parameters
else:
  print("Best parameters are: ", second_parameters)
  best_parameters = second_parameters

# Part 3
print("Enter years experience: ")
years_experience = float(input())

print("Predicted salary: ", hypothesis(best_parameters[0], best_parameters[1], years_experience))
