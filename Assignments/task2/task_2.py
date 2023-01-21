# Part 1

# Define linear function for gradient descent
def f(x):
  return 2*x + 1 # ax + b

# Let us take a step size for numerical differentiation
h = 0.0001

# This is the same h that would be used for h in 
# numerical differentiation for our numerical computing classes
# f'(x) = f(x + h) - f(x) / h

# Calculate gradient vector at the starting point 1
starting_point = [1]
gradient_vector = (f(starting_point[0] + h) - f(starting_point[0])) / h

gradient_vector_multiplied = gradient_vector * 2

# Since we chose 1 as starting point, its directions can be towards 1
new_point = [gradient_vector + gradient_vector_multiplied]

print("Gradient vector:", gradient_vector_multiplied)
print("New point:", new_point)

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 5, 100)
y = 2*x + 1 # This is our f(x)
plt.plot(x, y)
plt.scatter(1, f(1), c='r', marker='o')
plt.scatter(new_point[0], f(new_point[0]), c='b', marker='o')
plt.show()

# Part 2
max_iterations = 20
current_point = [1] # This is also same as starting point
for i in range(max_iterations):
  gradient_vector = [
      (f(current_point[0] + h) - f(current_point[0]))/h
  ]

  for i in range(len(current_point)):
    new_point[i] = current_point[i] + gradient_vector[i]

  if f(new_point[0]) < f(current_point[0]):
    break

  current_point = new_point


print("Final Point: ", current_point)

x = np.linspace(-5, 5, 100)
y = 2*x + 1
plt.plot(x, y)
plt.scatter(starting_point[0], f(starting_point[0]), c='r', marker='o')
plt.scatter(current_point[0], f(current_point[0]), c='b', marker='o')
plt.show()