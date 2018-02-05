#!/usr/bin/python

# Arturo Perez 
# Professor Koutsoukos
# CS 3891 
# 25 January 2018 

import os
import numpy as np
import struct
import matplotlib.pyplot as plt

# initalize variables 
costs = []
data = np.genfromtxt("food_truck_data.txt", delimiter=",")
m = data.shape[0]
X_withoutOnes = data[:,[0]]
X = np.insert(X_withoutOnes, 0, 1, axis=1) # population
y = data[:,1] # profit 
theta = np.zeros(X.shape[1])

alpha = 0.0005
for i in range(0,2500):
    cost = 0
    # compute cost function
    cost = (1/(2*m) * np.matmul((np.matmul(X, theta) - y).transpose(), (np.matmul(X, theta) - y)))
    costs.append(cost)
    # compute gradient
    gradient = (1/m) * ( (X.transpose() @ X @ theta) - (X.transpose() @ y) )
    # update parameters
    theta = theta - (alpha*gradient)

# plot cost function
plt.plot(costs)
plt.title("Food Truck Cost Function")
plt.xlabel("theta 0")
plt.ylabel("theta 1")
plt.show()

# plot gradient descent line and data points
line = X @ theta 
plt.scatter(X_withoutOnes, y)
plt.title("Food Truck Linear Regression")
plt.xlabel("Population of city in 10000s")
plt.ylabel("Profit in $10000s")
plt.xlim([0,25])
plt.ylim([-10,25])
plt.plot(X_withoutOnes, line, color = 'orange')
plt.show()