#!/usr/bin/python

# Arturo Perez 
# Professor Koutsoukos
# CS 3891 
# 25 January 2018 

import os
import numpy as np
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# initalize variables 
costs = []
data = np.genfromtxt("housing_price_data.txt", delimiter=",")
m = data.shape[0]
X_column1 = data[:,[0]] # size of the house 
X_column2 = data[:,[1]] # number of bedrooms
X_withoutOnes = np.concatenate((X_column1,X_column2), axis=1)
X = np.insert(X_withoutOnes, 0, 1, axis=1)
y = data[:,2] # price of the house 
theta = np.zeros(X.shape[1])

# perform feature scaling 
x_averages = np.mean(X, axis=0)
x_ranges = np.ptp(X, axis=0)
X[:,1] = (X[:,1] - x_averages[1]) / x_ranges[1]
X[:,2] = (X[:,2] - x_averages[2]) / x_ranges[2]

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

#plot cost function
plt.plot(costs)
plt.title("Housing Prices Cost Function")
plt.xlabel("theta 0")
plt.ylabel("theta 1")
plt.show()

#3d graph
fig = plt.figure()
ax = plt.axes(projection='3d')
zline = theta[0]
yline = theta[2] * X[:,2]
xline = theta[1] * X[:,1]
line = zline + yline + xline
plt.title("Housing Prices Multivariate Linear Regression")
plt.xlabel("theta 0")
plt.ylabel("theta 1")
ax.plot3D(line, line) 
plt.show()
