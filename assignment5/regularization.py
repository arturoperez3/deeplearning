# Arturo Perez 
# Professor Koutsoukos
# CS 3891 
# 15 March 2018 

import os
import numpy as np
import struct
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
import functools
from data_utils import plot_decision_boundary, load_moons

#   Part One 

# Generate 2-D train set
train_X, train_Y = load_moons()

# generate test set
test_X, test_Y = load_moons()

#   ReLU activation function
def ReLU (array):
    return np.maximum(0, array)
    
#   derivative of ReLU activation function
def dReLU(array):
    array[array < 0] = 0
    array[array > 0] = 1
    return array

#   Sigmoid activation function
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

#   compute the training and the test error
def checkAccuracy(w, b , X, neuralnetworkType) :
    # initialize our paramters for use in forward propogation 
    z = [None] * len(w)
    a = [None] * len(w)

    # forward propopgation 
    # we use a[-1] (the a from the very last layer) to get our prediction labels
    z[0] = w[0] @ X + b[0]
    for i in range(len(w)-1):
        a[i] = ReLU(z[i])
        z[i+1] = w[i+1] @ a[i] + b[i+1]
    a[-1] = sigmoid(z[-1])

    # get the prediction labels  
    result = np.where(a[-1] >= 0.5, 1.0, 0.0)

    # compare our prediction labels to the actual labels (test_Y in this case)
    dim = result.ndim
    count = 0
    if (dim == 1) :
        for i in range(1, test_Y.size):
            if result[i] == test_Y[0, i]:
                count += 1
    else :       
        for i in range(1, test_Y.size):
            if result[0, i] == test_Y[0, i]:
                count += 1

    # calculate accuracy and error rates 
    accuracy = (count/test_Y.shape[1]) * 100
    error_rate = 100 - accuracy

    # print results 
    print("accuracy for " + neuralnetworkType + " is : " + str(accuracy) + "%")
    print("error rate for " + neuralnetworkType + " is : " + str(error_rate) + "%")


def predictPlotLinearRegression (w, b , X) : 
    z = w @ X + b
    aTest = sigmoid(z)

    labels = np.array([1 if a > .5 else 0 for a in aTest.T])
    return labels

def predictPlotNeuralNetwork (w, b , X) : 
    z = [None] * len(w)
    a = [None] * len(w)

    # forward propopgation
    z[0] = w[0] @ X + b[0]
    for i in range(len(w)-1):
        a[i] = ReLU(z[i])
        z[i+1] = w[i+1] @ a[i] + b[i+1]
    a[-1] = sigmoid(z[-1])
    labels = np.where(a[-1] >= 0.5, 1.0, 0.0)
    return labels[0,:]










#   Part One

### Logistic Gradient Descent  ###
def LGD(X, y, alpha):
    costs = []
    m = X.shape[1]
    w = np.zeros((X.shape[0], 1))
    b = np.zeros((1, 1))

    for i in range(0,1000):

        z = (w.transpose() @ X) + b
        a = 1 / (1 + np.exp(-z))
        dZ = a - y.transpose()

        # compute cost function
        cost = 0
        cost = (-1/m) * np.sum(y.transpose() * np.log(a) + (1 - y.transpose()) * np.log(1 - a))
        costs.append(cost)

        # update parameters 
        dW = (1/m) * (X @ dZ.transpose())
        dB = (1/m) * np.sum(dZ)
        w = w - (alpha*dW)
        b = b - (alpha*dB)

    # plot cost function
    plt.clf()
    plt.plot(costs)
    plt.title("Logistic Regression Cost Function (" + str(m) + " samples)")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.show()

    return(w, b)

w, b = LGD(train_X, train_Y.T, .5)

neuralType = "Logistic Regression"
C = np.ravel(train_Y)
checkAccuracy(w.T, b.T, test_X, neuralType)
plot_decision_boundary(lambda x: predictPlotLinearRegression(w.T, b.T, x.T), train_X, C)



















#   Part Two

def unregularizedNeuralNetwork(X, y, alpha, layer_list):
    f, m = X.shape
    
    # initialize weights, biases and parameters 
    w = functools.reduce(lambda acc, neurons: (acc[0] + [np.random.randn(neurons, 
    acc[1]) * np.sqrt(2.0/acc[1])], neurons), layer_list, ([], f))[0]
    b = [ np.random.randn(neurons, 1) * 0.01 for neurons in layer_list ]
    z = [None] * len(w)
    a = [None] * len(w)
    costs = []

    for i in range(0, 1):

        print(X.shape)
        print(y.shape)
        # forward propopgation
        z[0] = w[0] @ X + b[0]
        for i in range(len(w)-1):
            a[i] = ReLU(z[i])
            z[i+1] = w[i+1] @ a[i] + b[i+1]
        a[-1] = sigmoid(z[-1])

        # compute cost
        cost = - (y @ np.log(a[-1].T) + (1 - y) @ np.log(1 - a[-1].T))[0,0] / m
        costs.append(cost)

        dZ = a[-1] - y
        gradients = [] 
        # backward propogation 
        for i in range(len(z)-2, -1, -1):
            # Calculate dW and DB
            gradients.append([ (dZ @ a[i].T) / m, 
                np.sum(dZ, axis=1).reshape(b[i+1].shape) / m ])
        
            dZ = (w[i+1].T @ dZ) * dReLU(z[i])

        # Calculate dW and dB for the first layer
        gradients.append([ (dZ @ X.T) / m, 
            np.sum(dZ, axis=1).reshape(b[0].shape) / m ])

        # flip gradients so order is from first to last layer
        gradients = gradients[::-1]

        # update parameters 
        for j in range(len(gradients)):
            w[j] = w[j] - (alpha * gradients[j][0])
            b[j] = b[j] - (alpha * gradients[j][1])

    # plot cost function
    plt.clf()
    plt.plot(costs)
    plt.title("Unregularized Neural Network Cost Function (" + str(m) + " samples)")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.show()
    return(w, b)
 
layers = [20, 10]
w,b = unregularizedNeuralNetwork(train_X, train_Y, .5, layers)
neuralType = "unregulated Neural Network"
checkAccuracy(w, b, test_X, neuralType)
plot_decision_boundary(lambda x: predictPlotNeuralNetwork(w, b, x.T), train_X, C)








#   Part Three 

def L2NeuralNetwork(X, y, alpha, lambd, layer_list):
    f, m = X.shape
    
    # initialize weights, biases and parameters 
    w = functools.reduce(lambda acc, neurons: (acc[0] + [np.random.randn(neurons, 
    acc[1]) * np.sqrt(2.0/acc[1])], neurons), layer_list, ([], f))[0]
    b = [ np.random.randn(neurons, 1) * 0.01 for neurons in layer_list ]
    z = [None] * len(w)
    a = [None] * len(w)
    costs = []

    for i in range(0, 500):

        # forward propopgation
        z[0] = w[0] @ X + b[0]
        for i in range(len(w)-1):
            a[i] = ReLU(z[i])
            z[i+1] = w[i+1] @ a[i] + b[i+1]
        a[-1] = sigmoid(z[-1])

        # compute L2 cost
        cost = - (y @ np.log(a[-1].T) + (1 - y) @ np.log(1 - a[-1].T))[0,0] / m
        L2 = 0
        for i in range(len(w)-1):
            L2 += np.sum(np.square(w[i]))
        L2 = L2 * (lambd/(2*m))
        costs.append(cost + L2)


        dZ = a[-1] - y
        gradients = [] 
        # backward propogation 
        for i in range(len(z)-2, -1, -1):
            # Calculate dW and DB
            gradients.append([ ((dZ @ a[i].T) / m) + (lambd/m)*w[i+1], 
                np.sum(dZ, axis=1).reshape(b[i+1].shape) / m ])
        
            dZ = (w[i+1].T @ dZ) * dReLU(z[i])

        # Calculate dW and dB for the first layer
        gradients.append([ ((dZ @ X.T) / m) + (lambd/m)*w[0], 
            np.sum(dZ, axis=1).reshape(b[0].shape) / m ])

        # flip gradients so order is from first to last layer
        gradients = gradients[::-1]

        # update parameters 
        for j in range(len(gradients)):
            w[j] = w[j] - (alpha * gradients[j][0])
            b[j] = b[j] - (alpha * gradients[j][1])

    # plot cost function
    plt.clf()
    plt.plot(costs)
    plt.title("L2 Neural Network Cost Function (" + str(m) + " samples)")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.show()
    return(w, b)

layers = [20, 10]
w,b = L2NeuralNetwork(train_X, train_Y, .5, .1, layers)
neuralType = "L2 Neural Network"
checkAccuracy(w, b, test_X, neuralType)
plot_decision_boundary(lambda x: predictPlotNeuralNetwork(w, b, x.T), train_X, C)










#   Part Four

def dropout(a, drop, keep_probability):
    drop = np.random.rand(a.shape[0], a.shape[1])
    drop = drop < keep_probability
    a = np.multiply(a, drop)
    a = a / keep_probability
    return (a, drop)

def dropoutNeuralNetwork(X, y, alpha, keep_probability, layer_list):
    f, m = X.shape
    
    # initialize weights, biases and parameters 
    w = functools.reduce(lambda acc, neurons: (acc[0] + [np.random.randn(neurons, 
    acc[1]) * np.sqrt(2.0/acc[1])], neurons), layer_list, ([], f))[0]
    b = [ np.random.randn(neurons, 1) * 0.01 for neurons in layer_list ]
    z = [None] * len(w)
    a = [None] * len(w)
    drop = [None] * len(w)
    costs = []

    for i in range(0, 500):

        # forward propopgation
        z[0] = w[0] @ X + b[0]
        for i in range(len(w)-1):
            a[i] = ReLU(z[i])
            a[i], drop[i] = dropout(a[i], drop[i], keep_probability)
            z[i+1] = w[i+1] @ a[i] + b[i+1]
        a[-1] = sigmoid(z[-1])

        # compute cost
        cost = np.nan_to_num(- (y @ np.log(a[-1].T) + (1 - y) @ np.log(1 - a[-1].T))[0,0] / m)
        costs.append(cost)

        gradients = []
        dZ = a[-1] - y
        # backward propogation 
        for i in range(len(z)-2, -1, -1):
            # Calculate dW and DB
            gradients.append([ (dZ @ a[i].T) / m, 
                np.sum(dZ, axis=1).reshape(b[i+1].shape) / m ])
        
            dZ = np.multiply((w[i+1].T @ dZ), drop[i]) * dReLU(z[i])
            dZ = dZ/keep_probability

        # Calculate dW and dB for the first layer
        gradients.append([ (dZ @ X.T) / m, 
            np.sum(dZ, axis=1).reshape(b[0].shape) / m ])

        # flip gradients so order is from first to last layer
        gradients = gradients[::-1]

        # update parameters 
        for j in range(len(gradients)):
            w[j] = w[j] - (alpha * gradients[j][0])
            b[j] = b[j] - (alpha * gradients[j][1])

    # plot cost function
    plt.clf()
    plt.plot(costs)
    plt.title("Unregularized Neural Network Cost Function (" + str(m) + " samples)")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.show()
    return(w, b)

w,b = dropoutNeuralNetwork(train_X, train_Y, .5, 0.9, [20, 10])
neuralType = "Dropout Neural Network"
checkAccuracy(w, b, test_X, neuralType)
plot_decision_boundary(lambda x: predictPlotNeuralNetwork(w, b, x.T), train_X, C)
