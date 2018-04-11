# Arturo Perez 
# Professor Koutsoukos
# CS 3891 
# 3 April 2018 

import os, sys
import numpy as np
import struct
from scipy.special import expit
import matplotlib.pyplot as plt
from mnist import MNIST
import functools
from mlxtend.preprocessing import shuffle_arrays_unison

# the training set is stored in this directory
path = "/Users/Arturo1/Desktop/Vanderbilt/2017-2018/Spring 2018/Deep Learning 3891/handwriting"
training_size, test_size = 60000, 10000   
ID = 5 # last digit of my student ID
np.set_printoptions(threshold=5)
np.seterr(all='ignore')

def relabeling(label):
    for l in label:
        if (l[0] == ID):
            l[0] = 1
        else:
            l[0] = 0

### LOAD THE MNIST DATA ###
mndata = MNIST(path)
train_images, train_labels = mndata.load_training() # training data
test_images, test_labels = mndata.load_testing() # testing data
train_images = np.array(train_images[:training_size], dtype=float).transpose() / 255
train_labels = np.array(train_labels[:training_size], dtype=float)
test_images = np.array(test_images[:test_size], dtype=float).transpose() / 255
test_labels = np.array(test_labels[:test_size], dtype = float)
train_labels = np.reshape(train_labels, (training_size, 1))
test_labels = np.reshape(test_labels, (test_size, 1))

### PREPROCESSING ###
# preprocess the whole set of 60,000 samples
relabeling(train_labels)
relabeling(test_labels)

# subset of 10,000 samples
subsetX = train_images[:,10000:20000]
subsetY = train_labels[10000:20000,:]

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
def checkAccuracy(w, b , X, Y, neuralnetworkType) :
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
        for i in range(1, Y.size):
            if result[i] == Y[0, i]:
                count += 1
    else :       
        for i in range(1, Y.size):
            if result[0, i] == Y[0, i]:
                count += 1

    # calculate accuracy and error rates 
    accuracy = (count/Y.shape[1]) * 100
    error_rate = 100 - accuracy

    # print results 
    print("accuracy for " + neuralnetworkType + " is : " + str(accuracy) + "%")
    print("error rate for " + neuralnetworkType + " is : " + str(error_rate) + "%")

def getNextMiniBatch(X, Y, batchSize) :
    for i in range(0, X.shape[1], batchSize) :
        yield (X[: , i:i + batchSize] , Y[:, i:i + batchSize ])

def minibatchNeuralNetwork(p, q, p1, p2, alpha, batchSize, layer_list):
    f, m = p.shape
    
    # initialize weights, biases and parameters 
    w = functools.reduce(lambda acc, neurons: (acc[0] + [np.random.randn(neurons, 
    acc[1]) * np.sqrt(2.0/acc[1])], neurons), layer_list, ([], f))[0]
    b = [ np.random.randn(neurons, 1) * 0.01 for neurons in layer_list ]
    Ba0 = [ np.zeros((neurons, 1)) for neurons in layer_list ]
    Ba1 = [ np.zeros((neurons, 1)) for neurons in layer_list ]
    z = [None] * len(w)
    a = [None] * len(w)
    costs = []

    newLayerList = [f]
    for i in range (0, len(layer_list)) :
        newLayerList.append(layer_list[i])

    Wa = []
    for i in range (1, len(newLayerList)) :
        Wa.append([np.zeros((newLayerList[i], newLayerList[i-1])), 
        np.zeros((newLayerList[i], newLayerList[i-1]))])

    for i in range(0, 1):
        # api function used to shuffle 2 numpy arrays in unison
        p, q = shuffle_arrays_unison(arrays=[p.T, q.T], random_seed=3)
        p = p.T
        q = q.T

        # for each minibatch: forward prop, compute cost, backward prop, update parameters
        for X, Y in getNextMiniBatch(p, q, batchSize) :
            # forward propopgation
            z[0] = w[0] @ X + b[0]
            for i in range(len(w)-1):
                a[i] = ReLU(z[i])
                z[i+1] = w[i+1] @ a[i] + b[i+1]
            a[-1] = sigmoid(z[-1])

            # compute cost
            print(0)
            cost = - (Y @ np.log(a[-1].T) + (1 - Y) @ np.log(1 - a[-1].T))[0,0] / m
            print(cost)
            costs.append(cost)

            dZ = a[-1] - Y
            gradients = [] 
            # backward propogation 
            for k in range(len(z)-2, -1, -1):
                # Calculate dW and DB
                gradients.append([ (dZ @ a[k].T) / m, 
                    np.sum(dZ, axis=1).reshape(b[k+1].shape) / m ])
            
                dZ = (w[k+1].T @ dZ) * dReLU(z[k])

            # Calculate dW and dB for the first layer
            gradients.append([ (dZ @ X.T) / m, 
                np.sum(dZ, axis=1).reshape(b[0].shape) / m ])

            # flip gradients so order is from first to last layer
            gradients = gradients[::-1]


            # update parameters 
            # [0] = s
            # [1] = r
            for j in range(len(gradients)):
                Wa[j][0] = Wa[j][0] * p1 + (1-p1) * gradients[j][0]
                Wa[j][1] = np.multiply(p2, Wa[j][1]) + np.multiply((1-p2), (np.multiply(gradients[j][0], gradients[j][0])))
                Wa[j][0] = Wa[j][0] / (1-p1)
                Wa[j][1] = Wa[j][1] / (1-p2)
                Ba0[j] = np.multiply(p1, Ba0[j]) + np.multiply((1-p1), gradients[j][1])
                Ba1[j] = np.multiply(p2, Ba1[j]) + np.multiply((1-p2), (np.multiply(gradients[j][1], gradients[j][1])))
                Ba0[j] = Ba0[j] / (1-p1)
                Ba1[j] = Ba1[j] / (1-p2)
                w[j] = w[j] - (alpha*(np.true_divide(Wa[j][0], np.sqrt(Wa[j][1]))))
                b[j] = b[j] - (alpha*(np.true_divide(Ba0[j], np.sqrt(Ba1[j]))))
            
    # plot cost function
    plt.clf()
    plt.plot(costs)
    plt.title("Adam Neural Network Cost Function (" + str(m) + " samples)")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.show() 
    return(w, b)
 
layers = [20, 10]
w,b = minibatchNeuralNetwork(subsetX, subsetY.T, .9, .9, .1, 100, layers)
neuralType = "Adam Neural Network"
checkAccuracy(w, b, subsetX, subsetY.T, neuralType)