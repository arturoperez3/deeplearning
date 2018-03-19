# Arturo Perez 
# Professor Koutsoukos
# CS 3891 
# 15 February 2018 

import os, sys
import numpy as np
import struct
import logging
from scipy.special import expit
import matplotlib.pyplot as plt
from mnist import MNIST
import functools


logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
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

### creating smaller data sets of differing size and differing data ### 
# subset1 of 10 samples
subsetOneImages = train_images[:,:10]
ID_labels1 = train_labels[:10,:]

# subset2 of 75 samples
subsetTwoImages = train_images[:,25:100]
ID_labels2 = train_labels[25:100,:]

# subset 3 of 1,000 samples
subsetThreeImages = train_images[:,200:1200]
ID_labels3 = train_labels[200:1200,:]

# subset 4 of 10,000 samples
subsetFourImages = train_images[:,10000:20000]
ID_labels4 = train_labels[10000:20000,:]

def ReLU (array):
    return np.maximum(0, array)
    
def dReLU(array):
    array[array < 0] = 0
    array[array > 0] = 1
    return array

    #   Sigmoid function
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def sigmoid_prime(Z):
    sig = sigmoid(Z)
    return sig @ (1 - sig)

def LGD(X, y, alpha, layer_list):
    f, m = X.shape
    
    # initialize weights and biases
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

        # compute cost
        cost = - (y @ np.log(a[-1]) + (1 - y) @ np.log(1 - a[-1]))[0,0] / m
        print(cost)
        costs.append(cost)

        dZ = a[-1] - y.T
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
    plt.plot(costs)
    plt.title("Unregularized Neural Network Cost Function (" + str(m) + " samples)")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.show()
    return(w, b)
 
layers = [784, 25, 15, 5, 1]
w,b = LGD(subsetFourImages, ID_labels4, .5, layers)

a = test_images
z = None

# forward propopgation 
for j in range (1, len(layers)):
    z = w[j] @ a + b[j]
    a = ReLU(z)
a = expit(z)

misclassified = []
count = 0
result = np.where(a >= 0.5, 1.0, 0.0)
for i in range(1,10000):
    if result[0, i] == test_labels[i,0]:
        count += 1
    else:
        misclassified.append(i)

accuracy = (count/test_labels.shape[0]) * 100
error_rate = 100 - accuracy

print("accuracy for ID = " + str(ID) + " is : " + str(accuracy) + "%")
print("error rate for ID = " + str(ID) + " is : " + str(error_rate) + "%")
