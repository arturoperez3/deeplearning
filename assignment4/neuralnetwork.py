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


logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
# the training set is stored in this directory
path = "/Users/Arturo1/Desktop/Vanderbilt/2017-2018/Spring 2018/Deep Learning 3891/handwriting"
training_size, test_size = 60000, 10000   
ID = 1 # last digit of my student ID
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
    return np.maximum(array,0)
    
def dReLU(array):
    return (array >= 0).astype(np.float64)

def LGD(samples, X, y, alpha, layer_list):
    l = len(layer_list)
    m = X.shape[1]
    costs = []
    w  = [0] * (l)
    b  = [0] * (l)
    z  = [0] * (l)
    a  = [0] * (l)
    dZ = [0] * (l)
    dW = [0] * (l)
    dB = [0] * (l)
    dA = [0] * (l)
    
    for i in range(1, l):
        w[i] = np.random.randn(layer_list[i], layer_list[i - 1]) / 10000
        b[i] = np.random.randn(layer_list[i], 1) / 10000
        z[i] = np.zeros((layer_list[i], m))
        a[i] = np.zeros((layer_list[i], m))
        dZ[i] = np.zeros((layer_list[i], m))
        dW[i] = np.zeros((layer_list[i], layer_list[i - 1]))
        dB[i] = np.zeros((layer_list[i], 1))
        dA[i] = np.zeros((layer_list[i], m))
    dA[0] = np.zeros((layer_list[0], m))
    
    a[0] = X 

    for i in range(0, 500):
        # forward propopgation
        for j in range (1, l):
            z[j] = (w[j] @ a[j-1]) + b[j]
            a[j] = ReLU(z[j])
   
        a[l-1] = expit(z[l-1])

        #compute cost
        cost = 0
        cost = (np.nan_to_num(
            -y @ np.log(a[l-1]) - (1 - y) @ np.log(1 - a[l-1]))[0, 0] / m)
        costs.append(cost)

        #backward propogation 
        dA[-1] = - (y.T @ np.log(a[-1].T) + (1 - y.T) @ np.log(1 - a[-1].T))
        for k in range(l-1, 0, -1):
            dZ[k] = dA[k] * dReLU(z[k])
            dW[k] = dZ[k] @ a[k-1].transpose() / m      
            dB[k] = dZ[k].sum(axis = 1).reshape(dB[k].shape[0], dB[k].shape[1]) / m
            dA[k - 1] = w[k].T @ dZ[k]

        #update parameters 
        for j in range (1, l):
            w[j] = w[j] - (alpha * dW[j])
            b[j] = b[j] - (alpha * dB[j])
    # plot cost function
    plt.plot(costs)
    plt.title("Student ID Cost Function (" + str(samples) + " samples)")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.show()
    return(w, b)
 
layers = [784, 25, 15, 5, 1]
w,b = LGD(60000, subsetTwoImages, ID_labels2, .5, layers)

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

# zTest = np.array([1 if c[0] > 0.5 else 0 for c in a])

# count = 0
# misclassified = []
# for i in range(1,10000):
#     if zTest[0,i] == test_labels[0,i]:
#         count += 1
#     else:
#         misclassified.append(i)

accuracy = (count/test_labels.shape[0]) * 100
error_rate = 100 - accuracy

print("accuracy for ID = " + str(ID) + " is : " + str(accuracy) + "%")
print("error rate for ID = " + str(ID) + " is : " + str(error_rate) + "%")
