# Arturo Perez 
# Professor Koutsoukos
# CS 3891 
# 15 February 2018 

import os
import numpy as np
import struct
import matplotlib.pyplot as plt
from mnist import MNIST

print("hello")
# the training set is stored in this directory
path = "/Users/Arturo1/Desktop/Vanderbilt/2017-2018/Spring 2018/Deep Learning 3891/handwriting"
training_size, test_size = 60000, 10000   
ID = 6 # last digit of my student ID
np.set_printoptions(threshold=np.inf)

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


def LGD(samples, X, y, alpha, n0, n1, n2, n3, n4, n5, l):
    costs = []
    m = X.shape[1] # number of samples
    f = X.shape[0] # number of features 
    w = []
    b = []
    z = []
    a = []
    dZ = []
    dW = []
    dB = []
    n = [n0, n1, n2, n3, n4, n5]

    for i in range(1, l):
        w = w.append(np.random.rand(n[i], n[i-1]))
        b = b.append(np.random.rand(n[i], 1))
        z = z.append(np.random.rand(n[i], m))
        a = a.append(np.random.rand(n[i], m))
        dZ = dZ.append(np.random.rand(n[i], m))
        dW = dW.append(np.random.rand(n[i-1], n[i]))
        dB = dB.append(np.random.rand(n[i], 1))



    for i in range(0,500):

        # forward propopgation 
        z[0] = (w[0] @ X) + b[0]

        for j in range (1, l):
            a[j-1] = np.array([a if a > 0 else 0 for a in z[j-1]])
            z[j] = (w[j] @ a[j-1]) + b[1]

        a[l-1] = 1 / (1 + np.exp(-z[l-1]))



        #compute cost
        cost = 0
        print(cost)
        cost = (-1/m) * np.sum(y * np.log(a[l-1]) + (1 - y) * np.log(1 - a[l-1]))
        print(cost)
        costs.append(cost)

        #backward propogation 
        dZ[l-1] = a[l-1] - y

        for k in range(l-1, 0, -1):
            dW[k] = (1/m) * (dZ[k] @ a[k-1].transpose())
            dB[k] = (1/m) * np.sum(dZ[k], axis = 1).reshape(dB[k].shape[0], dB[k].shape[1])
            dZ[k-1] = np.multiply((w[k].transpose() @ dZ[k]),(1-(np.tanh(z[k-1])**2)))
        
        dW[0] = (1/m) * (dZ[0] @ X.transpose())
        dB[0] = (1/m) * np.sum(dZ[0], axis = 1).reshape(dB[0].shape[0], dB[0].shape[1])

        #update parameters 
        for a in range (l-1, 0, -1):
            w[a] = w[a] - (alpha * dW[a])
            b[a] = b[a] - (alpha * dB[a])

        w[0] = w[0] - (alpha * dW[0])
        b[0] = b[0] - (alpha * dB[0])

    # plot cost function
    plt.plot(costs)
    plt.title("Student ID Cost Function (" + str(samples) + " samples)")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.show()

    return(w, b)

w,b = LGD(60000, train_images, train_labels.transpose(), .5, 20, 10, 5, 1, 0, 0, 4)

l = 4
n = [20, 10, 5, 0]
for i in range(1, l):
    ztest = z.append(np.random.rand(n[i], m))
    atest = a.append(np.random.rand(n[i], m))


    # forward propopgation 
    ztest[0] = (w[0] @ X) + b[0]
    for j in range (1, l):
        atest[j-1] = np.array([a if a > 0 else 0 for a in ztest[j-1]])
        ztest[j] = (w[j] @ atest[j-1]) + b[1]

    atest[l-1] = 1 / (1 + np.exp(-ztest[l-1]))

zTestLabels = np.array([1 if a > 0.5 else 0 for a in atest[l-1].T])

count = 0
misclassified = []
for i in range(1,10000):
    if zTestLabels[i] == test_labels[i]:
        count += 1
    else:
        misclassified.append(i)

accuracy = (count/10000) * 100
error_rate = 100 - accuracy

print("accuracy for ID = " + str(ID) + " is : " + str(accuracy) + "%")
print("error rate for ID = " + str(ID) + " is : " + str(error_rate) + "%")
