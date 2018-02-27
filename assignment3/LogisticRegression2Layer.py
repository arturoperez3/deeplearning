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

# how do we incorporate a hidden layer? activation function?

### Logistic Gradient Descent  ###
def LGD(samples, X, y, alpha, n):
    costs = []
    m = X.shape[1] # number of samples
    f = X.shape[0] # number of features 

    
    # n = number of neurons
    # f = number of features 
    # m = number of samples 
    w = [np.random.rand(n,f), np.random.rand(1,n)] 
    w[0] = w[0] * 0.01
    w[1] = w[1] * 0.01
    b = [np.random.rand(n,1), np.random.rand(1,1)] 
    b[0] = b[0] * 0.01
    b[1] = b[1] * 0.01 
    z = [np.zeros((n,m)), np.zeros((1,m))]        
    a = [np.zeros((n,m)), np.zeros((1,m))]
    dZ = [np.zeros((n,m)), np.zeros((1,m))]
    dW = [np.zeros((f,n)), np.zeros((n,1))]
    dB = [np.zeros((n,1)), np.zeros((1,1))]



    for i in range(0,1):

        # forward propopgation 
        z[0] = (w[0] @ X) + b[0]
        a[0] = np.tanh(z[0])
        z[1] = (w[1] @ a[0]) + b[1]
        a[1] = 1 / (1 + np.exp(-z[1]))

        #compute cost
        cost = 0
        print(cost)
        cost = (-1/m) * np.sum(y * np.log(a[1]) + (1 - y) * np.log(1 - a[1]))
        print(cost)
        costs.append(cost)

        #backward propogation 
        dZ[1] = a[1] - y
        dW[1] = (1/m) * (dZ[1] @ a[0].transpose())
        dB[1] = (1/m) * np.sum(dZ[1], axis = 1).reshape(dB[1].shape[0], dB[1].shape[1])
        dZ[0] = np.multiply((w[1].transpose() @ dZ[1]),(1-(np.tanh(z[0])**2)))
        dW[0] = (1/m) * (dZ[0] @ X.transpose())
        dB[0] = (1/m) * np.sum(dZ[0], axis = 1).reshape(dB[0].shape[0], dB[0].shape[1])

        #update parameters 
        w[1] = w[1] - (alpha * dW[1])
        b[1] = b[1] - (alpha * dB[1])

        w[0] = w[0] - (alpha * dW[0])
        b[0] = b[0] - (alpha * dB[0])

    # plot cost function
    plt.plot(costs)
    plt.title("Student ID Cost Function (" + str(samples) + " samples)")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.show()

    return(w, b)

# y has to be transposed because of the way we read in the data (in our case, y = train_labels)
w,b = LGD(60000, train_images, train_labels.transpose(), .5, 10)

# compute the training and the test error
m = test_images.shape[1]
zTest = [np.zeros((5,m)), np.zeros((1,m))]
aTest = [np.zeros((5,m)), np.zeros((1,m))]


zTest[0] = (w[0] @ test_images) + b[0]
aTest[0] = np.tanh(zTest[0])
zTest[1] = (w[1] @ aTest[0]) + b[1]
aTest[1] = 1 / (1 + np.exp(-zTest[1]))


print (aTest[1].T)
print(aTest[1].shape)

zTestLabels = np.array([1 if a > 0.5 else 0 for a in aTest[1].T])

count = 0
misclassified = []
for i in range(1,10000):
    if zTestLabels[i] == test_labels[i]:
        count += 1
    else:
        misclassified.append(i)


path = "/Users/Arturo1/Desktop/Vanderbilt/2017-2018/Spring 2018/Deep Learning 3891/handwriting"   # the training set is stored in this directory
fname_test_images = os.path.join(path, 't10k-images-idx3-ubyte')  # the training set image file path
fname_train_labels = os.path.join(path, 't10k-labels-idx1-ubyte')  # the training set label file path
    
# open the label file and load it to the "train_labels"
with open(fname_train_labels, 'rb') as flbl:
    magic, num = struct.unpack(">II", flbl.read(8))
    testlabels = np.fromfile(flbl, dtype=np.uint8)

with open(fname_test_images, 'rb') as fimg:
    magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
    testimages = np.fromfile(fimg, dtype=np.uint8).reshape(len(testlabels), rows, cols)       

accuracy = (count/10000) * 100
error_rate = 100 - accuracy

print("accuracy for ID = " + str(ID) + " is : " + str(accuracy) + "%")
print("error rate for ID = " + str(ID) + " is : " + str(error_rate) + "%")



plt.imshow(testimages[misclassified[5]], cmap='gray')  # plot the image in "gray" colormap
plt.show()

plt.imshow(testimages[misclassified[6]], cmap='gray')  # plot the image in "gray" colormap
plt.show()

plt.imshow(testimages[misclassified[7]], cmap='gray')  # plot the image in "gray" colormap
plt.show()

plt.imshow(testimages[misclassified[8]], cmap='gray')  # plot the image in "gray" colormap
plt.show()

plt.imshow(testimages[misclassified[9]], cmap='gray')  # plot the image in "gray" colormap
plt.show()

#keep track of what you miss