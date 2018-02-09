# Arturo Perez 
# Professor Koutsoukos
# CS 3891 
# 6 February 2018 

import os
import numpy as np
import struct
import matplotlib.pyplot as plt
from mnist import MNIST

print("hello")
# the training set is stored in this directory
path = "/Users/Arturo1/Desktop/Vanderbilt/2017-2018/Spring 2018/Deep Learning 3891/handwriting"
training_size, test_size = 60000, 10000   
ID = 9 # last digit of my student ID

def relabeling(label):
    for l in label:
        if (l[0] == ID):
            l[0] = 1
        else:
            l[0] = 0

### LOAD THE MNIST DATA ###
mndata = MNIST(path)
mnist_images, mnist_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()
mnist_images = np.array(mnist_images[:training_size], dtype=float).transpose() / 255
mnist_labels = np.array(mnist_labels[:training_size], dtype=float)
test_images = np.array(test_images[:test_size], dtype=float).transpose() / 255
test_labels = np.array(test_labels[:test_size], dtype = float)
mnist_labels = np.reshape(mnist_labels, (training_size, 1))
test_labels = np.reshape(test_labels, (test_size, 1))

### PREPROCESSING ###
# preprocess the whole set of 60,000 samples
relabeling(mnist_labels)
relabeling(test_labels)


### creating smaller data sets of differing size and differing data ### 
# subset1 of 10 samples
subsetOneImages_processed = mnist_images[:,:10]
ID_labels1 = mnist_labels[:10,:]
#print(subsetOneImages_processed.shape[0])
#print(ID_labels1.shape[0])

# subset2 of 75 samples
subsetTwoImages_processed = mnist_images[25:100]
ID_labels2 = mnist_labels[25:100]

# subset 3 of 1,000 samples
subsetThreeImages_processed = mnist_images[200:1200]
ID_labels3 = mnist_labels[200:1200]

#subset 4 of 10,000 samples
subsetFourImages_processed = mnist_images[10000:20000]
ID_labels4 = mnist_labels[10000:20000]



### Logistic Gradient Descent  ###
def LGD(samples, X, y, alpha):
    costs = []
    m = X.shape[1]
    w = np.zeros((X.shape[0], 1))
    b = np.zeros((1, 1))

    for i in range(0,1000):

        z = (w.transpose() @ X) + b
        a = 1 / (1 + np.exp(-z))
        dZ = a - y.transpose()

        #compute cost function
        cost = 0
        print(cost)
        cost = (-1/m) * np.sum(y.transpose() * np.log(a) + (1 - y.transpose()) * np.log(1 - a))
        print(cost)
        costs.append(cost)

        #update parameters 
        dW = (1/m) * (X @ dZ.transpose())
        dB = (1/m) * np.sum(dZ)
        w = w - (alpha*dW)
        b = b - (alpha*dB)

    # plot cost function
    plt.plot(costs)
    plt.title("Student ID Cost Function (" + str(samples) + " samples)")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.show()

    return(w, b)





## investigate the impact of the training data size and the learning rate alpha ###

# LGD(10, subsetOneImages_processed, ID_labels1, .001) #10 samples, alpha = .001
# LGD(10, subsetOneImages_processed, ID_labels1, .01) #10 samples, alpha = .01
# LGD(10, subsetOneImages_processed, ID_labels1, .1) #10 samples, alpha = .1

# LGD(75, subsetTwoImages_processed, ID_labels2, .001) #75 samples, alpha = .001
# LGD(75, subsetTwoImages_processed, ID_labels2, .01) #75 samples, alpha = .01
# LGD(75, subsetTwoImages_processed, ID_labels2, .1) #75 samples, alpha = .1

# LGD(1000, subsetThreeImages_processed, ID_labels3, .001) #1000 samples, alpha = .001
# w,b = LGD(1000, subsetThreeImages_processed, ID_labels3, .01) #1000 samples, alpha = .01
# LGD(1000, subsetThreeImages_processed, ID_labels3, .1) #1000 samples, alpha = .1

# LGD(10000, subsetFourImages_processed, ID_labels4, .001) #10000 samples, alpha = .001
# LGD(10000, subsetFourImages_processed, ID_labels4, .01) #10000 samples, alpha = .01

# print("we got here")
# # train with whole 60,000 samples and the learning rate alpha = .01
w, b = LGD(60000, mnist_images, mnist_labels, .01)

#compute the training and the test error
zTest = (w.transpose() @ test_images) + b

zTestLabels = np.array([1 if z[0] > 0 else 0 for z in zTest.T])

count = 0
for i in range(1,10000):
    if zTestLabels[i] == test_labels[i]:
        count += 1

accuracy = (count/10000) * 100
error_rate = 100 - accuracy

print("accuracy for ID = " + str(ID) + " is : " + str(accuracy) + "%")
print("error rate for ID = " + str(ID) + " is : " + str(error_rate) + "%")