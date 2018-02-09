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
ID = 6 # last digit of my student ID

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
