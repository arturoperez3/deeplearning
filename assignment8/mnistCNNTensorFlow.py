import tensorflow as tf
import os
import struct
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Training Parameters
learning_rate = 0.001
#num_steps = 500
num_epochs = 10
batch_size = 64

# Network Parameters
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.05 # Dropout, probability to drop a unit


def data_input():
    # Import data
    path = os.getcwd()
    # Train data
    filePath = '/Users/Arturo1/Desktop/Vanderbilt/2017-2018/Spring 2018/Deep Learning 3891/handwriting'   # the training set is stored in this directory
    fname_train_images = os.path.join(filePath, 'train-images-idx3-ubyte')  # the training set image file path
    fname_train_labels = os.path.join(filePath, 'train-labels-idx1-ubyte')  # the training set label file path

    # open the label file and load it to the "train_labels"
    with open(fname_train_labels, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        train_labels = np.fromfile(flbl, dtype=np.uint8)

    # open the image file and load it to the "train_images"
    with open(fname_train_images, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        train_images = np.fromfile(fimg, dtype=np.uint8).reshape(len(train_labels), rows, cols)


    # Test data
    fname_test_images = os.path.join(filePath, 't10k-images-idx3-ubyte')  # the test set image file path
    fname_test_labels = os.path.join(filePath, 't10k-labels-idx1-ubyte')  # the test set label file path

    # open the label file and load it to the "test_labels"
    with open(fname_test_labels, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        test_labels = np.fromfile(flbl, dtype=np.uint8)

    # open the image file and load it to the "train_images"
    with open(fname_test_images, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        test_images = np.fromfile(fimg, dtype=np.uint8).reshape(len(test_labels), rows, cols)

    # MNIST data input is a 2-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    train_images = train_images.reshape((-1, 28, 28, 1))
    test_images = test_images.reshape((-1, 28, 28, 1))

    train_set_x = train_images/255.
    test_set_x = test_images/255.

    return train_set_x.astype(np.float32),train_labels,test_set_x.astype(np.float32),test_labels


# Create the neural network
def conv_net(x_dict, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        x = x_dict['images']

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 64 filters and a kernel size of 3
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)

    return out


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train = conv_net(features, num_classes, dropout, reuse=False,
                            is_training=True)
    logits_test = conv_net(features, num_classes, dropout, reuse=True,
                           is_training=False)

    # Predictions
    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

        # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs

# Build the Estimator
print("start program")
model = tf.estimator.Estimator(model_fn)

print("get the data")
train_images,train_labels,test_images,test_labels=data_input()

print("Before training")
# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': train_images}, y=train_labels,
    batch_size=batch_size, num_epochs=num_epochs, shuffle=True)
# Train the Model
model.train(input_fn, steps=None)

print("Before Testing")
# Evaluate the Model
# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': test_images}, y=test_labels,
    batch_size=batch_size, shuffle=False)
# Use the Estimator 'evaluate' method
e = model.evaluate(input_fn)

print("Testing Accuracy:", e['accuracy'])
