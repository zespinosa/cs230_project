import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
import os
from  import_tiff import loadData
#from  import_png import loadData

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes
    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """
    X = tf.placeholder(shape=(None,n_H0,n_W0,n_C0), dtype=tf.float32, name="X")
    YF = tf.placeholder(shape=(None, n_y), dtype= tf.float32, name="YF")
    YE = tf.placeholder(shape=(None, n_y), dtype= tf.float32, name="YE")
    return X, YF, YE

def initialize_parameters():
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [4, 4, 3, 8]
                        W2 : [2, 2, 8, 16]
    Returns:
    parameters -- a dictionary of tensors containing W1, W2, W3, W4, W5, W6
    """
    # CNN Floating and Emergent
    W1 = tf.get_variable("W1", [8,8,4,8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [4,4,8,16], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    # CNN Floating
    W3 = tf.get_variable("W3", [4,4,16,32], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W4 = tf.get_variable("W4", [2,2,32,64], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    # CNN Emergent
    W5 = tf.get_variable("W5", [4,4,16,32], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W6 = tf.get_variable("W6", [2,2,32,64], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    parameters = {"W1": W1, "W2": W2, "W3": W3, "W4": W4, "W5": W5, "W6": W6}
    return parameters

def floatCNN(X, parameters):
    #P2 = tf.contrib.layers.flatten(X)
    #Z3 = tf.contrib.layers.fully_connected(P2, 9, activation_fn=None)
    #return Z3
    W3 = parameters['W3']
    W4 = parameters['W4']
    # CONV2D: stride of 1, padding 'SAME'
    Z1 = tf.nn.conv2d(X,W3, strides = [1,1,1,1], padding = 'SAME')
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 8x8, sride 8, padding 'SAME'
    P1 = tf.nn.max_pool(A1, ksize = [1,8,8,1], strides = [1,8,8,1], padding = 'SAME')
    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1,W4, strides = [1,1,1,1], padding = 'SAME')
    # RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2_0 = tf.nn.max_pool(A2, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')
    P2 = tf.contrib.layers.flatten(P2_0)
    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None"
    Z3 = tf.contrib.layers.fully_connected(P2, 9, activation_fn=None)

    return Z3

def emergentCNN(X, parameters):
    #P2 = tf.contrib.layers.flatten(X)
    #Z3 = tf.contrib.layers.fully_connected(P2, 9, activation_fn=None)
    #return Z3
    W5 = parameters['W5']
    W6 = parameters['W6']

    # CONV2D: stride of 1, padding 'SAME'
    Z1 = tf.nn.conv2d(X,W5, strides = [1,1,1,1], padding = 'SAME')
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 8x8, sride 8, padding 'SAME'
    P1 = tf.nn.max_pool(A1, ksize = [1,8,8,1], strides = [1,8,8,1], padding = 'SAME')
    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1,W6, strides = [1,1,1,1], padding = 'SAME')
    # RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2_0 = tf.nn.max_pool(A2, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')

    P2 = tf.contrib.layers.flatten(P2_0)
    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None"
    Z3 = tf.contrib.layers.fully_connected(P2, 9, activation_fn=None)

    return Z3

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters
    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    W2 = parameters['W2']

    # CONV2D: stride of 1, padding 'SAME'
    Z1 = tf.nn.conv2d(X,W1, strides = [1,1,1,1], padding = 'SAME')
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 8x8, sride 8, padding 'SAME'
    P1 = tf.nn.max_pool(A1, ksize = [1,8,8,1], strides = [1,8,8,1], padding = 'SAME')
    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1,W2, strides = [1,1,1,1], padding = 'SAME')
    # RELU
    A2_0 = tf.nn.relu(Z2)
    A2_1 = A2_0
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    #P2_0 = tf.nn.max_pool(A2, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')
    #P2_1 = P2_0

    Z3_0 = floatCNN(A2_0, parameters)
    Z3_1 = emergentCNN(A2_1, parameters)
    return Z3_1, Z3_0,

def compute_cost(Z3, Y):
    """
    Computes the cost
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    Returns:
    cost - Tensor of the cost function
    """
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))
    #cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Z3, labels = Y))
    return cost

def random_mini_batches(X_train, YF_train, YE_train, minibatch_size, seed):
    minibatches = []
    counter = X_train.shape[0]
    start = 0
    while counter >= minibatch_size:
        minibatch_X = X_train[start:start+minibatch_size,:,:,:]
        minibatch_YF = YF_train[start:start+minibatch_size,:]
        minibatch_YE = YE_train[start:start+minibatch_size,:]
        minibatches.append((minibatch_X,minibatch_YF, minibatch_YE))
        counter -= minibatch_size
        start += minibatch_size
    if counter > 0:
        minibatch_X = X_train[start:X_train.shape[0],:,:,:]
        minibatch_YF = YF_train[start:Y_train.shape[0],:]
        minibatch_YE = YE_train[start:start+minibatch_size,:]
        minibatches.append((minibatch_X,minibatch_YF, minibatch_YE))
    return minibatches


def model(X_train, YF_train, YE_train, X_test, YF_test ,YE_test, filenames, learning_rate = 0.009,
          num_epochs = 30, minibatch_size = 1, print_cost = True):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Arguments:
    X_train -- training set, of shape (None, 64, 64, 3)
    Y_train -- test set, of shape (None, n_y = 6)
    X_test -- training set, of shape (None, 64, 64, 3)
    Y_test -- test set, of shape (None, n_y = 6)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep results consistent (tensorflow seed)
    seed = 3                                          # to keep results consistent (numpy seed)
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = YF_train.shape[1]
    costs = []                                        # To keep track of the cost

    # Create Placeholders of the correct shape
    X, YF, YE = create_placeholders(n_H0, n_W0, n_C0,n_y)
    # Initialize parameters
    parameters = initialize_parameters()
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3_F, Z3_E = forward_propagation(X, parameters)
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3_F, YF) + compute_cost(Z3_E, YE)
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    # Initialize all the variables globally
    init = tf.global_variables_initializer()
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):
            epoch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, YF_train, YE_train, minibatch_size, seed)

            for minibatch in minibatches:
                (minibatch_X, minibatch_YF, minibatch_YE) = minibatch
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, YF: minibatch_YF, YE: minibatch_YE})
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(epoch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        predict_F = tf.argmax(Z3_F, 1)
        predict_E = tf.argmax(Z3_E, 1)
        correct_predictionF = tf.abs(tf.subtract(predict_F, tf.argmax(YF, 1))) <= 3
        correct_predictionE = tf.abs(tf.subtract(predict_E, tf.argmax(YE, 1))) <= 3

        # Calculate accuracy on the test set
        accuracyF = tf.reduce_mean(tf.cast(correct_predictionF, "float"))
        accuracyE = tf.reduce_mean(tf.cast(correct_predictionE, "float"))
        print(accuracyF, accuracyE)
        train_accuracyYF = accuracyF.eval({X: X_train, YF: YF_train})
        test_accuracyYF = accuracyF.eval({X: X_test, YF: YF_test})
        train_accuracyYE = accuracyE.eval({X: X_train, YE: YE_train})
        test_accuracyYE = accuracyE.eval({X: X_test, YE: YE_test})
        print("Train Accuracy Floating:", train_accuracyYF)
        print("Test Accuracy Floating:", test_accuracyYF)
        print("Train Accuracy Emergent:", train_accuracyYE)
        print("Test Accuracy Emergent:", test_accuracyYE)

        return train_accuracyYF, test_accuracyYE, parameters

def main():
    #X_train, Y_train, X_test, Y_test = loadData()
    X_train, YF_train, YE_train, X_test, YF_test, YE_test, filenames = loadData()
    train_accuracy, test_accuracy, parameters = model(X_train, YF_train, YE_train, X_test, YF_test, YE_test, filenames, learning_rate = 0.009,
              num_epochs = 20, minibatch_size = 1, print_cost = True)

main()
