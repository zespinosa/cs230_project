import sys
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
import os
import tensorflow as tf
from PIL import Image
from scipy import ndimage
from tensorflow.python.framework import ops
from  import_tiff import loadData, tiffToArray
from create_heatmap import create_heatmap

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

def forward_propagation_expanded(X, parameters):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters
    Returns:
    Z6_F, Z6_E  -- the output of the last LINEAR unit
    """
    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    W4 = parameters['W4']
    W5 = parameters['W5']
    W6 = parameters['W6']
    W7 = parameters['W7']
    W8 = parameters['W8']

    A1 = tf.layers.conv2d(X, W1, [3,3], activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
    A2 = tf.layers.conv2d(A1, W2, [3,3], activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
    A3 = tf.layers.conv2d(A2, W3, [3,3], activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
    Z4 = tf.nn.max_pool(A3, ksize = [1,4,4,1], strides = [1,2,2,1], padding = 'VALID')
    A4 = tf.layers.conv2d(Z4, W4, [3,3], activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
    Z5 = tf.nn.max_pool(A4, ksize = [1,4,4,1], strides = [1,2,2,1], padding = 'VALID')

    # Branch into floating and emergent paths
    Z5_F = Z5
    Z5_E = Z5

    # CNN Branching
    # Floating
    A6 = tf.layers.conv2d(Z5_F, W5, [3,3], activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
    Z6 = tf.nn.max_pool(A6, ksize = [1,4,4,1], strides = [1,2,2,1], padding = 'VALID')
    A7 = tf.layers.conv2d(Z6, W6, [3,3], activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
    Z7 = tf.nn.max_pool(A7, ksize = [1,4,4,1], strides = [1,2,2,1], padding = 'VALID')
    Z7 = tf.contrib.layers.flatten(Z7)
    A8_F = tf.layers.dense(Z7, W7, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
    Z8_F = tf.layers.dense(A8_F, W8, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
    # Emergent
    A6 = tf.layers.conv2d(Z5_E, W5, [3,3], activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
    Z6 = tf.nn.max_pool(A6, ksize = [1,4,4,1], strides = [1,2,2,1], padding = 'VALID')
    A7 = tf.layers.conv2d(Z6, W6, [3,3], activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
    Z7 = tf.nn.max_pool(A7, ksize = [1,4,4,1], strides = [1,2,2,1], padding = 'VALID')
    Z7 = tf.contrib.layers.flatten(Z7)
    A8_E = tf.layers.dense(Z7, W7, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
    Z8_E = tf.layers.dense(A8_E, W8, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())

    return Z8_F, Z8_E

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters
    Returns:
    Z6_F, Z6_E  -- the output of the last LINEAR unit
    """
    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    W4 = parameters['W4']
    W5 = parameters['W5']
    W6 = parameters['W6']

    A1 = tf.layers.conv2d(X, W1, [3,3], activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
    Z1 = tf.nn.max_pool(A1, ksize = [1,4,4,1], strides = [1,2,2,1], padding = 'VALID')
    A2 = tf.layers.conv2d(Z1, W2, [3,3], activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
    Z2 = tf.nn.max_pool(A2, ksize = [1,4,4,1], strides = [1,2,2,1], padding = 'VALID')
    A3 = tf.layers.conv2d(Z2, W3, [3,3], activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
    Z4 = tf.nn.max_pool(A3, ksize = [1,4,4,1], strides = [1,2,2,1], padding = 'VALID')
    A4 = tf.layers.conv2d(Z4, W4, [3,3], activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
    Z5 = tf.nn.max_pool(A4, ksize = [1,4,4,1], strides = [1,2,2,1], padding = 'VALID')
    Z5 = tf.contrib.layers.flatten(Z5)

    # Branch into floating and emergent paths
    Z5_F = Z5
    Z5_E = Z5

    # Branch for floating -> tanh -> FC
    A5_F = tf.layers.dense(Z5_F, W5, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
    Z6_F = tf.layers.dense(A5_F, W6, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())

    # Branch for emergent -> tanh -> FC
    A5_E = tf.layers.dense(Z5_E, W5, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
    Z6_E = tf.layers.dense(A5_E, W6, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
    return Z6_F, Z6_E

def compute_cost(Z, Y):
    """
    Computes the cost
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    Returns:
    cost - Tensor of the cost function
    """

    # rank_1_weight = 1.0   # PARAMETER WE CHOOSE: try different values!
    # class_weights = tf.constant([[rank_1_weight, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    # weights = tf.reduce_sum(class_weights * Y, axis=1)
    # unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y)
    # weighted_losses = unweighted_losses * weights
    # cost = tf.reduce_mean(weighted_losses)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Z, labels = Y))

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
        minibatch_YF = YF_train[start:YF_train.shape[0],:]
        minibatch_YE = YE_train[start:start+minibatch_size,:]
        minibatches.append((minibatch_X,minibatch_YF, minibatch_YE))
    return minibatches


def composite_cost(predict, Y_labels):
    # Seperate Ones
    where_ones = tf.equal(Y_labels, 0)
    indices_ones = tf.where(where_ones)
    predict_ones = tf.gather(predict, indices_ones)
    Y_labels_ones = tf.boolean_mask(Y_labels, where_ones)
    # Seperate NonOnes
    where = tf.not_equal(Y_labels, 0)
    indices = tf.where(where)
    predict = tf.gather(predict,indices)
    Y_labels_others = tf.boolean_mask(Y_labels, where)
    return predict, Y_labels_others, predict_ones, Y_labels_ones

def model(X_train, YF_train, YE_train, X_test, YF_test, YE_test, filenames, generate, learning_rate = 0.009,
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

    # Forward propagation: Build the forward propagation in the tensorflow graph
    #parameters = { "W1": 16, "W2": 32, "W3": 64, "W4": 64, "W5": 128, "W6":128, "W7": 256, "W8": 9}
    #Z6_F, Z6_E = forward_propagation_expanded(X,parameters)
    parameters = { "W1": 16, "W2": 32, "W3": 64, "W4": 64, "W5": 256, "W6": 2}
    Z6_F, Z6_E = forward_propagation(X, parameters)

    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z6_F, YF) + compute_cost(Z6_E, YE)

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
            epoch_cost = 0.0
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, YF_train, YE_train, minibatch_size, seed)

            for minibatch in minibatches:
                (minibatch_X, minibatch_YF, minibatch_YE) = minibatch
                d = np.array(minibatch_X, dtype = np.float32)
                d = d - 127.5
                d /= 127.5
                _ , minibatch_cost, = sess.run([optimizer, cost], feed_dict={X: d, YF: minibatch_YF, YE: minibatch_YE})
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 1 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(epoch_cost)


        predict_F = tf.argmax(Z6_F, 1)
        predict_E = tf.argmax(Z6_E, 1)
        if generate:
            return predict_F, predict_E, X

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate Composite Predictions Floating:
        YF_labels = tf.argmax(YF, 1)
        predict_F, YF_labels_others, predict_F_ones, YF_labels_ones = composite_cost(predict_F, YF_labels)

        # Calculate Composite Predictions Emergent:
        YE_labels = tf.argmax(YE, 1)
        predict_E, YE_labels_others, predict_E_ones, YE_labels_ones = composite_cost(predict_E, YE_labels)

        # NonOnes
        correct_predictionF = tf.abs(tf.subtract(predict_F, YF_labels_others)) <= 1
        correct_predictionE = tf.abs(tf.subtract(predict_E, YE_labels_others)) <= 1
        # Ones
        correct_predictionF_ones = tf.equal(predict_F_ones, YF_labels_ones)
        correct_predictionE_ones = tf.equal(predict_E_ones, YE_labels_ones)

        # Calculate accuracy on the test set: NonOnes
        accuracyF = tf.reduce_mean(tf.cast(correct_predictionF, "float"))
        accuracyE = tf.reduce_mean(tf.cast(correct_predictionE, "float"))
        # Calculate accuracy on the test set: Ones
        accuracyF_ones = tf.reduce_mean(tf.cast(correct_predictionF_ones, "float"))
        accuracyE_ones = tf.reduce_mean(tf.cast(correct_predictionE_ones, "float"))

     
        train_accuracyYF = accuracyF.eval({X: X_train, YF: YF_train})
        test_accuracyYF = accuracyF.eval({X: X_test, YF: YF_test})
        train_accuracyYF_ones = accuracyF_ones.eval({X: X_train, YF: YF_train})
        test_accuracyYF_ones = accuracyF_ones.eval({X: X_test, YF: YF_test})

        train_accuracyYE = accuracyE.eval({X: X_train, YE: YE_train})
        test_accuracyYE = accuracyE.eval({X: X_test, YE: YE_test})
        train_accuracyYE_ones = accuracyE_ones.eval({X: X_train, YE: YE_train})
        test_accuracyYE_ones = accuracyE_ones.eval({X: X_test, YE: YE_test})

        print("Train Accuracy Floating:", train_accuracyYF)
        print("Test Accuracy Floating:", test_accuracyYF)
        print("Train Accuracy Floating Ones:", train_accuracyYF_ones)
        print("Test Accuracy Floating Ones:", test_accuracyYF_ones)
        print("Train Accuracy Emergent:", train_accuracyYE)
        print("Test Accuracy Emergent:", test_accuracyYE)
        print("Train Accuracy Emergent Ones:", train_accuracyYE_ones)
        print("Test Accuracy Emergent Ones:", test_accuracyYE_ones)

        return predict_F, predict_E, X


def main(map_directory=False):
    generate = bool(map_directory)
    X_train, YF_train, YE_train, X_test, YF_test, YE_test, filenames = loadData()
    Z6_F, Z6_E, X = model(X_train, YF_train, YE_train, X_test, YF_test, YE_test, filenames, generate, learning_rate = 0.0009,
              num_epochs=10, minibatch_size = 16, print_cost = True)
    if map_directory:
        filenames, X_map, _, _ = tiffToArray(map_directory) # X is a list
        X_map = np.stack(X_map, axis=0)
        (m, n_H0, n_W0, n_C0) = X_map.shape
        # Initialize all the variables globally
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # Run the initialization
            sess.run(init)
            predict_F = Z6_F.eval({X: X_map})
            predict_E = Z6_E.eval({X: X_map})
            create_heatmap(map_directory, filenames, predict_F, predict_E)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 2:
        map_directory = sys.argv[1]
        main(map_directory)
    
