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

def importImages(directory):
    imgs = []
    os.chdir("./"+ directory)
    for filename in os.listdir("."):
        if filename.endswith(".png"):
            img = scipy.ndimage.imread(filename)
            img = img[:,0:703, :]
            imgs.append(img)
    return np.stack(imgs, axis=0)

def processImages(directory, ones):
    dataX = importImages(directory)
    if ones: dataY = np.ones((dataX.shape[0],1))
    else: dataY = np.zeros((dataX.shape[0],1))
    os.chdir("..")
    return dataX, dataY

def concatData(data1, data2):
    return np.concatenate((data1, data2), axis=0)

def loadData():
    # Import Training Data #
    floatX, floatY = processImages("floating/", True)
    nfloatX, nfloatY = processImages("nofloating/", False)
    X_train = concatData(floatX, nfloatX)
    Y_train = concatData(floatY, nfloatY)

    # Import Test Data #
    X_test = X_train[30:34,:,:,:] # Change so not hard coded
    Y_test = Y_train[30:34,:] # Change so not hard coded
    X_train = X_train[0:30,:,:,:]
    Y_train = Y_train[0:30,:]

    # Normalize
    X_train = X_train/255.
    X_test = X_test/255.
    return X_train, Y_train, X_test, Y_test
