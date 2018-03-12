import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import os
import rasterio
from augment_data import augmentData
import random

# Splits Image Set into a Train and Test set
# Test set in 5% of full Image set
# Note: Add Dev set later
def splitData(X,YF,YE):
    p = int(math.ceil(.05*X.shape[0])) # Test Set = 5% Full Image Set
    X_test = X[0:p,:,:,:]
    YF_test = YF[0:p,:]
    YE_test = YE[0:p,:]
    X_train = X[p:,:,:,:]
    YF_train = YF[p:,:]
    YE_train = YE[p:,:]
    return X_train, YF_train, YE_train, X_test, YF_test, YE_test

# Shuffles training data:
# Necessary for minibatches in future
# Reduces skew in learning
def shuffleData(X,YF,YE):
    temp = list(zip(X, YF, YE))
    random.shuffle(temp)
    X, YF, YE = zip(*temp)
    YF = np.asarray(YF)
    YF = np.reshape(YF,(YF.shape[0],1))
    YE = np.asarray(YE)
    YE = np.reshape(YE,(YE.shape[0],1))
    return np.stack(X, axis=0), YF, YE

# Read raster bands directly to Numpy arrays.
def tiffToArray():
    directory = './tiff_data/' # Change Directory for local machine
    YF = []      # Floating labels
    YE = []      # Emergent labels
    X = []       # Images
    for filename in os.listdir(directory):
        with rasterio.open(directory+filename) as src:
            # Create X_instance from Tiff
            r, g, b, a = src.read()
            np.reshape(g, (150,150,1))
            np.reshape(b, (150,150,1))
            np.reshape(a, (150,150,1))
            X_instance = np.dstack((g,b,a))
            X.append(X_instance)

            # Create Y instances from Tiff FileName
            labels = filename.split("-")
            YF.append(int(labels[0]))
            YE.append(int(labels[1]))
    return X, YF, YE

def loadData():
    X, YF, YE = tiffToArray()
    X, YF, YE = augmentData(X,YF,YE)
    X, YF, YE = shuffleData(X,YF,YE)
    X_train, YF_train, YE_train, X_test, YF_test, YE_test = splitData(X,YF,YE)
    return X_train, YF_train, YE_train, X_test, YF_test, YE_test
