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
def shuffleData(X,YF,YE, filenames):
    temp = list(zip(X, YF, YE, filenames))
    random.seed(5)
    random.shuffle(temp)
    X, YF, YE, filenames = zip(*temp)
    YF = np.asarray(YF)
    YF = np.reshape(YF,(YF.shape[0],1))
    YE = np.asarray(YE)
    YE = np.reshape(YE,(YE.shape[0],1))
    YF = convert_to_one_hot(YF,9).T
    YE = convert_to_one_hot(YE,9).T
    return np.stack(X, axis=0), YF, YE, filenames

# Read raster bands directly to Numpy arrays.
def tiffToArray():
    directory = './test_data/' # Change Directory for local machine
    YF = []      # Floating labels
    YE = []      # Emergent labels
    X = []       # Images
    filenames = []
    for filename in os.listdir(directory):
        with rasterio.open(directory+filename) as src:
            # Create Y instances from Tiff FileName
            filenames.append(filename)
            labels = filename.split("-")
            if int(labels[0]) > 9 or int(labels[1]) > 9: continue
            if int(labels[0]) <= 0 and int(labels[1]) <= 0: continue
            r, g, b, a = src.read()
            if g.shape != (150,150): continue
            YF.append(float(labels[0])/10)
            YE.append(float(labels[1])/10)

            # Create X_instance from Tiff
            np.reshape(r, (150,150,1))
            np.reshape(g, (150,150,1))
            np.reshape(b, (150,150,1))
            np.reshape(a, (150,150,1))
            X_instance = np.dstack((r,g,b,a))
            X.append(X_instance)
    return X, YF, YE, filenames

def loadData():
    X, YF, YE, filenames = tiffToArray()
    print("tiffToArray Complete")
    X, YF, YE = augmentData(X,YF,YE)
    print("augmentData Complete")
    X, YF, YE, filenames = shuffleData(X,YF,YE, filenames)
    print("shuffleData Complete")
    X_train, YF_train, YE_train, X_test, YF_test, YE_test = splitData(X,YF,YE)
    print("splitData Complete")
    return X_train, YF_train, YE_train, X_test, YF_test, YE_test, filenames
