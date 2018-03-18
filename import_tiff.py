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
import tensorflow as tf
from keras.utils.np_utils import to_categorical

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
    random.seed(5)
    random.shuffle(temp)
    X, YF, YE = zip(*temp)
    YF = np.asarray(YF)
    YF = np.reshape(YF,(YF.shape[0],1))
    YE = np.asarray(YE)
    YE = np.reshape(YE,(YE.shape[0],1))
    YF = to_categorical(YF,num_classes=9)
    YE = to_categorical(YE,num_classes=9)
    return np.stack(X, axis=0), YF, YE

# Read raster bands directly to Numpy arrays.
def tiffToArray(directory):
    generate = bool(directory)
    if not generate:
        directory = './train_data' # Change Directory for local machine
    YF = []      # Floating labels
    YE = []      # Emergent labels
    X = []       # Images
    filenames = []
    for filename in os.listdir(directory):
        with rasterio.open(directory+ '/' + filename) as src:
            if not generate:
                r, g, b, a = src.read()
                if g.shape != (150,150): 
                    continue

                # Create Y instances from Tiff FileName
                labels = filename.split("-")
                if int(labels[0]) > 9 or int(labels[1]) > 9: continue
                if int(labels[0]) <= 0 and int(labels[1]) <= 0: continue
                YF.append(int(labels[0])-1)
                YE.append(int(labels[1])-1)



            # Create X_instance from Tiff
            np.reshape(r, (150,150,1))
            np.reshape(g, (150,150,1))
            np.reshape(b, (150,150,1))
            np.reshape(a, (150,150,1))
            X_instance = np.dstack((r,g,b,a))
            X.append(X_instance)
            filenames.append(filename)
    return filenames, X, YF, YE

def loadData():
    filenames, X, YF, YE = tiffToArray(None)
    print("tiffToArray Complete")
    print('X len:', len(X))
    print('YF len:', len(YF))
    print('YE len:', len(YE))
    print('filenames:', len(filenames))

    X, YF, YE = augmentData(X,YF,YE)
    print("augmentData Complete")
    print('X len:', len(X))
    print('YF len:', len(YF))
    print('YE len:', len(YE))
    print('filenames:', len(filenames))

    X, YF, YE = shuffleData(X,YF,YE)
    print("shuffleData Complete")
    print('X len:', len(X))
    print('YF len:', len(YF))
    print('YE len:', len(YE))
    print('filenames:', len(filenames))

    X_train, YF_train, YE_train, X_test, YF_test, YE_test = splitData(X,YF,YE)
    print("splitData Complete")
    print('X_test len:', len(X_test))
    print('YF_test len:', len(YF_test))
    print('YE_test len:', len(YE_test))
    print('filenames:', len(filenames))
    return X_train, YF_train, YE_train, X_test, YF_test, YE_test, filenames
