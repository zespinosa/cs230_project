import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import os
import rasterio

# TO DO:
# 1. Expand CNN: Add a few more Layers
# 2. Tune Parameters!
# 3. Write Functions to Flip, Rotate, and Mirror Data
    # Note that we should only augment instances with positive labels
    # in order to speed up learning
# 4. Write Function to Normalize Data
    # Divide by 255, subtract off average, divide by variance
# 5. Figure out how to build Heat Maps

# At each stage of improvement we should record our previous results
# in order to document progress. This will help with final paper.

#Augments data by flipping image up/down, flipping image left/right, rotating 90 degrees once, and rotating 90 degrees twice.
def augmentData(X,YF,YE):
    num_augments = 7     #the number of augmentations we're doing.
    F = []
    for index, image_arr in enumerate(X):
        if (YF[index] != 1 or YE[index] != 1):      #if associated rank is not 1-1
            for i in range(0, num_augments):
                flipped_arr = image_arr
                if (i == 0): flipped_arr = np.flipud(flipped_arr)
                elif (i == 1): flipped_arr = np.fliplr(flipped_arr)
                elif (i == 2): flipped_arr = np.rot90(flipped_arr, k=1, axes=(0,1))
                elif(i == 3): flipped_arr = np.rot90(flipped_arr, k=2, axes=(0,1))
                elif(i == 4): flipped_arr = np.rot90(flipped_arr, k=3, axes=(0,1))
                if (i == 5):
                    flipped_arr = np.flipud(flipped_arr)
                    flipped_arr = np.rot90(flipped_arr, k=1, axes=(0,1))
                if (i == 6):
                    flipped_arr = np.flipud(flipped_arr)
                    flipped_arr = np.rot90(flipped_arr, k=3, axes=(0,1))

                F.append(flipped_arr)
                YF.append(YF[index])
                YE.append(YE[index])

                # im1 = Image.fromarray(flipped_arr)
                # im1.save(str(str(i)+'im1'+str(index)+'.tiff'))     #for testing
    #Append all flipped arrays to X
    for f in F:
        X.append(f)

    return X, YF, YE
