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
# in order to document progress. This will help with final paper
def augmentData(X,YF,YE):
    return X, YF, YE
