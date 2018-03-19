import rasterio
import sys
import numpy as np
import os

from subprocess import call

# in_directory = './test_data/'
# out_directory = './heatmap/'

PIXEL_DELTA = 10
THRESHOLD = 5


def create_heatmap(map_directory, filenames, predict_F, predict_E):
  heatmap_directory = 'heatmap_' + map_directory
  makedir_command = 'mkdir ' + heatmap_directory
  call(makedir_command.split())

  F_values = [0 for i in range(9)]
  E_values = [0 for i in range(9)]

  for i in range(len(filenames)):
    filename = filenames[i]
    F = predict_F[i]
    E = predict_E[i]
    # print('filename:', filename)
    # print('F:', F)
    # print('E:', E)
    # print('\n')
    F_values[F] += 1
    E_values[E] += 1

    with rasterio.open(map_directory + '/' + filename) as tif:
      r, g, b, a = tif.read()
      meta = tif.meta

    heatmap_filename = 'heatmap_' + filename
    with rasterio.open(heatmap_directory + '/' + heatmap_filename, 'w', **meta) as dst:
      if F > THRESHOLD and E <= THRESHOLD: # Tile contains floating but not emergent - should tint red
        g *= 0
        b *= 0
        r -= 8 * PIXEL_DELTA
        r += F * PIXEL_DELTA

      elif F <= THRESHOLD and E > THRESHOLD: # Tile contains emergent but not floating - should tint blue
        r *= 0
        g *= 0
        b -= 8 * PIXEL_DELTA
        b += E * PIXEL_DELTA
      
      elif F > THRESHOLD and E > THRESHOLD: # Tile contains both floating and emergent - should tint green
        r *= 0
        b *= 0
        g -= 8 * PIXEL_DELTA
        g += ((F + E) * PIXEL_DELTA) // 2

      dst.write_band(1, r)
      dst.write_band(2, g)
      dst.write_band(3, b)
      dst.write_band(4, a)

  for i in range(9):
    F_values[i] /= len(filenames)
    E_values[i] /= len(filenames)

  print('F_values:', F_values)
  print('E_values:', E_values)

# test_X = [
#   '1-1-unlabeled.tif.123',
#   '1-1-unlabeled.tif.125',
#   '1-1-unlabeled.tif.761',
#   '3-3-unlabeled.tif.1280',
#   '9-3-unlabeled.tif.549',
# ]



  

