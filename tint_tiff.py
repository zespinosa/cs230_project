import rasterio
import sys

in_directory = './labeled/'
out_directory = './tinted/'


def tint_tiff(X, g_delta, b_delta):
  for filename in X:
    with rasterio.open(in_directory + filename + '.tif') as tif:
      r, g, b, a = tif.read()
      meta = tif.meta

    with rasterio.open(out_directory + filename + '-both.tif', 'w', **meta) as dst:
      if g_delta == 1 and b_delta == 1:
        return

      r //= 10
      g //= 10
      b //= 10

      g *= g_delta
      b *= b_delta

      dst.write_band(1, r)
      dst.write_band(2, g)
      dst.write_band(3, b)
      dst.write_band(4, a)

      # dst.write_colormap(1, {
      #   0: (0, 0, 0, 0),
      #   255: (0, 0, 0, 0)
      # })
      # dst.write_colormap(2, {
      #   0: (127, 127, 127, 127),
      #   255: (255, 255, 255, 255)
      # })
      # dst.write_colormap(3, {
      #   0: (0, 0, 0, 0),
      #   255: (255, 255, 255, 255)
      # })
      # dst.write_colormap(4, {
      #   0: (0, 0, 0, 0),
      #   255: (0, 0, 0, 0)
      # })
      # print(dst.profile)


tint_tiff(test_X, 9, 9)


  

