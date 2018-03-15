import rasterio
import sys

in_directory = './labeled/'
out_directory = './tinted/'


def tint_tiff(X, g_delta, b_delta):
  for filename in X:
    with rasterio.open(in_directory + filename + '.tif') as tif:
      r, g, b, a = tif.read()
      meta = tif.meta

    with rasterio.open(out_directory + filename + '-tint.tif', 'w', **meta) as dst:
      if g_delta == 1 and b_delta == 1:
        return

      r //= 9
      g //= 9
      b //= 9

      r *= (9 - g_delta)
      r *= (9 - b_delta)
      g *= g_delta
      b *= b_delta

      # r *= (9 - b_delta)
      # g *= (9 - b_delta) 
      # b *= b_delta

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

test_X = [
  '1-1-unlabeled.tif.123',
  '1-1-unlabeled.tif.125',
  '1-1-unlabeled.tif.761',
  '3-3-unlabeled.tif.1280',
  '9-3-unlabeled.tif.549',
]


tint_tiff(test_X, 4, 1)


  

