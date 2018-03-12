from PIL import Image
import numpy as np
import rasterio

# Read raster bands directly to Numpy arrays.
with rasterio.open('./parsed_tifs/Parsed_TIFF.99.tif') as src:
  r, g, b, a = src.read()
  meta = src.meta
  meta.update(count = 3)

# Combine arrays in place. Expecting that the sum will
# temporarily exceed the 8-bit integer range, initialize it as
# a 64-bit float (the numpy default) array. Adding other
# arrays to it in-place converts those arrays "up" and
# preserves the type of the total array.
bands = [g, b, a]

# Write the product as a raster band to a new 8-bit file. For
# the new file's profile, we start with the meta attributes of
# the source file, but then change the band count to 1, set the
# dtype to uint8, and specify LZW compression.


with rasterio.open('example-total.tif', 'w', **meta) as dst:
  for i, band in enumerate(bands):
    print(band)
    # print(band)
    # np_band = np.zeros_like(band)
    # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    # print(np_band)
    # dst.write_band(i + 1, band)
  # for i, band in enumerate(data):
  #   print(i)
  #   print(band)
  #   dest = np.zeros_like(band)
  #   dst.write(dest.astype(rasterio.uint8), indexes=i)

outfile = 'translated-tif.jpg'
try:
  im = Image.open('example-total.tif')
  out = im.convert('RGB')
  out.save(outfile, 'jpg', quality=100)
except Exception as e:
  print(e)
