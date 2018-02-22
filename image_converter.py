import os
from PIL import Image

current_path = os.getcwd() 
for full_filename in os.listdir('./assets/original_tifs'):
  filename, ext = os.path.splitext(full_filename)
  if ext == '.tif':
    if os.path.isfile('./assets/jpegs/' + filename + '.jpg'):
      print('A jpeg file already exists for ' + filename)
    # If a jpeg with the name does *NOT* exist, covert one from the tif.
    else:
      outputfile = './assets/jpegs/' + filename + '.jpg'
      try:
        im = Image.open('./assets/original_tifs/' + full_filename)
        print('Converting jpeg for ' + filename)
        im.thumbnail(im.size)
        im.save(outputfile, 'JPEG', quality=100)
      except Exception as e:
        print(e)
        