import os
from PIL import Image, TiffImagePlugin
TiffImagePlugin.READ_LIBTIFF=True

## http://deeplearning.lipingyang.org/2017/02/15/converting-tiff-to-jpeg-in-python/
def tifToJPG():
    current_path = os.getcwd()
    path = current_path + "/rawData"
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            print(os.path.join(root, name))
            #if os.path.splitext(os.path.join(root, name))[1].lower() == ".tiff":
            if os.path.splitext(os.path.join(root, name))[1].lower() == ".tif":
                if os.path.isfile(os.path.splitext(os.path.join(root, name))[0] + ".jpg"):
                    print("A jpeg file already exists for %s" % name)
                    # If a jpeg with the name does *NOT* exist, covert one from the tif.
                else:
                    outputfile = os.path.splitext(os.path.join(root, name))[0] + ".jpg"
                    try:
                        im = Image.open(os.path.join(root, name))
                        print("Converting jpeg for %s" % name)
                        im.thumbnail(im.size)
                        out = im.convert("RBG")
                        out.save(outputfile, "JPEG", quality=100)
                    except Exception as e: print(e)
tifToJPG()
