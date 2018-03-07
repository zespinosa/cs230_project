import arcpy
arcpy.env.workspace = r"C:\Users\elyons\Desktop"
arcpy.CheckOutExtension('spatial')
from arcpy.sa import Raster

##NIR = Raster("Parsed_TIFF.68.tif/Band_4")
##RED = Raster("Parsed_TIFF.68.tif/Band_3")
##GRN = Raster("Parsed_TIFF.68.tif/Band_2")

arcpy.CompositeBands_management("Parsed_TIFF.68.tif/Band_4;Parsed_TIFF.68.tif/Band_3;Parsed_TIFF.68.tif/Band_2", "test4.tif")
