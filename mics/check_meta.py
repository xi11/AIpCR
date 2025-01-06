import os
#os.add_dll_directory(r'C:\Tools\openslide-win64-20171122\bin')
import openslide as openslide
from glob import glob
import numpy as np

#import javabridge as jv, bioformats as bf
#from xml.etree import ElementTree as ETree



file_path = '/rsrch6/home/trans_mol_path/yuan_lab/TIER1/artemis_lei/Discovery'
file = sorted(glob(os.path.join(file_path, '027*.svs')))
count = 0
for file_name in file:
    openslide_obj = openslide.OpenSlide(filename=file_name)
    print(file_name)
    print(openslide_obj.properties)
    try:
        count = count+1
        objective_power = float(openslide_obj.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
        objective_mppx = float(openslide_obj.properties[openslide.PROPERTY_NAME_MPP_X])
        print(openslide_obj.properties['aperio.ICC Profile'])
        #objective_mppy = float(openslide_obj.properties[openslide.PROPERTY_NAME_MPP_Y])
        #print(os.path.basename(file_name), objective_mppx, objective_mppy, count)
        #if np.round(objective_mppx, 1) ==0.5:
            #print(os.path.basename(file_name), objective_mppx)
    except:
        KeyError
        print(os.path.basename(file_name))