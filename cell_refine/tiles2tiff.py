import libtiff   #pip install pylibtiff
import numpy as np
import cv2
import os
import glob
import math

def tiles2tiff(tile_path, tile_size, image_size, out_file, ext='jpg', jpeg_compression=True):
    out_folder = os.path.dirname(out_file)
    os.makedirs(out_folder, exist_ok=True)
    
    n_tiles = len(glob.glob(os.path.join(tile_path, 'Da*.'+ext)))

    if n_tiles == 0:
        print('No tiles found in ' + tile_path + '!')
    else:
        tiff = libtiff.TIFF.open(out_file, mode='w')
        tiff.SetField(libtiff.libtiff_ctypes.TIFFTAG_IMAGEWIDTH, image_size[0])
        tiff.SetField(libtiff.libtiff_ctypes.TIFFTAG_IMAGELENGTH, image_size[1])
        tiff.SetField(libtiff.libtiff_ctypes.TIFFTAG_PHOTOMETRIC, libtiff.libtiff_ctypes.PHOTOMETRIC_RGB)
        tiff.SetField(libtiff.libtiff_ctypes.TIFFTAG_PLANARCONFIG, libtiff.libtiff_ctypes.PLANARCONFIG_CONTIG)
        tiff.SetField(libtiff.libtiff_ctypes.TIFFTAG_BITSPERSAMPLE, 8)
        tiff.SetField(libtiff.libtiff_ctypes.TIFFTAG_SAMPLESPERPIXEL, 3)
        tiff.SetField(libtiff.libtiff_ctypes.TIFFTAG_TILEWIDTH, tile_size[0])
        tiff.SetField(libtiff.libtiff_ctypes.TIFFTAG_TILELENGTH, tile_size[1])
        
        if jpeg_compression:
            tiff.SetField(libtiff.libtiff_ctypes.TIFFTAG_COMPRESSION, libtiff.libtiff_ctypes.COMPRESSION_JPEG)
            tiff.SetField(libtiff.libtiff_ctypes.TIFFTAG_JPEGQUALITY, 95)
        else:
            tiff.SetField(libtiff.libtiff_ctypes.TIFFTAG_COMPRESSION, libtiff.libtiff_ctypes.COMPRESSION_LZW)
            
        tile_grid = [math.ceil(image_size[0]/tile_size[0]), math.ceil(image_size[1]/tile_size[1])]
        grid_tiles = tile_grid[0]*tile_grid[1]
        
        for i in range(grid_tiles):
            print('Writing tile %d of %d to tif' % (i+1, grid_tiles))

            tile_file = os.path.join(tile_path, 'Da' + str(i) + '.' + ext)

            if os.path.isfile(tile_file):
                im = cv2.cvtColor(cv2.imread(tile_file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

                y, x = divmod(i, tile_grid[0])
                
                expected_tile_size = (min(tile_size[1], image_size[1]-(tile_size[1]*y)), min(tile_size[0], image_size[0]-(tile_size[0]*x))) # Yeman list --> tuple

                if not (im.shape[0] == expected_tile_size[0] and im.shape[1] == expected_tile_size[1]):
                    im = cv2.resize(im, expected_tile_size)
                
                if not (tile_size[1] == expected_tile_size[0] and tile_size[0] == expected_tile_size[1]):
                    im = np.pad(im, ((0, tile_size[1] - expected_tile_size[0]), (0, tile_size[0] - expected_tile_size[1]), (0, 0)), 'constant') # Yeman added mode=constant
                
                tiff.WriteTile(np.ascontiguousarray(im).ctypes.data, x*tile_size[0], y*tile_size[1], 0, 0)

        tiff.close()

