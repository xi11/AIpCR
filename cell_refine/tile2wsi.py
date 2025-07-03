import tifffile
import numpy as np
import os
import cv2
import math

def tiles2tiff(tile_path, tile_size, image_size, out_file, ext='png'):
    full_image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

    tile_grid_x = math.ceil(image_size[0] / tile_size[0])
    tile_grid_y = math.ceil(image_size[1] / tile_size[1])

    for y in range(tile_grid_y):
        for x in range(tile_grid_x):
            tile_index = y * tile_grid_x + x
            tile_file = os.path.join(tile_path, f'Da{tile_index}.{ext}')
            if os.path.exists(tile_file):
                im = cv2.imread(tile_file, cv2.IMREAD_COLOR)
                if im is None:
                    print(f"Warning: Cannot read tile {tile_file}")
                    continue

                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

                # Compute actual region to place in the full image
                y_start = y * tile_size[1]
                x_start = x * tile_size[0]
                y_end = min(y_start + tile_size[1], image_size[1])
                x_end = min(x_start + tile_size[0], image_size[0])

                expected_tile_size = (y_end - y_start, x_end - x_start)

                # Resize if not matching the expected size
                if (im.shape[0], im.shape[1]) != expected_tile_size:
                    im = cv2.resize(im, (expected_tile_size[1], expected_tile_size[0]))

                full_image[y_start:y_end, x_start:x_end] = im
           

    tifffile.imwrite(out_file, full_image, compression="lzw")

