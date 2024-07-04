import os
from glob import glob
import numpy as np
from PIL import Image
import shutil
import cv2

##2024 Apr, to convert curated patches (mostly 512*512 at 10x) into 256, and convert label 8 to label 9 as blood is separated from necrosis as 8
src_path = '/Users/xiaoxipan/Documents/project/public_data/wsss4luad/size256abovex10/patch512_1024luad/mask'
dst_path = '/Users/xiaoxipan/Documents/project/public_data/wsss4luad/size256abovex10/patch512_1024luad/mask9'
os.makedirs(dst_path, exist_ok=True)

files = sorted(glob(os.path.join(src_path, '*.png')))
for file in files:
    file_name = os.path.basename(file)[5:]
    mask = cv2.imread(file, 0)
    mask[mask == 8] = 9
    cv2.imwrite(os.path.join(dst_path, file_name), mask)

'''
src_path = '/Users/xiaoxipan/Documents/project/public_data/wsss4luad/size256abovex10/mask_std_color'
dst_path = '/Users/xiaoxipan/Documents/project/public_data/wsss4luad/size256abovex10/mask_digital'
os.makedirs(dst_path, exist_ok=True)
files = sorted(glob(os.path.join(src_path, '*.png')))
for file in files:
    file_name = os.path.basename(file)
    image = Image.open(file)
    image = image.convert('RGB')  # Ensure image is in RGB format
    # Convert the image to a NumPy array
    image_array = np.array(image)
    # Define the color to label mapping
    color_to_label = {
        (0, 0, 0): 0,
        (128, 0, 0): 1,
        (255, 255, 0): 2,
        (0, 0, 128): 8
    }

    # Initialize an empty array for the labels with the same height and width as the input image
    labels_array = np.zeros(image_array.shape[:2], dtype=np.uint8)

    # Iterate over the mapping and update the labels array based on the color to label mapping
    for color, label in color_to_label.items():
        # Find pixels matching the current color
        matches = np.all(image_array == color, axis=-1)
        # Update the labels array with the current label where there is a color match
        labels_array[matches] = label

    # At this point, labels_array contains the digital labels corresponding to the original image's colors

    # Optional: Save the labels_array to an image file to visualize the labels
    labels_image = Image.fromarray(np.uint8(labels_array), mode='L')
    labels_image.save(os.path.join(dst_path, file_name))  # Specify the save path


##to remove files
src_path = '/Volumes/xpan7/tmesegK8/patchx10x20_bcss_wsss4luad/image'
ref_path = '/Volumes/xpan7/tmesegK8/patchx10x20_bcss_wsss4luad/maskPng'
dst_path = '/Volumes/xpan7/tmesegK8/patchx10x20_bcss_wsss4luad/maskPng_removed'
os.makedirs(dst_path, exist_ok=True)

files = sorted(glob(os.path.join(ref_path, '*.png')))
for file in files:
    file_name = os.path.basename(file)[5:]
    src_file = os.path.join(src_path, file_name)
    dst_file = os.path.join(dst_path, 'mask_'+file_name)
    if not os.path.exists(src_file):
        shutil.move(file, dst_file)


# to color masks
src_path = '/Users/xiaoxipan/Documents/project/public_data/wsss4luad/size256abovex10/mask'
dst_path = '/Users/xiaoxipan/Documents/project/public_data/wsss4luad/size256abovex10/mask_std_color'
os.makedirs(dst_path, exist_ok=True)

files = sorted(glob(os.path.join(src_path, '*.png')))
for file in files:
    file_name = os.path.basename(file)
    image = Image.open(file)
    image = image.convert('RGB')
    # Define the colors to replace
    # Format: {(source_color): (target_color)}
    color_mapping = {
        (0, 64, 128): (128, 0, 0),
        (64, 128, 0): (255, 255, 0),
        (243, 152, 0): (0, 0, 128),
        (255, 255, 255): (0, 0, 0)
    }

    # Get the pixel data
    pixels = image.load()

    # Iterate over each pixel
    for y in range(image.height):
        for x in range(image.width):
            current_color = pixels[x, y]
            if current_color in color_mapping:
                pixels[x, y] = color_mapping[current_color]

    # Save or display the modified image
    image.save(os.path.join(dst_path, file_name))

# to pick up images with width/height>256
src_path = '/Users/xiaoxipan/Documents/project/public_data/wsss4luad/3.testing'
dst_path_img = '/Users/xiaoxipan/Documents/project/public_data/wsss4luad/size256abovex10/image'
dst_path_mask = '/Users/xiaoxipan/Documents/project/public_data/wsss4luad/size256abovex10/mask'
os.makedirs(dst_path_img, exist_ok=True)
os.makedirs(dst_path_mask, exist_ok=True)

files = sorted(glob(os.path.join(src_path, 'img', '*.png')))
for file in files:
    file_name = os.path.basename(file)
    img = np.array(Image.open(file))
    if min(img.shape[0], img.shape[1]) > 256:
        file_mask = os.path.join(src_path, 'mask', file_name)
        shutil.copy(file, os.path.join(dst_path_img, 'test_'+file_name))
        shutil.copy(file_mask, os.path.join(dst_path_mask, 'test_'+file_name))
'''