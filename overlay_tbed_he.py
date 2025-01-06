# -*- coding: utf-8 -*-
# adapted from https://github.com/idso-fa1-pathology/mihc-quantification/blob/main/SegFormer/postprocess/mdacc/overlay_seg_mask.py

import os, sys
import argparse
import numpy as np
#import openslide
#from skimage import io, transform
import cv2
from glob import glob



def set_args():
    parser = argparse.ArgumentParser(description = "Overlay predicted tumor bed onto H&Es")
    parser.add_argument("--data_root",  type=str,   default="/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei")
    parser.add_argument("--cws_dir",     type=str,   default="til/1_cws_tiling")    
    parser.add_argument("--file_pattern", type=str, default=".svs")
    parser.add_argument("--mask_dir",   type=str,   default="tbed1536_ss1/maskLuadLusc_2use")
    parser.add_argument("--overlay_dir",  type=str, default="tbed1536_ss1/maskLuadLusc_overlay")    

    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    args = set_args()
    dset_name = "discovery"
    dset_root = os.path.join(args.data_root, dset_name)
    cws_dir = os.path.join(dset_root, args.cws_dir)        
    tumor_mask_dir = os.path.join(dset_root, args.mask_dir)
    tumor_overlay_dir = os.path.join(dset_root, args.overlay_dir)
    if not os.path.exists(tumor_overlay_dir):
        os.makedirs(tumor_overlay_dir)

    files = sorted(glob(os.path.join(tumor_mask_dir, "*.png")))
    for file in files:
        file_name = os.path.basename(file)[:-9]
        # read ss1 H&E image
        ss1_image = cv2.imread(os.path.join(cws_dir, file_name+args.file_pattern, 'Ss1.jpg'))
        mask_image = cv2.imread(os.path.join(tumor_mask_dir, file_name+'_tbed.png'))
        # overlay prediction
        overlaid_image = ss1_image.copy()
        overlay_color = np.array([0, 255, 204], dtype='uint8')  # Green color for prediction
        mask_image = mask_image[:,:,0]
        overlaid_image[mask_image == 255] = overlay_color
        overlaid_image = cv2.addWeighted(ss1_image, 0.7, overlaid_image, 0.3, 0)

        # save the results
        overlay_slide_path = os.path.join(tumor_overlay_dir, file_name + "_overlay.png")
        cv2.imwrite(overlay_slide_path, overlaid_image)    