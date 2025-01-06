import os
from glob import glob
import numpy as np
import pandas as pd
import scipy.io as sio
import cv2
import multiprocessing
from functools import partial

### this is to test the optimal cutoff, after running ai-til pipeline, if wanting to revisit the nuclei area,
### option 1
## step 1, remove smaller nucleus, get annotated image and csv files
## step 2, then use the filtered csv files from cell_seg to guide refined cell types which cells should be kept
### pros: can have annotated images to vis, cons: have to repeat for different cutoffs

### option 2
## step 1, keep smaller nucleus, get annotated image and csv files
## step 2, then use the filtered csv files from cell_seg to guide refined cell types which cells should be removed
### pros: can have annotated images to vis for filtered cells, cons: have to repeat for different cutoffs

### option 3
## step 1, get nucle area and add them to csv files as an additional col
## step 2, then just removed from refined cell types as per the predefined cutoff
### pros: can quickly test different cutoffs, cons: cannot do the vis


#option 3
def post_process_images(results_dir, sub_dir_name, output_dir):
    if not os.path.isdir(os.path.join(output_dir, sub_dir_name)):
        os.makedirs(os.path.join(output_dir, sub_dir_name))
    
    files = sorted(glob(os.path.join(results_dir, 'mat', sub_dir_name, 'Da*.mat')))
    with multiprocessing.Pool(max(1, multiprocessing.cpu_count()-1)) as pool:
        pool.map(partial(save_segmentation_output, results_dir=results_dir, sub_dir_name=sub_dir_name, output_dir=output_dir), files)


def save_segmentation_output(file_path, results_dir, sub_dir_name, output_dir):
    mat_file_name = os.path.basename(file_path)
    #print('%s\n' % mat_file_name)
    
    mat = sio.loadmat(os.path.join(results_dir, 'mat', sub_dir_name, mat_file_name))
    if 'mat' in mat:
        mat = mat['mat']
    bin_label = mat['BinLabel']
    if isinstance(bin_label, np.ndarray) and bin_label.dtype == object:
        bin_label = bin_label[0, 0]  # Extract the actual 2D array
    # Ensure the array is in uint8 format
    bin_label = np.ascontiguousarray(bin_label.astype(np.uint8))
    #mat['BinLabel'] = np.ascontiguousarray(np.uint8(np.array(mat['output'])[:,:,1]>0.2)) # this line has been done and stored in the ai-til pipeline
    contours, _ = cv2.findContours(bin_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #opencv4.x has two returns
    
    if len(contours) > 0: 
        centroids = np.array([np.int64(np.round(np.mean(contour[:, 0, :], axis=0))) for contour in contours])
        areas_contour = np.array([cv2.contourArea(contour) for contour in contours]) #subject to the contour shape

        #filled-in areas for each contour
        areas_sum = []
        for contour in contours:
            contour_mask = np.zeros_like(bin_label, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 1, thickness=cv2.FILLED)
            filled_area = np.sum(bin_label * contour_mask)
            areas_sum.append(filled_area)
        areas_sum = np.array(areas_sum)  # Convert areas to a NumPy array
        
    else:
        centroids = np.empty((0, 2))
        areas_contour = np.empty(0, dtype=float)
        areas_sum = np.empty(0, dtype=float)

    pd.DataFrame(data={'V1': [None]*centroids.shape[0], 'V2': centroids[:, 0]+1, 'V3': centroids[:, 1]+1, 
                'Area_contour': areas_contour, 'Area_sum': areas_sum}).to_csv(os.path.join(output_dir, sub_dir_name, mat_file_name[:-4]+'.csv'), index=False)


if __name__ == "__main__":
    results_dir = '/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/discovery/til/3_cell_seg' #3_cell_seg
    data_dir = '/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/discovery/til/1_cws_tiling'
    output_dir = os.path.join(results_dir, 'csv_area')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    file_name_pattern = '*.svs'
    files = sorted(glob(os.path.join(data_dir, file_name_pattern)))
    for file in files:
        sub_dir_name = os.path.basename(file)
        print(sub_dir_name)
        post_process_images(results_dir=results_dir,
                            sub_dir_name=sub_dir_name,
                            output_dir = output_dir)