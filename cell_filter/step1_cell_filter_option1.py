import os
from glob import glob
import numpy as np
import pandas as pd
import scipy.io as sio
import time
import cv2

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


#option1
#Define the area cutoff
area_cutoff = 10  
def save_segmentation_output(file_path, opts, sub_dir_name):
    mat_file_name = os.path.basename(file_path)
    # print('%s\n' % mat_file_name)
    
    if not os.path.isfile(os.path.join(opts.results_dir, 'annotated_images', sub_dir_name, mat_file_name[:-3]+'png')):
        image_path_full = os.path.join(opts.data_dir, sub_dir_name, mat_file_name[:-3]+'jpg')
    
        mat = sio.loadmat(os.path.join(opts.results_dir, 'mat', sub_dir_name, mat_file_name))
        if 'mat' in mat:
            mat = mat['mat']
        mat['BinLabel'] = np.ascontiguousarray(np.uint8(np.array(mat['output'])[:,:,1]>0.2))
        _, contours, _ = cv2.findContours(mat['BinLabel'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
       ###filter small nucles
        if len(contours) > 0:
            filtered_centroids = []
            for contour in contours:
                if cv2.contourArea(contour) >= area_cutoff:
                    centroid = np.int64(np.round(np.mean(contour, axis=0)))
                    filtered_centroids.append(centroid)
            if filtered_centroids:
                centroids = np.concatenate(filtered_centroids, axis=0)
            else:
                centroids = np.empty((0, 2))
        else:
            centroids = np.empty((0, 2))

        pd.DataFrame(data={'V1': [None]*centroids.shape[0], 'V2': centroids[:, 0]+1, 'V3': centroids[:, 1]+1}).to_csv(os.path.join(opts.results_dir, 'csv', sub_dir_name, mat_file_name[:-4]+'.csv'), index=False)
        if not opts.minimal_output:
            mat['BinLabel'] = np.bool_(cv2.drawContours(mat['BinLabel'], contours, -1, 1, cv2.FILLED))
            sio.savemat(os.path.join(opts.results_dir, 'mat', sub_dir_name, mat_file_name),  {'mat': mat})
            
            im = cv2.imread(image_path_full)
            annotated_image = cv2.drawContours(im, contours, -1, (0, 255, 0))
            cv2.imwrite(os.path.join(opts.results_dir, 'annotated_images', sub_dir_name, mat_file_name[:-4]+'.png'), annotated_image)
    else:
        # print('Already Processed %s\n' % os.path.join(opts.results_dir, 'annotated_images', sub_dir_name, mat_file_name[:-3]+'png'))
        pass
