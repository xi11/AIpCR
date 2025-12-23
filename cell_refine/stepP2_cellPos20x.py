import numpy as np
import pickle
import os
import math
import pandas as pd
from glob import glob

###stepP2 - to have coordiates of cells at different scales

def merge_csv_files(wsi_path, results_dir, output_csv):
    if not os.path.isdir(os.path.dirname(output_csv)):
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    param = pickle.load(open(os.path.join(wsi_path, 'param.p'), 'rb'))
    #print(param)

    slide_dimension = np.array(param['slide_dimension']) / param['rescale']
    slide_h = slide_dimension[1]
    slide_w = slide_dimension[0]
    #print(slide_h, slide_w)
    cws_read_size = param['cws_read_size']
    cws_h = cws_read_size[0]
    cws_w = cws_read_size[1]
    divisor = np.float64(16)

    # Initialize Pandas Data Frame
    cellPos = pd.DataFrame(columns=['x', 'y', 'class', 'x_tile', 'y_tile', 'tile', 'x_20x', 'y_20x', 'x_fullres', 'y_fullres'])
    iter_tot_tiles = 0
    for h in range(int(math.ceil((slide_h - cws_h) / cws_h + 1))):
        for w in range(int(math.ceil((slide_w - cws_w) / cws_w + 1))):
            #print('Processing Da_' + str(iter_tot_tiles))
            start_h = h * cws_h
            start_w = w * cws_w

            if os.path.isfile(os.path.join(results_dir, 'Da' + str(iter_tot_tiles) + '.csv')):
                csv = pd.read_csv(os.path.join(results_dir, 'Da' + str(iter_tot_tiles) + '.csv'), usecols = ['V1', 'V2','V3', 'V6']) # v6 is the refined class
                csv.columns = ['class', 'x', 'y', 'class2']
                ##added to get tile loc
                cell_num = len(csv.x)
                csv['x_tile'] = csv.x
                csv['y_tile'] = csv.y
                #print(csv.x, csv.y)
                csv['tile'] = ['Da'+ str(iter_tot_tiles)]* cell_num
                ##added to get tile loc
                csv.x = csv.x + start_w
                csv.y = csv.y + start_h
                # detection = np.divide(np.float64(detection), divisor)
                cellPos = pd.concat([cellPos, csv], ignore_index = True) #As of pandas 2.0, append (previously deprecated) was removed. You need to use concat instead (for most applications):
                #print(cellPos.columns)


            iter_tot_tiles += 1

    cellPos.x_20x = np.float64(cellPos.x)
    cellPos.y_20x = np.float64(cellPos.y)
    cellPos.x_fullres = np.round(param['rescale'] * np.float64(cellPos.x)).astype('int')
    cellPos.y_fullres = np.round(param['rescale'] * np.float64(cellPos.y)).astype('int')
    cellPos.x = np.round(np.divide(np.float64(cellPos.x), divisor)).astype('int')
    cellPos.y = np.round(np.divide(np.float64(cellPos.y), divisor)).astype('int') #xp tried to removed np.round
    cellPos.loc[cellPos.x == 0, 'x'] = 1
    cellPos.loc[cellPos.y == 0, 'y'] = 1
    new_order = ['class2'] + [col for col in cellPos.columns if col != 'class2']
    cellPos = cellPos[new_order]
    cellPos.to_csv(output_csv, index=False)

if __name__ == "__main__":
    wsi_path_all = '/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/TransNeo_Nature/1_cws_tiling'
    results_dir_all = '/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/TransNeo_Nature/4_cell_class_segformerBRCAartemis1733/csv'
    cellPos = '/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/TransNeo_Nature/4_cell_class_segformerBRCAartemis1733/CellPos'
    if not os.path.isdir(cellPos):
        os.makedirs(cellPos)
    files = sorted(glob(os.path.join(wsi_path_all, '*.svs')))
    for file in files:
        file_name = os.path.basename(file)
        results_dir = os.path.join(results_dir_all, file_name)
        output_csv = os.path.join(cellPos, file_name[:-4]+'.csv')
        print(results_dir)
        if not os.path.exists(output_csv):
            merge_csv_files(file, results_dir, output_csv)