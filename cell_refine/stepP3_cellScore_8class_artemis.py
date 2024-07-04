import os
import pandas as pd
from collections import OrderedDict
from glob import glob
import cv2
import numpy as np


def cell_count(cell_csv_path, output_path, tmeSS1_path, file_name=None, classes=('f', 'l', 't', 'o')):
    if file_name is None:
        file_name = os.path.basename(cell_csv_path)

    cellPos = pd.read_csv(cell_csv_path)
    cell_counts = [sum(cellPos.loc[:, 'class2'] == c) for c in classes]
    total = sum(cell_counts)
    cell_percentages = [100 * count / total if total != 0 else 0 for count in cell_counts]

    tmeSS1 = cv2.imread(tmeSS1_path) #BGR
    yellow_color = np.array([0, 255, 255])  # BGR
    brown_color = np.array([0, 0, 128])  # BGR
    red_color = np.array([0, 0, 255])  # BGR
    mask_stroma = np.logical_or(np.all(tmeSS1 == yellow_color, axis=-1),
                                np.all(tmeSS1 == red_color, axis=-1))
    mask_tumor = np.all(tmeSS1 == brown_color, axis=-1)

    x_indices = cellPos['x'].values - 1
    y_indices = cellPos['y'].values - 1

    #cellPos['y'] - 1, cellPos['x'] - 1 #same with y_indices, x_indices

    stroma_fibro = cellPos[(cellPos['class2'] == 'f') & (mask_stroma[y_indices, x_indices] == 1)]
    stroma_lym = cellPos[(cellPos['class2'] == 'l') & (mask_stroma[y_indices, x_indices] == 1)]
    tumor_lym = cellPos[(cellPos['class2'] == 'l') & (mask_tumor[y_indices, x_indices] == 1)]
    tumor_can = cellPos[(cellPos['class2'] == 't') & (mask_tumor[y_indices, x_indices] == 1)]

    column_names = ['FileName'] + ['#' + c for c in classes] + ['%' + c for c in classes] + ['#f_stroma'] + ['#l_stroma'] + ['#l_tumor'] + ['#t_tumor']
    row = [file_name] + cell_counts + cell_percentages + [len(stroma_fibro)] + [len(stroma_lym)] + [len(tumor_lym)] + [len(tumor_can)]
    M_f = pd.DataFrame(data=OrderedDict(zip(column_names, row)), index=[0])
    M_f.to_csv(output_path, index=False)


if __name__ == '__main__':
    wsi_path_all = '/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/discovery/til/1_cws_tiling'
    cell_pos_path = '/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/discovery/til/4_cell_class_segformerBRCAonly/CellPos'
    cell_score_path = '/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/discovery/til/4_cell_class_segformerBRCAonly/CellScore'
    tme_seg_path = '/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/discovery/mit-b3-finetuned-tmeTCGA-60-lr00001-s512-20x768/mask_ss1512'
    if not os.path.isdir(cell_score_path):
        os.makedirs(cell_score_path)
    files = sorted(glob(os.path.join(wsi_path_all, '*.svs')))
    for file in files:
        file_name = os.path.basename(file)
        print(file_name)
        filePos_csv = os.path.join(cell_pos_path, file_name[:-4]+'.csv')
        fileScore_csv = os.path.join(cell_score_path, file_name[:-4]+'.csv')
        tme_mask = os.path.join(tme_seg_path, file_name + '_Ss1.png')
        if not os.path.exists(fileScore_csv):
            cell_count(filePos_csv, fileScore_csv, tme_mask, file_name)