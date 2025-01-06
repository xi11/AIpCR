import os
from glob import glob
import numpy as np
import pandas as pd
import multiprocessing
from functools import partial

def multi_process_merge(seg_csv_dir, class_csv_dir, sub_dir_name, output_dir):
    if not os.path.isdir(os.path.join(output_dir, sub_dir_name)):
        os.makedirs(os.path.join(output_dir, sub_dir_name))

    files = sorted(glob(os.path.join(seg_csv_dir, sub_dir_name, '*.csv')))
    with multiprocessing.Pool(max(1, multiprocessing.cpu_count()-1)) as pool:
        pool.map(partial(merge_seg_class_csv, seg_csv_dir=seg_csv_dir, class_csv_dir=class_csv_dir, sub_dir_name=sub_dir_name, output_dir=output_dir), files)


def merge_seg_class_csv(file_path, seg_csv_dir, class_csv_dir, sub_dir_name, output_dir):
    file_name = os.path.basename(file_path)
    seg_csv = pd.read_csv(os.path.join(seg_csv_dir, sub_dir_name, file_name), usecols=["V2", "V3", "Area_contour", "Area_sum"])
    class_csv = pd.read_csv(os.path.join(class_csv_dir, sub_dir_name, file_name))
    merged_csv = pd.merge(seg_csv, class_csv, on=["V2", "V3"], how="inner")
    output_file = os.path.join(output_dir, sub_dir_name, file_name)
    merged_csv.to_csv(output_file, index=False)



if __name__ == '__main__':
    seg_csv_dir = '/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/discovery/til/3_cell_seg/csv_area'
    class_csv_dir = '/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/discovery/til/4_cell_class_segformerBRCAartemis/csv'
    output_dir = '/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/discovery/til/4_cell_class_segformerBRCAartemis/csv_area'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    slides = sorted(glob(os.path.join(seg_csv_dir, '*.svs')))
    for slide in slides:
        sub_dir_name = os.path.basename(slide)
        print(sub_dir_name)
        multi_process_merge(seg_csv_dir=seg_csv_dir,
                            class_csv_dir=class_csv_dir,
                            sub_dir_name=sub_dir_name,
                            output_dir = output_dir)
