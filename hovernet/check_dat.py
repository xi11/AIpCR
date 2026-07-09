import os
from glob import glob

str_dir = '/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/IMPRESS_HER2/hovernet'
folders = sorted(glob(os.path.join(str_dir, '*')))
for folder in folders:
    folder_name = os.path.basename(folder)
    if not os.path.exists(os.path.join(folder, 'file_map.dat')):
        print(folder_name)
