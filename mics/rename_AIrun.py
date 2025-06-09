import os
from glob import glob
import pandas as pd
import shutil
import numpy as np
import cv2
import pandas as pd

ref_path = '/rsrch6/home/trans_mol_path/yuan_lab/TIER1/artemis_lei/AIrun_pilot181'
src_path = '/rsrch6/home/trans_mol_path/yuan_lab/TIER1/artemis_lei/Discovery'
dst_path = '/rsrch9/home/plm/idso_fa1_pathology/TIER1/aitil_clia/ImageValidationPilot2025'
label_map = sorted(glob(os.path.join(ref_path,'*.csv')))[0]

df = pd.read_csv(label_map)
aperio_id = list(df["ID"])
de_id = list(df["pilot2_image_ID"])

for i in range(len(aperio_id)):
    file_name = aperio_id[i] + '.svs'
    file = os.path.join(src_path, file_name)
    new_name = de_id[i] + '.svs'
    dst_file = os.path.join(dst_path, new_name)
    shutil.copy2(file, dst_file)
    


'''
src_path = '/rsrch6/home/trans_mol_path/yuan_lab/TIER1/artemis_lei/AIrun_pilot20'
label_map = sorted(glob(os.path.join(src_path,'*.csv')))[0]

df = pd.read_csv(label_map)
aperio_id = list(df["ID"])
de_id = list(df["pilotID"])

for i in range(len(aperio_id)):
    file_name = aperio_id[i] + '.svs'
    file = os.path.join(src_path, file_name)
    new_name = de_id[i] + '.svs'
    new_file_name = os.path.join(src_path, new_name)
    os.rename(file, new_file_name)
'''