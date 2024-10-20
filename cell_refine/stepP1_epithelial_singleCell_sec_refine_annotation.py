import os
from glob import glob
import numpy as np
import pandas as pd
import cv2

###stepP1 - to correct cell class with TMEseg cws masks
def Save_Classification_Output(data_dir, tme_dir, results_dir, sub_dir_name, csv_classification_dir, csv_correction_dir, color_code_path):
    #mat_file_name = os.path.basename(file_path)[:-3] + 'mat'
    csv_files = sorted(glob(os.path.join(csv_classification_dir, sub_dir_name, '*.csv')))
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    if not os.path.isdir(csv_correction_dir):
        os.makedirs(csv_correction_dir)
    if not os.path.isdir(os.path.join(results_dir, sub_dir_name)):
        os.makedirs(os.path.join(results_dir, sub_dir_name))
    if not os.path.isdir(os.path.join(csv_correction_dir, sub_dir_name)):
        os.makedirs(os.path.join(csv_correction_dir, sub_dir_name))
    for file_path in csv_files:
        csv_file_name = file_path

        image_path_full = os.path.join(data_dir, sub_dir_name, os.path.basename(file_path)[:-3] + 'jpg')
        tme_path_full = os.path.join(tme_dir, sub_dir_name, os.path.basename(file_path)[:-3] + 'png')

        if not os.path.isfile(os.path.join(results_dir, sub_dir_name, os.path.basename(file_path)[:-3] + 'png')):
            # print(os.path.basename(file_path)[:-3])
            strength = 5
            A = pd.read_csv(csv_file_name)
            detection = np.stack((np.array(A.loc[:, 'V2']), np.array(A.loc[:, 'V3'])), axis=1)
            cell_class = A.loc[:, 'V1'].tolist()
            # colorcodes = pd.read_csv(color_code_path)
            tmeseg = np.float64(cv2.cvtColor(cv2.imread(tme_path_full), cv2.COLOR_BGR2RGB))
            if not A.empty:
                for i in range(len(A.index)):
                    # print(len(A.index))
                    #to correct non-tumor to 2nd max
                    if tmeseg[detection[i, 1] - 1, detection[i, 0] - 1, 0] == 128 and tmeseg[detection[i, 1] - 1, detection[i, 0] - 1, 1] == 0 and tmeseg[detection[i, 1] - 1, detection[i, 0] - 1, 2] == 0:
                        A.loc[i, 'V5'] = 'yes'
                    else:
                        A.loc[i, 'V5'] = 'no'

                    if A.loc[i, 'V1'] == 't' and A.loc[i, 'V5'] == 'no':
                        A.loc[i, 'V6'] = A.loc[i, 'V4']
                    else:
                        A.loc[i, 'V6'] = A.loc[i, 'V1']
                    # to correct parenchyma and fat to Other
                    if tmeseg[detection[i, 1] - 1, detection[i, 0] - 1, 0] == 0 and tmeseg[detection[i, 1] - 1, detection[i, 0] - 1, 1] == 255 and tmeseg[detection[i, 1] - 1, detection[i, 0] - 1, 2] == 255:
                        A.loc[i, 'V6'] = 'o'  #parenchyma
                    if tmeseg[detection[i, 1] - 1, detection[i, 0] - 1, 0] == 128 and tmeseg[detection[i, 1] - 1, detection[i, 0] - 1, 1] == 128 and tmeseg[detection[i, 1] - 1, detection[i, 0] - 1, 2] == 0:
                        A.loc[i, 'V6'] = 'o'  #fat

                A.to_csv(os.path.join(csv_correction_dir, sub_dir_name, os.path.basename(file_path)[:-3] + 'csv'), index=False)
            else:
                A['V5'] = ''
                A['V6'] = ''
                A.to_csv(os.path.join(csv_correction_dir, sub_dir_name, os.path.basename(file_path)[:-3] + 'csv'), index=False)

            cell_class = A.loc[:, 'V6']
            colorcodes = pd.read_csv(color_code_path)
            image = np.float64(cv2.cvtColor(cv2.imread(image_path_full), cv2.COLOR_BGR2RGB)) / 255.0
            if not A.empty:
                for c in range(len(colorcodes.index)):
                    #print(detection[cell_class == colorcodes.loc[c, 'class'], :])
                    image = annotate_image_with_class(image, detection[cell_class == colorcodes.loc[c, 'class'], :], np.float64(hex2rgb(colorcodes.loc[c, 'color'])) / 255.0, strength)
            cv2.imwrite(os.path.join(results_dir, sub_dir_name, os.path.basename(file_path)[:-3] + 'png'), cv2.cvtColor(np.uint8(image * 255.0), cv2.COLOR_RGB2BGR))


def hex2rgb(hx):
    hx = hx.lstrip('#')
    return np.array([int(hx[i:i + 2], 16) for i in (0, 2, 4)])


def annotate_image_with_class(image, points, colour, strength):
    label = np.zeros(image.shape[0:2])
    label[points[:, 1] - 1, points[:, 0] - 1] = 1
    strel = np.uint8(np.fromfunction(lambda x, y: (x - strength + 1) ** 2 + (y - strength + 1) ** 2 < strength ** 2,
                                     ((2 * strength) - 1, (2 * strength) - 1), dtype=int))
    label = cv2.dilate(label, strel) > 0
    image[label] = colour
    return image


if __name__ == "__main__":
    results_dir = '/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/discovery/til/4_cell_class_segformerBRCAartemis/annotated_images' #output
    tme_dir = '/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/discovery/mit-b3-finetunedBRCA-Artemis-e60-lr00001-s512-20x512/mask_cws512_orng' #mask_cws folder as input
    csv_classification_dir = '/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/discovery/til/4_cell_class/csv'  #input
    csv_correction_dir = '/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/discovery/til/4_cell_class_segformerBRCAartemis/csv' #output
    data_dir = '/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/discovery/til/1_cws_tiling' #input
    file_name_pattern = '*.svs'
    files = sorted(glob(os.path.join(data_dir, file_name_pattern)))
    for file in files:
        sub_dir_name = os.path.basename(file)
        print(sub_dir_name)
        Save_Classification_Output(data_dir=data_dir,
                                   tme_dir=tme_dir,
                                   results_dir=results_dir,
                                   sub_dir_name=sub_dir_name,
                                   csv_classification_dir=csv_classification_dir,
                                   csv_correction_dir=csv_correction_dir,
                                   color_code_path=os.path.join('/rsrch5/home/trans_mol_path/xpan7/pipelines/artemis', 'colorcodes','HE_Fib_Lym_Tum_Others.csv')
                                   )