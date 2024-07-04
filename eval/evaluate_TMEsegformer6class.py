import os
import numpy as np
from glob import glob
import pandas as pd
from PIL import Image
#V1 is to consicer true_sum: true_i.sum() + pred_i.sum() with smooth term
#V2 is to consider only true_i.sum() with smoother term

def dice_coefficient_multi(true, pred, num_classes):
    dice = np.zeros(num_classes)
    smooth = 0.001
    for i in range(num_classes):
        true_i = (true == i)
        pred_i = (pred == i)
        intersection = np.logical_and(true_i, pred_i).sum()
        true_sum = true_i.sum() + pred_i.sum()
        if true_sum == 0:
            dice[i] = 1.0  # If there are no true pixels for this class, set Dice to 1
        else:
            dice[i] = 2 * intersection / (true_sum + smooth)
    return dice


color_mapping = {
    (128, 0, 0): 1,
    (255, 255, 0): 2,
    (0, 255, 255): 3,
    (255, 0, 255): 4,
    (128, 128, 0): 5,
    
}
nClasses = 6
pred_mask = '/rsrch5/home/trans_mol_path/xpan7/tmesegK8/patch512artemis/result_test_mit-b3-finetunedBRCA-Artemis4Eval-s512-20x512'
gt_mask = '/rsrch5/home/trans_mol_path/xpan7/tmesegK8/patch512artemis/maskPng_test'
data_row = []

files = sorted(glob(os.path.join(pred_mask, '*.png')))
for file in files:
    file_name = os.path.basename(file)
    dst_file = os.path.join(gt_mask, 'mask_'+file_name)
    img = np.array(Image.open(file))
    pred = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    # Map colors to labels
    for key, value in color_mapping.items():
        pred[(img == key).all(axis=-1)] = value

    gt = np.array(Image.open(dst_file))
    dice_score = dice_coefficient_multi(gt, pred, nClasses)
    print(dice_score)

    data = {'file_name': file_name}
    for i in range(nClasses):
        data[f'class{i}_dice'] = dice_score[i]

    data['img_shape0'] = img.shape[0]
    data['img_shape1'] = img.shape[1]
    data['img_pix'] = img.shape[0] * img.shape[1]
    data['class1_pix'] = np.sum(gt == 1)
    data['class1_per'] = np.sum(gt == 1)*1.0 / (img.shape[0] * img.shape[1])
    data['class2_pix'] = np.sum(gt == 2)
    data['class2_per'] = np.sum(gt == 2)*1.0 / (img.shape[0] * img.shape[1])
    data['class3_pix'] = np.sum(gt == 3)
    data['class3_per'] = np.sum(gt == 3) * 1.0 / (img.shape[0] * img.shape[1])
    data['class4_pix'] = np.sum(gt == 4)
    data['class4_per'] = np.sum(gt == 4)*1.0 / (img.shape[0] * img.shape[1])
    data['class5_pix'] = np.sum(gt == 5)
    data['class5_per'] = np.sum(gt == 5) * 1.0 / (img.shape[0] * img.shape[1])
    data_row.append(data)

df = pd.DataFrame(data_row)
df.to_csv('/rsrch5/home/trans_mol_path/xpan7/tmesegK8/patch512artemis/tmesegformer_test4discovery11slide_size_pixel.csv', index=False)
