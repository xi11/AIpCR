import os
import numpy as np
from glob import glob
import cv2
import pandas as pd
from PIL import Image
#V1 is to consicer true_sum: true_i.sum() + pred_i.sum()
#V2 is to consider only true_i.sum()
#V3 is to add smooth term; if not full class for that image, then it will be 0 for the lacking class, which should not
#V4 is to add smooth term and then use true_i.sum() as a guide for existing class in the image; similar with V1
#V1 is to consicer true_sum: true_i.sum() + pred_i.sum()
def dice_coefficient_multi(true, pred, num_classes):
    dice = np.zeros(num_classes)
    smooth = 0.001
    for i in range(num_classes):
        true_i = (true == i)
        pred_i = (pred == i)
        intersection = np.logical_and(true_i, pred_i).sum()
        true_sum = true_i.sum() + pred_i.sum()
        if true_sum == 0:
            dice[i] = 1.0  # If there are no true pixels for this class, set Dice to 1, then need to remove from the averaging.
        else:
            dice[i] = 2 * intersection / (true_sum + smooth)
    return dice


color_mapping = {
    (128, 0, 0): 1,
    (255, 255, 0): 2,
    (255, 0, 0): 3,
    (255, 0, 255): 4,
    (128, 128, 0): 5,
    (0, 255, 255): 6,
    (0, 0, 255): 7
}
nClasses = 8
pred_mask = '/Volumes/xpan7/tmesegK8/patch896_digital_025all_blood/testBCSS_4tnbc/test_square/pred_mask_bloodMerge896str224_E2H100'
gt_mask = '/Users/xiaoxipan/Documents/project/public_data/bcss/mask_digital'
data_row = []

files = sorted(glob(os.path.join(pred_mask, '*.png')))
for file in files:
    file_name = os.path.basename(file)
    dst_file = os.path.join(gt_mask, file_name)
    img = np.array(Image.open(file))
    # Initialize an empty array for labels
    pred = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    # Map colors to labels
    for key, value in color_mapping.items():
        # Find all pixels matching the current color and set the corresponding label
        pred[(img == key).all(axis=-1)] = value

    #pred[pred == 6] = 0
    #pred[pred == 7] = 0
    gt = np.array(Image.open(dst_file))
    gt[gt == 8] = 4
    #gt[gt == 6] = 0
    #gt[gt == 7] = 0
    dice_score = dice_coefficient_multi(gt, pred, nClasses)
    print(dice_score)

    data = {'file_name': file_name}
    for i in range(nClasses):
        data[f'class{i}_dice'] = dice_score[i]

    data['img_shape0'] = img.shape[0]
    data['img_shape1'] = img.shape[1]
    data['img_pix'] = img.shape[0] * img.shape[1]
    data['class4_pix'] = np.sum(gt == 4)
    data['class4_per'] = np.sum(gt == 4)*1.0 / (img.shape[0] * img.shape[1])
    data['class5_pix'] = np.sum(gt == 5)
    data['class5_per'] = np.sum(gt == 5) * 1.0 / (img.shape[0] * img.shape[1])
    data_row.append(data)

df = pd.DataFrame(data_row)
df.to_csv('/Volumes/xpan7/tmesegK8/patch896_digital_025all_blood/testBCSS_4tnbc/test_square/tmesegformer_bloodNec896str224_E2H100V5github_test_size_pixel.csv', index=False)
