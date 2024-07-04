import numpy as np
import pickle
import os
from glob import glob
import cv2
import re



cws_folder = os.path.normpath('/rsrch6/home/trans_mol_path/yuan_lab/TIER2/anthracosis/never_smoker_multi/1_cws_tiling')
annotated_dir = '/rsrch6/home/trans_mol_path/yuan_lab/TIER2/anthracosis/never_smoker_multi/AIgrading/mit-b3-finetuned-gp-e60-lr00001-s512-20x768/mask_cws768'
output_dir = "/rsrch6/home/trans_mol_path/yuan_lab/TIER2/anthracosis/never_smoker_multi/AIgrading/mit-b3-finetuned-gp-e60-lr00001-s512-20x768/mask_ss1"
scale_factor = 0.0615
annotated_files = sorted(glob(os.path.join(annotated_dir, '*.svs')))
if os.path.exists(output_dir) is False:
    os.makedirs(output_dir)

def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='='):  # chr(0x00A3)
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration / total)
    bar = fill * filledLength + '>' + '.' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end="")
    # Print New Line on Complete
    if iteration == total:
        print()


for i in range(0, len(annotated_files)):
    wsi_name = os.path.split(annotated_files[i])[-1]
    if os.path.exists(os.path.join(output_dir, wsi_name + "_Ss1.png")):
        print(wsi_name, 'exists')
        pass
    else:
        # print(wsi_name)
        filename = os.path.split(annotated_files[i])[-1]

        param = pickle.load(open(os.path.join(cws_folder, filename, 'param.p'), 'rb'))
        ss1 = cv2.imread(os.path.join(cws_folder, filename, 'Ss1.jpg'))

        slide_dimension = np.array(param['slide_dimension']) / param['rescale']

        slide_w, slide_h = slide_dimension
        cws_w, cws_h = param['cws_read_size']

        divisor_w = np.ceil(slide_w / cws_w)
        divisor_h = np.ceil(slide_h / cws_h)

        w, h = int(slide_w * scale_factor), int(slide_h * scale_factor)
        print('\n%s, Target size: %i,%i, level_2 size: %i,%i' % (os.path.basename(filename), w, h, ss1.shape[1], ss1.shape[0]))
        img_all = np.zeros((max(h, ss1.shape[0])+1, max(ss1.shape[1],w)+1, 3))
        # img_all = np.zeros((w,h,3))

        drivepath, imagename = os.path.split(annotated_files[i])
        annotated_dir_slide = os.path.join(annotated_dir, imagename)
        annotated_path = annotated_dir_slide
        images = sorted(glob(os.path.join(annotated_path, 'Da*')), key=natural_key)
        printProgressBar(0, len(images), prefix='Progress:', suffix='Complete', length=50)
        i_count = 0
        for ii in images:
            ii = os.path.basename(ii)
            #print(ii)
            i_count += 1

            cws_i = int(re.search(r'\d+', ii).group())
            h_i_ori = int(np.floor(cws_i / divisor_w)) * cws_h
            w_i = int((cws_i - h_i_ori / cws_h * divisor_w)) * int(cws_w * scale_factor)

            h_i = int(np.floor(cws_i / divisor_w)) * int(cws_h * scale_factor)
            # h_i = int(np.floor(h_i * d_y))
            # w_i = int(np.floor(w_i * d_x))
            # print(cws_i, w_i, h_i)

            img = cv2.imread(os.path.join(annotated_path, ii))
            #print(os.path.join(annotated_path, ii))
            # check if dimension of cws matches the actual value
            if i_count == 1:
                cws_real_h = img.shape[0]
                cws_real_w = img.shape[1]

            cws_divisor_h = cws_h / img.shape[0]
            cws_divisor_w = cws_w / img.shape[1]

            w_r = max(int(img.shape[1] * scale_factor), 1)
            h_r = max(int(img.shape[0] * scale_factor), 1)

            img_r = cv2.resize(img, (w_r, h_r), interpolation=cv2.INTER_NEAREST)

            img_all[h_i: h_i + int(img_r.shape[0]), w_i: w_i + int(img_r.shape[1]), :] = img_r

            printProgressBar(cws_i, len(images), prefix='Progress:',
                             suffix='Completed for %s' % i, length=50)

            # if w_i + cws_w / divider > w:
            if i_count % 20 == 0 or i_count >= len(images):
                # print("save", w_i)
                cv2.imwrite(os.path.join(output_dir, imagename + "_Ss1.png"), img_all)

