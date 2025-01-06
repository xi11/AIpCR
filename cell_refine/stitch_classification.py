import pickle
import numpy as np
from tiles2tiff import tiles2tiff
import os
from glob import glob

def stitch_classification(classification_path, param_file, out_file, ext='png', jpeg_compression=True):
    with open(param_file, 'rb') as p:
        param = pickle.load(p)

    image_size = np.round(np.array(param['slide_dimension']) / param['rescale']).astype(np.int32)
    tile_size = param['cws_read_size'].astype(np.int32)

    tiles2tiff(classification_path, tile_size, image_size, out_file, ext=ext, jpeg_compression=jpeg_compression)


if __name__ == "__main__":
    classification_dir = '/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/discovery/til/4_cell_class_segformerBRCAartemis/annotated_images_tmeseg'
    cws_dir = '/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/discovery/til/1_cws_tiling'
    output_dir = '/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/discovery/til/4_5_stiched_segformerRefine_tmeCell_overlay'
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)

    files = sorted(glob(os.path.join(classification_dir, '*.svs')))
    for file in files:
        file_name = os.path.basename(file)
        param_file = os.path.join(cws_dir, file_name, 'param.p')
        out_file = os.path.join(output_dir, file_name[:-4]+'.tif')
        stitch_classification(file, param_file, out_file, ext='png', jpeg_compression=True)