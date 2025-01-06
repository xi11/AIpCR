import h5py
import numpy as np

save_path = '/rsrch5/home/trans_mol_path/xpan7/pipelines/CLAM-download/heatmaps/heatmap_pcr_raw_vis1_bwr_testAtten/HEATMAP_OUTPUT/Unspecified/002_HE_A1_primary/002_HE_A1_primary_blockmap.h5'


with h5py.File(save_path, 'r') as file:
    file = h5py.File(save_path, 'r')
    dset = file['attention_scores']
    coord_dset = file['coords']
    scores = dset[:]
    print(scores)
    print(1.0/len(scores))
    quantiles = np.quantile(scores, [ 0.25, 0.5, 0.75, 1])
    print(quantiles)
    coords = coord_dset[:]
'''
with h5py.File(save_path, 'r') as hdf:
    # Print the names of all groups and datasets in the file
    def print_attrs(name, obj):
        print(name)
        for key, value in obj.attrs.items():
            print(f"    {key}: {value}")
    
    # Recursively print the contents of the file
    hdf.visititems(print_attrs)
'''

# if using A (after softmax), then it will really compress the the scores, good thing is you know the threshold, 1/len
# you either do a normalization, still be compressed, but then you don't know the threshold, I guess that's why they choose to use 
# A_raw