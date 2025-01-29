import os
import pandas as pd
from glob import glob

src_path = '/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/discovery/til/4_cell_class_segformerBRCAartemis/CellScore_areaSum/combined'
tme_path = '/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/discovery/mit-b3-finetunedBRCA-Artemis-e60-lr00001-s512-20x512/discovery_post_tme_pix.xlsx'
dst_path = '/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/discovery/til/4_cell_class_segformerBRCAartemis/CellScore_areaSum'

tme_score = pd.read_excel(tme_path)
tme_score['stroma_mm2'] = tme_score['stroma_pix'] * 16 * 16 * 0.44 * 0.44 * 1e-6

combined_results = pd.DataFrame()

files = sorted(glob(os.path.join(src_path, '*.csv')))
for idx, file in enumerate(files, start=2):
    # Read the relevant columns
    df = pd.read_csv(file, usecols=['FileName', '#l_stroma'])
    df.rename(columns={'FileName': 'ID', '#l_stroma': 'lym_stroma'}, inplace=True)
    df = df.merge(tme_score[['ID', 'stroma_mm2']], on='ID', how='inner')
    df[f'ai-stil-area{idx}'] = df['lym_stroma'] / df['stroma_mm2']
    combined_results = pd.concat([combined_results, df[['ID', f'ai-stil-area{idx}']]], axis=0)

# Pivot the combined results to get the desired format
combined_results = combined_results.pivot_table(index='ID', values=[col for col in combined_results.columns if 'ai-stil' in col], aggfunc='first').reset_index()

# Save the combined results to a CSV file
output_file = os.path.join(dst_path, 'combined_ai_stil_scores.csv')
combined_results.to_csv(output_file, index=False)
