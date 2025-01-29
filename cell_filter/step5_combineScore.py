import os
import pandas as pd

# Specify the directory containing the CSV files

ouput_dir = '/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/discovery/til/4_cell_class_segformerBRCAartemis/CellScore_areaContour/combined'
if not os.path.exists(ouput_dir):
    os.makedirs(ouput_dir)

for i in range(2, 21):
    input_dir = f'/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/discovery/til/4_cell_class_segformerBRCAartemis/CellScore_areaContour/CellScoreOth_area{i}'
    # Initialize an empty list to hold the dataframes
    df_list = []

    # Loop through all files in the directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_dir, filename)
            try:
                df = pd.read_csv(file_path)
                df_list.append(df)
                print(f'Successfully read {filename}')
            except Exception as e:
                print(f'Error reading {filename}: {e}')

    # Concatenate all dataframes in the list
    combined_df = pd.concat(df_list, ignore_index=True)

    # Save the combined dataframe to a new CSV file
    combined_df.to_csv(os.path.join(ouput_dir,f'combined_scoreOther_segformerBRCAartemis_area{i}.csv'), index=False)

