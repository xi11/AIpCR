import json
import os
from collections import OrderedDict
import pandas as pd
from glob import glob

def combine_json_files_2_csv(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    combined_data = pd.DataFrame()
    files = sorted(glob(os.path.join(input_dir, '*.json')))
    for file in files:
        file_name = os.path.basename(file)
        print(file_name)
        #if not file_name.endswith('.json'):
            #continue

        with open(os.path.join(input_dir, file_name), 'r') as fp:
            scores = json.load(fp)
        # scores_expanded = dict()
        scores_expanded = OrderedDict()
        scores_expanded['file_name'] = scores['file_name']
        for key, values in scores.items():
            if key == 'file_name':
                continue
            if isinstance(values, dict):
                scores_expanded = {**scores_expanded, **values}
                #print(scores_expanded)
            else:
                scores_expanded[key] = values
        # scores_ = {key: [value] for key, value in scores_expanded.items()}
        #type(scores_expanded)
        combined_data = pd.concat([combined_data, pd.DataFrame([scores_expanded])], ignore_index=True)
        #combined_data = combined_data.append(scores_expanded, ignore_index=True, sort=False)
    columns = list(scores_expanded.keys())
    #combined_data = combined_data[columns]
    combined_data.to_csv(os.path.join(output_dir, 'combined_scores_v3.csv'), index=False)
    print(combined_data.head())


if __name__ == '__main__':
    input_dir = r'Z:\TIER2\artemis_lei\discovery\til\step5_spatial_refineV3_thFF'
    output_dir = r'Z:\TIER2\artemis_lei\discovery\til\step6_combined_refineV3_thFF'
    combine_json_files_2_csv(input_dir=input_dir, output_dir=output_dir)
