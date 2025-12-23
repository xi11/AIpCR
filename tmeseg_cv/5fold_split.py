import pandas as pd
from sklearn.model_selection import GroupKFold
from pathlib import Path

# ------------------------
# Input paths
# ------------------------
csv1_path = "/rsrch5/home/trans_mol_path/xpan7/tmesegK8/patch512artemis/tmesegformer_test4discovery11slide_size_pixel.csv"
csv2_path = "/rsrch5/home/trans_mol_path/xpan7/tmesegK8/patch512artemis/tmesegformer_test4validation15slide_size_pixel.csv"
out_dir = Path("/rsrch5/home/trans_mol_path/xpan7/pipelines/artemis/eval/five_folds")
out_dir.mkdir(exist_ok=True)

# ------------------------
# Load CSVs
# ------------------------
df1 = pd.read_csv(csv1_path)
df2 = pd.read_csv(csv2_path)

# Add source label (optional but useful)
df1["source"] = "csv1"
df2["source"] = "csv2"

# Combine
df = pd.concat([df1, df2], ignore_index=True)

# ------------------------
# Derive sample ID
# ------------------------
df["sample_id"] = df["file_name"].str[:2]

# Sanity check
print("Number of samples:", df["sample_id"].nunique())
print("Total patches:", len(df))

# ------------------------
# Group-wise 5-fold split
# ------------------------
gkf = GroupKFold(n_splits=5)

X = df.index  # dummy feature
groups = df["sample_id"]

for fold, (_, test_idx) in enumerate(gkf.split(X, groups=groups), start=1):
    fold_df = df.iloc[test_idx]

    out_path = out_dir / f"fold_{fold}.csv"
    fold_df.to_csv(out_path, index=False)

    print(
        f"Fold {fold}: "
        f"{fold_df['sample_id'].nunique()} samples, "
        f"{len(fold_df)} patches → {out_path}"
    )
