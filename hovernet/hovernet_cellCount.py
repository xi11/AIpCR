#!/usr/bin/env python3

import os
import joblib
import pandas as pd
from collections import Counter
import numpy as np

# -----------------------------
# HoverNet class mapping
# -----------------------------
CELL_CLASSES = {
    0: "background",
    1: "neoplastic_epithelial",
    2: "Inflammatory",
    3: "Connective",
    4: "Dead",
    5: "non_neoplastic_epithelial",
}

# -----------------------------
# Core counting logic
# -----------------------------
def count_cells_from_dat(dat_path, skip_background=True):
    """
    Count nuclei per class from a HoverNet .dat file.
    Returns Counter {class_name: count}.
    """
    preds = joblib.load(dat_path)
    if not isinstance(preds, dict):
        raise ValueError(f"{dat_path} does not contain a dict")

    counter = Counter()

    for nuc in preds.values():
        if not isinstance(nuc, dict):
            continue

        cls_id = int(nuc.get("type", -1))
        if skip_background and cls_id == 0:
            continue

        cls_name = CELL_CLASSES.get(cls_id, "unknown")
        counter[cls_name] += 1

    return counter


# -----------------------------
# Discover slide folders
# -----------------------------
def find_slide_folders(parent_dir, dat_name="0.dat"):
    """
    Return list of (slide_id, dat_path) for all subfolders containing dat_name.
    """
    slide_entries = []

    for entry in sorted(os.listdir(parent_dir)):
        slide_dir = os.path.join(parent_dir, entry)
        dat_path = os.path.join(slide_dir, dat_name)

        if os.path.isdir(slide_dir) and os.path.isfile(dat_path):
            slide_entries.append((entry, dat_path))
        else:
            print(slide_dir)
    return slide_entries


# -----------------------------
# Aggregate all slides
# -----------------------------
def aggregate_all_slides(parent_dir):
    rows = []

    slide_entries = find_slide_folders(parent_dir)
    if not slide_entries:
        raise RuntimeError(f"No HoverNet outputs found in {parent_dir}")

    for slide_id, dat_path in slide_entries:
        counts = count_cells_from_dat(dat_path)

        inflammatory = counts.get("Inflammatory", 0)
        connective = counts.get("Connective", 0)

        denom = inflammatory + connective
        infl_ratio = inflammatory / denom if denom > 0 else np.nan

        row = {
            "ID": slide_id,
            "Inflammatory_count": inflammatory,
            "Connective_count": connective,
            "neoplastic_epithelial_count": counts.get("neoplastic_epithelial", 0),
            "non_neoplastic_epithelial_count": counts.get("non_neoplastic_epithelial", 0),
            "Dead_count": counts.get("Dead", 0),
            "Inflammatory_ratio": infl_ratio,
        }

        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":

    parent_hovernet_dir = (
        "/rsrch6/home/trans_mol_path/yuan_lab/TIER2/"
        "artemis_lei/TransNeo_Nature/hovernet"
    )

    out_csv = os.path.join(
        parent_hovernet_dir,
        "hovernet_cell_counts_all_slides.csv"
    )

    df = aggregate_all_slides(parent_hovernet_dir)
    df.to_csv(out_csv, index=False)

    print(f"Aggregated {len(df)} slides")
    print(f"Saved CSV to:\n{out_csv}")
