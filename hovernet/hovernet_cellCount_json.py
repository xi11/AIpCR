#!/usr/bin/env python3

import os
import json
import csv
from glob import glob

# ==========================
# USER INPUT
# ==========================
JSON_DIR = "/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/IMPRESS_HER2/hovernet/086_HE-Ping"      # folder containing *.json
OUTPUT_CSV = "/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/IMPRESS_HER2/hovernet/cell_type_summary_086.csv"  # output CSV path

# ==========================
# CELL TYPE DEFINITIONS
# ==========================
# Matches your color/type logic
TYPE_LABELS = {
    1: "neoplastic",
    2: "inflammatory",
    3: "connective",
    4: "dead",
    5: "non_neoplastic",
}

# ==========================
# COUNT FUNCTION
# ==========================
def count_cell_types(json_path):
    """
    Count cell types from a JSON file where:
    top-level dict = instances
    each value has key 'type'
    """

    with open(json_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Top-level JSON is not a dict: {json_path}")

    counts = {
        "Inflammatory_count": 0,
        "Connective_count": 0,
        "neoplastic_count": 0,
        "non_neoplastic_count": 0,
        "Dead_count": 0,
    }

    for inst_id, inst_info in data.items():
        if not isinstance(inst_info, dict):
            continue

        cell_type = inst_info.get("type", None)

        if cell_type == 2:
            counts["Inflammatory_count"] += 1
        elif cell_type == 3:
            counts["Connective_count"] += 1
        elif cell_type == 1:
            counts["neoplastic_count"] += 1
        elif cell_type == 5:
            counts["non_neoplastic_count"] += 1
        elif cell_type == 4:
            counts["Dead_count"] += 1
        else:
            # unknown or missing type → ignore safely
            continue

    denom = counts["Inflammatory_count"] + counts["Connective_count"]
    counts["Inflammatory_ratio"] = (
        counts["Inflammatory_count"] / denom if denom > 0 else 0.0
    )

    return counts


# ==========================
# MAIN AGGREGATION
# ==========================
def main():
    json_files = sorted(glob(os.path.join(JSON_DIR, "*.json")))

    if len(json_files) == 0:
        raise RuntimeError(f"No JSON files found in {JSON_DIR}")

    fieldnames = [
        "ID",
        "Inflammatory_count",
        "Connective_count",
        "neoplastic_count",
        "non_neoplastic_count",
        "Dead_count",
        "Inflammatory_ratio",
    ]

    rows = []

    for json_path in json_files:
        sample_id = os.path.splitext(os.path.basename(json_path))[0]

        try:
            counts = count_cell_types(json_path)
        except Exception as e:
            print(f"[WARNING] Skipping {sample_id}: {e}")
            continue

        counts["ID"] = sample_id
        rows.append(counts)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] Processed {len(rows)} samples")
    print(f"[DONE] Output written to: {OUTPUT_CSV}")


# ==========================
# ENTRY POINT
# ==========================
if __name__ == "__main__":
    main()