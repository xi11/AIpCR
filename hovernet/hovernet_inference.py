import os
from pathlib import Path
from glob import glob
import numpy as np
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
device = "cuda:0"

#files = ["015_HE_A2_primary", "019_HE_A2_primary", "020_HE_A1_Primary", "021_HE_A1_Primary", "024_HE_A1_Primary", "031_HE_A1_Primary"]
parent_dir = "/rsrch6/home/trans_mol_path/yuan_lab/TIER1/artemis_lei/IMPRESS_TNBC/HE"
files = sorted(glob(os.path.join(parent_dir, '*.svs')))[10:11]
out_base = Path("/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/IMPRESS_TNBC/hovernet")

for file in files:
    svs_path = Path(file)
    file_name = svs_path.stem   # correct way
    out_dir = out_base / file_name
    if not os.path.exists(out_dir):
        wsi = WSIReader.open(str(svs_path))
        segmentor = NucleusInstanceSegmentor(
            pretrained_model="hovernet_fast-pannuke",  # change if you use a different one
            num_loader_workers=8,
            num_postproc_workers=8,
            batch_size=8,
            auto_generate_mask=True,
            verbose=False,
        )

        preds = segmentor.predict(
            [str(svs_path)],
            masks=None,
            save_dir=str(out_dir),
            mode="wsi",
            device=device,
            crash_on_exception=True,
        )

        print("Done. Outputs saved to:", out_dir)
