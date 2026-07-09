import os
from glob import glob
from pathlib import Path
import numpy as np
import argparse
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
device = "cuda:0"


def run_hovernet(data_dir, out_base, file_pattern='*.svs', nfile=0):
    out_base = Path(out_base)   # ✅ FIX
    out_base.mkdir(parents=True, exist_ok=True)

    file = sorted(glob(os.path.join(data_dir, file_pattern)))[nfile]
    svs_path = Path(file)
    file_name = svs_path.stem
    out_dir = out_base / file_name

    if not os.path.exists(out_dir):
        wsi = WSIReader.open(str(svs_path))
        segmentor = NucleusInstanceSegmentor(
            pretrained_model="hovernet_fast-pannuke",  # change if you use a different one
            num_loader_workers=32,
            num_postproc_workers=32,
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


# get arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_dir', dest='data_dir', help='path to cws data')
parser.add_argument('-o', '--save_dir', dest='save_dir', help='path to save all output files', default=None)
parser.add_argument('-p', '--pattern', dest='file_name_pattern', help='pattern in the files name', default='*.ndpi')
parser.add_argument('-n', '--nfile', dest='nfile', help='the nfile-th file', default=150, type=int)
parser.add_argument('-nJ', '--number_pods', dest='nJob', help='how many pods to be used in K8s', default=32, type=int)
args = parser.parse_args()

datapath=args.data_dir
nfile=args.nfile
file_pattern=args.file_name_pattern
files = sorted(glob(os.path.join(datapath, file_pattern)))
njob = args.nJob


if len(files) <= 8:
        start_file = nfile
        end_file = nfile + 1
else:
        file_job = len(files) // njob +1
        start_file = nfile * file_job
        end_file = nfile * file_job + file_job

for i in range(start_file, end_file):
    run_hovernet(args.data_dir, args.save_dir, file_pattern=args.file_name_pattern, nfile=i) #before is nfile=args.nfile


