#!/usr/bin/env python3
"""
DisplacedVertex_converter.py

Convert ROOT files into per-file HDF5 graph-classification data.
One HDF5 file is produced per ROOT input file, and existing outputs are skipped,
following the same high-level workflow as Bucket_converter.py.

Graph label:
  y = 0  background
  y = 1  signal, only when the ROOT filename matches any configured signal pattern and at
         least one truth vertex in the event satisfies 800 mm < rho < 8000 mm.

Node features keep the cylindrical converter convention:
  [r, theta_pos, phi_pos, theta_dir, phi_dir, energy_like, nCells_or_DoF]

Example:
python -u DisplacedVertex_converter.py \
    --dir-in /media/hdd/ddicroce/MuonBucketDump_displacedVtx \
    --dir-out /eos/project-f/fcc-ml/ddicroce/ATLAS_MuonSpectrometer/data/data_segments_mu200_r4_graphs \
    --min-tower-energy-mev 1000 \
    --max-tower-segment-dr 0.4 \
    2>&1 | tee log_displacedvertex_converter.txt
"""

import argparse

from dv_converter_utils import (
    add_dv_classification_converter_args,
    run_dv_directory_conversion,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ROOT files to DisplacedVertex HDF5 graphs for graph-level classification."
    )
    add_dv_classification_converter_args(parser)
    args = parser.parse_args()
    run_dv_directory_conversion(args)


if __name__ == "__main__":
    main()
