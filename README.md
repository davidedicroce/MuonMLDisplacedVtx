
# MuonMLDisplacedVtx

Utilities and training scripts for displaced-vertex machine learning studies in the ATLAS muon spectrometer.

This repository provides an end-to-end workflow:
- convert ROOT dumps to graph datasets in HDF5,
- create deterministic train/validation splits,
- train distributed GNN regressors for displaced vertex position.

## What The Code Does

The project builds graph-based event representations for displaced-vertex regression.

- Event-level target:
	- Cartesian pipeline: predict y_vertex = [x, y, z] in meters.
	- Polar pipeline: predict y_vertex = [r, theta, phi] (r in meters, angles in radians).
- Nodes:
	- Muon segment nodes and calorimeter tower nodes.
- Edges:
	- Directed segment<->tower edges for geometrically compatible pairs.
- Training:
	- Multi-GPU DistributedDataParallel (DDP) training with optional W and B logging.

## Main Files

- DisplacedVertex_converter.py
	- Converts ROOT files into Cartesian HDF5 graph datasets under ./data.
- DisplacedVertex_polar_converter.py
	- Converts ROOT files into polar-coordinate HDF5 graph datasets under ./data_polar.
- DisplacedVertex_splitter.py
	- Creates deterministic train/val split files for Cartesian datasets.
- DisplacedVertex_polar_splitter.py
	- Creates deterministic train/val split files for polar datasets.
- train_DisplacedVertex_position.py
	- Trains the Cartesian regression model with DDP.
- train_DisplacedVertex_polar_position.py
	- Trains the polar regression model with DDP.

## Prerequisites

- Python 3.10+ recommended
- One or more CUDA GPUs for training
- Python packages:
	- numpy
	- h5py
	- uproot
	- torch
	- wandb

Install dependencies (example):

```bash
python -m pip install numpy h5py uproot torch wandb
```

## Expected Input Layout

ROOT inputs should be available under a directory similar to:

```text
hdd_data/
	MuonBucketDump_H*/outputs/MuonBucketDump_group.det-muon.*root
```

You can adapt the pattern with --input-dir and --pattern.

## Run Sequence

Choose one pipeline (Cartesian or Polar). The order is always:

1. Convert ROOT -> HDF5 graph parts
2. Create deterministic split file
3. Train model with torchrun

### A) Polar Pipeline (recommended for polar training script)

1. Convert to polar dataset

```bash
python -u DisplacedVertex_polar_converter.py \
	--input-dir hdd_data/ \
	--pattern "MuonBucketDump_H*/outputs/MuonBucketDump_group.det-muon.*root" \
	--output-dir ./data_polar \
	--output-name displaced_vertex_dataset \
	--vertex-r-max-mm 8000.0 \
	--vertex-z-max-mm 12000.0 \
	--calo-r-max-mm 4250 \
	--calo-z-max-mm 6500 \
	--min-tower-energy-mev 1000 \
	--max-tower-segment-dr 0.4
```

2. Build train/val split

```bash
python DisplacedVertex_polar_splitter.py \
	--data-glob "./data_polar/displaced_vertex_dataset_part*.h5" \
	--val-fraction 0.1 \
	--seed 12345 \
	--out ./data_polar/split_displaced_vertex_polar_seed12345.npz
```

3. Train polar model (multi-GPU)

```bash
torchrun --standalone --nproc_per_node=8 train_DisplacedVertex_polar_position.py \
	--data-glob "./data_polar/displaced_vertex_dataset_part*.h5" \
	--split-file "./data_polar/split_displaced_vertex_polar_seed12345.npz" \
	--epochs 200 \
	--lr 2e-4 \
	--hidden-dim 128 \
	--layers 4 \
	--dropout 0.1 \
	--layer-type mpnn \
	--num-workers 4 \
	--pin-memory \
	--wandb \
	--wandb-project "DisplacedVertex" \
	--wandb-name "dv_polar_regression_v1" \
	--early-stop \
	--fourier \
	--weight-decay 0.02 \
	--edge-dropout 0.1 \
	--feat-noise-std 0.01 \
	--ema \
	--ema-decay 0.999 \
	--save "./models/displaced_vertex_polar_gnn.pt"
```

### B) Cartesian Pipeline

1. Convert to Cartesian dataset

```bash
python -u DisplacedVertex_converter.py \
	--input-dir hdd_data/ \
	--pattern "MuonBucketDump_H*/outputs/MuonBucketDump_group.det-muon.*root" \
	--output-dir ./data \
	--output-name displaced_vertex_dataset \
	--vertex-r-max-mm 8000.0 \
	--vertex-z-max-mm 12000.0 \
	--calo-r-max-mm 4250 \
	--calo-z-max-mm 6500 \
	--min-tower-energy-mev 1000 \
	--max-tower-segment-dr 0.4
```

2. Build train/val split

```bash
python DisplacedVertex_splitter.py \
	--data-glob "./data/displaced_vertex_dataset_part*.h5" \
	--val-fraction 0.1 \
	--seed 12345 \
	--out ./data/split_displaced_vertex_seed12345.npz
```

3. Train Cartesian model (multi-GPU)

```bash
torchrun --standalone --nproc_per_node=8 train_DisplacedVertex_position.py \
	--data-glob "./data/displaced_vertex_dataset_part*.h5" \
	--split-file "./data/split_displaced_vertex_seed12345.npz" \
	--epochs 200 \
	--lr 2e-4 \
	--hidden-dim 128 \
	--layers 4 \
	--dropout 0.1 \
	--layer-type mpnn \
	--num-workers 4 \
	--pin-memory \
	--wandb \
	--wandb-project "DisplacedVertex" \
	--wandb-name "dv_regression_v1" \
	--early-stop \
	--fourier \
	--weight-decay 0.02 \
	--edge-dropout 0.1 \
	--feat-noise-std 0.01 \
	--ema \
	--ema-decay 0.999 \
	--save "./models/displaced_vertex_gnn.pt"
```
