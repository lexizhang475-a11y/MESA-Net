# MESA-Net 

This repository is a  release for **MESA-Net**, containing only the main model, the training code, and the cross-dataset evaluation code.

## What is included

- **Main model only**: `model/mesa_net.py`
- **MESA loss**: `losses/mesa_loss.py`
- **Complete training code** on **Kvasir-SEG**: `scripts/train.py`
- **Evaluation code** on **ClinicDB / ColonDB / PolypGen**: `scripts/eval.py`
- **Dataset loaders**: `datasets/polyp_datasets.py`
- **Utilities**: metrics and augmentation helpers

## Directory structure

```text
MESA-Net_minimal_release/
├── LICENSE
├── README.md
├── requirements.txt
├── configs/
│   └── mesa_net.yaml
├── datasets/
│   ├── __init__.py
│   └── polyp_datasets.py
├── losses/
│   ├── __init__.py
│   └── mesa_loss.py
├── model/
│   ├── __init__.py
│   └── mesa_net.py
├── scripts/
│   ├── eval.py
│   └── train.py
└── utils/
    ├── __init__.py
    ├── metrics.py
    └── random_lightspot.py
```

## Component naming aligned with the paper

The code uses Python-safe identifiers while preserving paper terminology:

- **MESA-Net** → `MESANet`
- **GAB (Ghost Axial Block)** → `GhostAxialBlock`
- **AFM** → `AttentionFusionModule`
- **Lite-AFM** → `LiteAttentionFusionModule`
- **Aligner** → `DeepSupervisionAligner`
- **MESA Loss** → `MESALoss`

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset preparation

Datasets are **not** distributed in this repository. Please download them yourself and arrange them as follows.

### 1. Kvasir-SEG (training)

Expected structure:

```text
data/Kvasir-SEG/
├── images/
└── masks/
```

Example download command:

```bash
mkdir -p data
wget -O data/kvasir-seg.zip https://datasets.simula.no/downloads/kvasir-seg.zip
unzip data/kvasir-seg.zip -d data/
```

### 2. ClinicDB (evaluation)

Expected structure:

```text
data/CVC-ClinicDB/
├── images/
└── masks/
```

### 3. ColonDB (evaluation)

Expected structure:

```text
data/CVC-ColonDB/
├── images/
└── masks/
```

### 4. PolypGen (evaluation)

Expected structure:

```text
data/PolypGen/
├── seq01/
│   ├── images/
│   └── masks/
├── seq02/
│   ├── images/
│   └── masks/
└── ...
```

Notes:

- The release does **not** include the easy/hard sequence split used in some internal analysis.
- PolypGen is evaluated as a whole dataset.

## Configuration

Edit `configs/mesa_net.yaml` and make sure the dataset paths match your local layout.

Default paths:

```yaml
data:
  kvasir_images_dir: data/Kvasir-SEG/images
  kvasir_masks_dir: data/Kvasir-SEG/masks
  clinicdb_images_dir: data/CVC-ClinicDB/images
  clinicdb_masks_dir: data/CVC-ClinicDB/masks
  colondb_images_dir: data/CVC-ColonDB/images
  colondb_masks_dir: data/CVC-ColonDB/masks
  polypgen_root_dir: data/PolypGen
```

## Training

Train on **Kvasir-SEG** with degraded training inputs and aligned auxiliary supervision:

```bash
python scripts/train.py --config configs/mesa_net.yaml
```

The script saves outputs under the directory defined by:

```yaml
train:
  save_dir: outputs
```

It writes:

- `outputs/<experiment_name>/checkpoints/best.pth`
- `outputs/<experiment_name>/checkpoints/last.pth`
- `outputs/<experiment_name>/history.json`
- `outputs/<experiment_name>/used_config.yaml`

## Evaluation

Evaluate a trained checkpoint on all three cross-dataset benchmarks:

```bash
python scripts/eval.py --config configs/mesa_net.yaml --checkpoint outputs/mesa_net_minimal_release/checkpoints/best.pth --dataset all
```

Evaluate a single dataset:

```bash
python scripts/eval.py --config configs/mesa_net.yaml --checkpoint outputs/mesa_net_minimal_release/checkpoints/best.pth --dataset clinicdb
python scripts/eval.py --config configs/mesa_net.yaml --checkpoint outputs/mesa_net_minimal_release/checkpoints/best.pth --dataset colondb
python scripts/eval.py --config configs/mesa_net.yaml --checkpoint outputs/mesa_net_minimal_release/checkpoints/best.pth --dataset polypgen
```

Optionally evaluate ClinicDB / ColonDB with degraded inputs:

```bash
python scripts/eval.py --config configs/mesa_net.yaml --checkpoint outputs/mesa_net_minimal_release/checkpoints/best.pth --dataset all --use-degraded-input
```

## Training objective implemented in code

`MESALoss` implements:

- **Segmentation supervision**: `BCEWithLogits + global Dice`
- **Deep supervision** with aligned auxiliary heads
- **Boundary consistency** via Sobel-based edge maps
- **BCE-based self-distillation** with temperature scaling

Default loss weights in `configs/mesa_net.yaml`:

```yaml
loss:
  aux_weights: [0.3, 0.15]
  boundary_weight: 0.05
  distill_weight: 0.1
  temperature: 4.0
```

