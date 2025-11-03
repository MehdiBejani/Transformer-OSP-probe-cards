# Transformer-Based Optimal Sensor Placement for Structural Health Monitoring of Probe Cards

This repository contains code and materials for a **Transformer-based** approach to **Optimal Sensor Placement (OSP)** and mechanical failure detection (baseline / loose screw / crack) on semiconductor **probe cards** using Frequency Response Function (FRF) data. A hybrid **CNN + Transformer** architecture with attention is used to both classify failures and highlight informative sensor locations.

> **Publication status**  
> The paper has been **submitted and is under review**.  
> A **preprint** is available on arXiv: **https://arxiv.org/abs/2509.07603**  
> Please **cite the preprint** for now. We will add the official paper link here after publication.

---

## Table of Contents

- [Highlights](#highlights)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Data](#data)
- [Training & Grid Search](#training--grid-search)
- [Results & Sensor Importance](#results--sensor-importance)
- [Reproducibility Tips](#reproducibility-tips)
- [Cite This Work](#cite-this-work)
- [License](#license)
- [Contact](#contact)

---

## Highlights

- **Hybrid CNN + Transformer** with attention for interpretable **optimal sensor placement** (OSP).
- FRF-based pipeline for probe card structural health monitoring with physics-informed augmentation.
- End-to-end script for data loading, class balancing (e.g., SMOTE), cross-validated training, and hyperparameter grid search.
- Exports metrics, curves, and attention maps to facilitate analysis and reporting.

---

## Repository Structure

```
.
├─ README.md
├─ LICENSE
├─ CITATION.cff
├─ requirements.txt
├─ .gitignore
├─ src/
│  └─ transformer_grid_1.py
├─ data_files/                # (not versioned; see "Data" section)
│  ├─ baseline_data_New.npy
│  ├─ screws_data_New.npy
│  └─ cracks_data_New.npy
├─ results/                   # optional; example logs/plots
│  └─ out_grid_1.txt
└─ docs/
   └─ OSP_Springer_preprint.pdf   # optional; we primarily link to arXiv
```

> **Note:** `data_files/` is intentionally excluded from version control to keep the repository small and to respect data-sharing constraints.

---

## Getting Started

### Prerequisites

- Python **3.9+**
- A machine with a recent CPU; a CUDA-capable GPU is recommended for faster training.

### Installation

```bash
# 1) (Recommended) create a virtual environment
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt
```

---

## Data

This project expects three NumPy arrays in `data_files/`:

- `baseline_data_New.npy`
- `screws_data_New.npy`
- `cracks_data_New.npy`

**Format:** arrays of FRF samples organized consistently across the three classes.  
**Location:** place the `.npy` files inside `./data_files/`.  
**Note:** Raw data are **not** included in the repository.

---

## Training & Grid Search

Run the training script (which performs cross-validation and a small grid search):

```bash
python src/transformer_grid_1.py
```

By default, the script:

- Loads the three `.npy` files and constructs the dataset.
- Optionally balances classes (e.g., via SMOTE) when available.
- Applies physics-aware augmentations during training.
- Performs cross-validation and a grid search over Transformer-related hyperparameters.
- Writes artifacts (metrics, curves, best config) under a results directory (e.g., `results_grid_1/`).
- You may find an example training log in `results/out_grid_1.txt`.

> Adjust hyperparameters and output paths inside `src/transformer_grid_1.py` as needed.

---

## Results & Sensor Importance

The Transformer’s attention can be summarized into **sensor importance** scores that help identify **informative sensor locations** for OSP. The script saves per-fold metrics and plots to aid analysis. You can visualize attention heatmaps or export the learned importance to CSV/NumPy for downstream use.

---

## Reproducibility Tips

- **Seed everything:** set seeds for NumPy, PyTorch, and data loaders.
- **Version pinning:** after a successful run, freeze package versions to a lockfile or update `requirements.txt` with exact versions (e.g., `numpy==...`).
- **Determinism:** consider `torch.use_deterministic_algorithms(True)` (with care for ops that may not support it).
- **Artifacts:** keep all model checkpoints, logs, and configs in `results/` for later comparison.

---

## Cite This Work

For now, **please cite the arXiv preprint**:

```
@misc{bejani2025transformerOSP,
  title         = {Transformer-Based Approach to Optimal Sensor Placement for Structural Health Monitoring of Probe Cards},
  author        = {Mehdi Bejani and Marco Mauri and Daniele Acconcia and Simone Todaro and Stefano Mariani},
  year          = {2025},
  eprint        = {2509.07603},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2509.07603}
}
```

> We will update this README with the **official paper link** after publication.

---

## License

This project is released under the **MIT License**. See [`LICENSE`](./LICENSE) for details.

---

## Contact

For questions or issues, please open a GitHub Issue or contact the authors.  
You’re welcome to suggest improvements via pull requests!
