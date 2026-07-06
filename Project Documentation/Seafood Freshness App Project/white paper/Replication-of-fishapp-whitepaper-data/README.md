# Replication of FishApp White Paper Data

## Overview

This repository contains the datasets, notebooks, scripts, generated visualizations, and supporting documents used to reproduce the experiments and figures presented in the FishApp white paper for sardine and mackerel freshness assessment.

The repository serves as a reproducible reference for the complete experimental workflow, including:

* Fish segmentation
* Image preprocessing
* Cut (damage) detection
* CNN-based freshness classification
* Model architecture visualization
* Experimental notebooks
* White paper assets and generated results

---

## Repository Structure

```text
Replication-of-fishapp-whitepaper-data
│
├── README.md
├── requirements.txt
├── requirements_macos.txt
├── SETUP.md
│
├── docs/
│   ├── Sardine and Mackerel White paper.docx
│   └── Sardine and Mackerel White paper.pdf
│
├── notebooks/
│   └── prod test sardine&mackerel.ipynb
│
├── notebook-results/
│   ├── prod_flow_mackerel_results.csv
│   ├── prod_flow_mackerel_results_after_rounding.csv
│   ├── prod_flow_sardine_results.csv
│   └── prod_flow_sardine_results_after_rounding.csv
│
└── white-paper-assets/
    ├── sample-inputs/
    ├── images/
    ├── generated-results/
    ├── segment_fish.py
    ├── highlight_cuts.py
    ├── visualize.py
    └── plot_model_architectures.py
```

---

## Components

### notebooks/

Contains the Jupyter notebooks used during experimentation and validation.

### notebook-results/

Contains exported CSV files generated from notebook execution.

### docs/

Contains the final white paper in DOCX and PDF formats.

### white-paper-assets/

Contains all scripts, sample images, generated visualizations, and supporting resources used throughout the experiments.

#### sample-inputs/

Example input images used for inference demonstrations.

#### images/

Images used in the white paper, including preprocessing visualizations, architecture diagrams, and damage detection illustrations.

#### generated-results/

Outputs produced by the provided scripts.

---

## Python Scripts

| Script                        | Description                                                           |
| ----------------------------- | --------------------------------------------------------------------- |
| `segment_fish.py`             | Performs fish segmentation using the trained segmentation model.      |
| `highlight_cuts.py`           | Detects damaged regions and highlights cuts on segmented fish images. |
| `visualize.py`                | Generates preprocessing and comparison visualizations.                |
| `plot_model_architectures.py` | Produces model architecture diagrams.                                 |

---

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

For macOS:

```bash
pip install -r requirements_macos.txt
```

---

## Reproducing the Experiments

1. Install the required dependencies.
2. Open the notebook in the `notebooks` directory.
3. Execute the notebook sequentially.
4. Generated outputs will be stored in the corresponding results directories.

---

## Notes

This repository accompanies the FishApp white paper and is intended for reproducing the reported experiments and visualizations.


