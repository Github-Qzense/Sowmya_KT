# Data Collection

## Overview

This directory contains the notebooks used to collect, organize, validate, and prepare seafood image datasets for the freshness classification project.

The collected data originates from two primary sources:

* **Warehouse Application:** Images captured in seafood warehouses and uploaded to AWS S3 through the data collection application.
* **Manual Collection:** Images manually captured and uploaded for additional data generation and experimentation.

The output of this pipeline serves as the foundation for all subsequent data analysis, model training, and research experiments.

---

## Folder Contents

| File                             | Description                                                                                                     |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `Download_s3_data.ipynb`         | Downloads image datasets from AWS S3 and organizes them locally.                                                |
| `Manual_data.ipynb`              | Processes manually collected images and integrates them into the dataset.                                       |
| `Final Data.ipynb`               | Merges, validates, and prepares the final dataset used for experimentation.                                     |
| `reorganize_S3_Final_Data.ipynb` | Reorganizes downloaded S3 data into the required directory structure.                                           |
| `reason_based_data.ipynb`        | Generates reason-based datasets (e.g., softness, cuts & damage, size) using metadata collected during labeling. |

---

## Directory Contents

### `Download_s3_data.ipynb`

Downloads seafood images from AWS S3 and organizes them into a date-wise directory structure.

**Features**

* Downloads data from AWS S3.
* Organizes images by collection date and species.
* Supports downloading data for specific dates or date ranges.
* Generates species-wise image count reports.
* Excludes manually identified misclassified images.
* Corrects known species name inconsistencies during report generation.

**Requirements**

* `AWS_ACCESS_KEY_ID`
* `AWS_SECRET_ACCESS_KEY`
* `Misclassified_data.txt`

**Output Structure**

```text
download_path/
└── YYYY-MM-DD/
    └── species/
        ├── Good/
        └── Bad/
```

---

### `Manual_data.ipynb`

Processes manually collected images and integrates them into the final training dataset.

**Input Structure**

```text
manual_data/
└── date/
    └── species/
        └── Single or Group/
            ├── Good/
            └── Bad/
```

**Output Structure**

```text
final_data/
└── species/
    ├── Good/
    └── Bad/
```

---

### `Final Data.ipynb`

Builds the final training dataset by combining manually collected images with warehouse data downloaded from AWS S3.

**Input Structure**

```text
S3_Data/
└── Daily_data/
    └── date/
        └── species/
            └── Single or Group/
                ├── Good/
                └── Bad/
```

**Output Structure**

```text
final_data/
└── species/
    ├── Good/
    └── Bad/
```

---

### `reason_based_data.ipynb`

Creates datasets based on freshness reasons provided by domain experts.

The notebook includes:

* Retrieving annotation data from Amazon RDS
* Data cleaning and filtering
* Exporting processed metadata
* Loading processed metadata
* Creating reason-specific datasets
* Downloading corresponding images from AWS S3
* Dataset statistics and exploratory analysis

These datasets are later used for experiments such as:

* Softness-based freshness classification
* Cuts & damage detection
* Size-based analysis
* Other reason-specific studies

---

### `reorganize_S3_Final_Data.ipynb`

Converts date-wise downloaded datasets into the species-wise directory structure used for model training.

**Input**

```text
date/
└── species/
    ├── Good/
    └── Bad/
```

**Output**

```text
species/
├── Good/
└── Bad/
```


---

## Dataset Organization

The collected images follow a structured hierarchy similar to:

```text
dataset/
└── <collection_date>/
    └── <species>/
        ├── Good/
        └── Bad/
```

This structure enables efficient dataset versioning, filtering, and experiment management.

---

## Data Sources

* AWS S3 warehouse uploads
* Manual image collection
* Human-generated freshness labels
* Reason-based annotations stored in Amazon RDS

---

## Output

The notebooks in this directory generate cleaned and organized datasets that are later used by:

* Dataset Analysis
* Blur Detection experiments
* Single vs Group experiments
* Lighting experiments
* Fish Segmentation
* Eye Turbidity experiments
* Cuts & Damage Detection
* Softness-based Freshness Classification
* Size Estimation
* Species Classification
* Final Freshness Classification models

---

## Notes

* All freshness labels are assigned by domain experts.
* Images are captured under controlled acquisition conditions.
* Dataset preparation is performed before any model training or experimentation.
