# App Testing

## Overview

This directory contains scripts and notebooks used to evaluate the deployed Fish Freshness mobile application using real-world testing data.

The pipeline retrieves prediction results and user feedback from the application database, downloads the corresponding input and output images from AWS S3, performs statistical analysis, and generates performance reports.

This folder is intended for post-deployment model validation rather than model training.

---

## Workflow

```text
Mobile App
      │
      ▼
PostgreSQL Database
      │
      ▼
Retrieve Prediction Results
      │
      ▼
Data Cleaning & Filtering
      │
      ▼
Download Images from AWS S3
      │
      ▼
Performance Analysis
      │
      ▼
Confusion Matrices & Evaluation Metrics
```

---

## Directory Contents

### `New app data analysis.ipynb`

Interactive notebook used for exploratory analysis of application testing results.

**Features**

* Retrieve testing results from the application database
* Clean and preprocess prediction data
* Analyze prediction performance
* Generate confusion matrices
* Calculate classification metrics
* Download testing images from AWS S3
* Download images filtered by rejection reasons (e.g., softness, cuts & damage, smell)

---

### `app_testing_results_analysis.py`

Standalone evaluation pipeline for application testing.

**Features**

* Connects to PostgreSQL
* Retrieves prediction results
* Cleans and enriches metadata
* Filters data by date, species, and user
* Computes confusion matrices
* Calculates classification metrics
* Generates evaluation reports and visualizations

Can be executed independently or imported as a Python module.

---

### `run_downloads.py`

Downloads application testing images from AWS S3.

**Workflow**

1. Configure AWS credentials
2. Load application testing metadata
3. Filter the required samples
4. Download corresponding images

---

### `s3_downloader.py`

Utility module that downloads input and output images from AWS S3 using metadata generated during application testing.

---

### `segment_app_testing_data.py`

Segments fish instances from application testing images using a trained YOLO segmentation model.

The segmented images preserve the original directory hierarchy and are used for evaluating segmentation-based freshness models.

Input Structure

```text
date/
└── species/
    ├── Good/
    └── Bad/
```

Output Structure

```text
date/
└── species/
    ├── Good/
    └── Bad/
```

where each detected fish instance is saved as an individual segmented image.

---
## Reports

This folder contains the confusion matrices and classification reports generated for the app testing data.

---

## Data Sources

* PostgreSQL database
* AWS S3
* Fish Freshness mobile application
* User feedback collected during application testing

---

## Purpose

The tools in this directory are used to:

* Validate deployed model performance
* Analyze prediction errors
* Evaluate user feedback
* Download application testing datasets
* Generate performance reports
* Prepare testing images for further experimentation
