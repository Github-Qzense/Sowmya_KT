# Fish Gills Model Investigation

## Overview

This document summarizes the investigation conducted on the **Fish_Gills_dataset** and the available training scripts to identify the model currently used in the Fish Freshness application.

The objective was to understand the dataset organization, model architecture, training methodology, and determine which model corresponds to the one deployed in the application.

Folder link: [Fish Gills](https://drive.google.com/drive/folders/1nqU6t5-JcmqhwxXKQ7FnPjDQm2xSaRkn?usp=drive_link)
---

# Model Architecture

The current gills model used in the application is a **transfer learning-based CNN** with **InceptionResNet** as the backbone.

The model predicts freshness values represented as:

* 0
* 20
* 40
* 60
* 80
* 100

These values correspond to different freshness levels.

---

# Dataset Investigation

## 1. Hourly Dataset

Images are organized into folders representing storage time:

```text
Hour_0/
Hour_2/
Hour_4/
...
Hour_14/
```

### Training Approaches

* SVM classifier
* CNN classifier

Both models were trained to classify images into the corresponding hourly categories.

Some notebooks trained models using only **Hour_0 to Hour_8**, although the reason for excluding the remaining classes is unknown.

### Relevance

This dataset does not appear to be related to the model currently used in the application, although it is referenced in multiple training files.

---

## 2. M / N / S Classification Dataset

The dataset contains three classes:

```text
M/
N/
S/
```

A notebook located at:

```text
Fish_Gills_dataset/
└── Fish Image data analysis/
    └── 5 Part Images Fish Gills slicing.ipynb
```

describes how these datasets were created.

### Dataset Creation Process

Each fish gill image is divided into five regions:

* Center
* Left
* Right
* Top
* Bottom

Each cropped region is then manually assigned to one of the following:

* **M**
* **N**
* **Skipped (likely stored in S)**

### Unknowns

The following information could not be determined:

* What M and N represent
* The criteria used for assigning images to M or N
* The purpose of the S class

### Models Found

#### SVM

An SVM model was trained for M/N classification.

#### CNN

Another CNN model was trained using images from the `cropfish` directory.

Notebook:

```text
Fish Classification_raghvendra/
└── Untitled.ipynb
```

### `fishmodel.h5`

A trained model named **fishmodel.h5** is used in:

```text
upd_FishPredict_and_save_images_to_folder.ipynb
```

Model characteristics:

* InceptionResNet backbone
* Sigmoid activation
* Binary classification

The prediction probability is multiplied by **100** and converted into freshness levels using a function named **grouper()**, producing:

* 0
* 20
* 40
* 60
* 80
* 100

### Relevance

This model closely matches the model currently available for the application and is therefore the most likely candidate for the production model.

However, the notebook used to train **fishmodel.h5** could not be found.

---

## 3. Main Freshness Dataset

Images are organized into folders representing freshness scores:

```text
0/
20/
40/
60/
80/
100/
```

### Labeling Process

No documentation explaining how these freshness labels were assigned was found.

### Models Found

Notebook:

```text
ordinalfish.ipynb
```

Two different approaches were identified.

#### Regression Model

* InceptionResNet backbone
* ReLU output activation
* Mean Squared Error (MSE) loss

#### Classification Model

* InceptionResNet backbone
* Softmax output activation
* Negative Log Likelihood (NNLoss)

### Missing Files

The notebook expects a training file:

```text
fish_scr_train.csv
```

which is not present in the shared dataset.

### Relevance

Since the production model uses **sigmoid activation**, this dataset and its associated models do not appear to match the deployed application model.

---

## 4. Color-Based Dataset

Datasets are available in:

* `RGY-Rubber Band`
* `GoMicro`

Images are organized into color-based freshness categories:

* 20 - Red
* 40 - Yellow
* 60 - Yellow
* 80 - Green
* 100 - Green

### Findings

No notebooks or training scripts using this dataset were found.

---

# Key Observations

* The **M/N classification task** is the only binary classification problem identified in the repository and aligns with the architecture of the deployed model.

* **fishmodel.h5** is the strongest candidate for the model currently used in the application.

* The training notebook corresponding to **fishmodel.h5** could not be located.

* The hourly dataset and regression-based models do not appear to be related to the current production pipeline.

* The methodology used to assign freshness labels (0–100) remains undocumented.

* Since the M/N/S datasets are created by dividing each gill image into five cropped regions, it remains unclear how the application derives a single freshness score for the complete gill image from the five individual predictions.

---

# Conclusion

Based on the available datasets, notebooks, and model files, **fishmodel.h5** appears to be the model currently used by the Fish Freshness application.

However, several important pieces of information remain unavailable:

* The original training notebook for `fishmodel.h5`
* The dataset used to train the model
* The definition of the M and N classes
* The labeling methodology for the freshness datasets
* The algorithm used to combine predictions from the five cropped gill regions into a final freshness score

These missing components make it difficult to fully reproduce or understand the complete training pipeline.
