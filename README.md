# Data Preprocessing – Kepler Exoplanet Detection

## Overview

This module handles data preprocessing for the Kepler Exoplanet Detection project using NASA Kepler light curve dataset metadata.

## Preprocessing Pipeline

The following operations were performed:

1. Dataset loading using Pandas.
2. Removal of unnecessary and target-related columns.
3. Target label encoding.
4. Removal of completely empty features.
5. Missing value imputation using median and most frequent strategies.
6. Categorical feature encoding using One-Hot Encoding.
7. Feature scaling using StandardScaler.
8. Group-based dataset splitting to prevent data leakage between stars.
9. Class weight calculation to address class imbalance.

## Target Encoding

| Class | Label |
|---|---|
| FALSE POSITIVE | 0 |
| CONFIRMED | 1 |
| CANDIDATE | 2 |

## Dataset Splitting

| Dataset | Samples |
|---|---:|
| Training | 6,695 |
| Validation | 1,463 |
| Testing | 1,406 |

All splits are grouped by `kepid` to prevent the same star from appearing in multiple datasets.

## Final Dataset Dimensions

- Number of features: 36
- Training shape: (6695, 36)
- Validation shape: (1463, 36)
- Testing shape: (1406, 36)
- Feature data type: float32
- Target data type: int64

## Class Weights

| Class | Weight |
|---|---:|
| 0 | 0.6308 |
| 1 | 1.3983 |
| 2 | 1.4296 |

Class weights are provided to help the Deep Learning model handle class imbalance during training.

## Output Files

### 1. processed_kepler_data.npz

Contains:
- X_train
- y_train
- X_val
- y_val
- X_test
- y_test

### 2. kepler_preprocessor.pkl

Contains the fitted preprocessing pipeline, including imputation, scaling, and encoding.

### 3. class_weights.json

Contains the calculated class weights for model training.

## Loading the Processed Data

```python
import numpy as np
import json
import joblib

data = np.load('processed_kepler_data.npz')

X_train = data['X_train']
y_train = data['y_train']

X_val = data['X_val']
y_val = data['y_val']

X_test = data['X_test']
y_test = data['y_test']

with open('class_weights.json', 'r') as f:
    class_weights = json.load(f)

preprocessor = joblib.load('kepler_preprocessor.pkl')
```

## Data Quality Verification

- No missing values after preprocessing.
- No infinite values.
- No overlap between star groups across datasets.
- Preprocessing fitted exclusively on training data.

## Status

Preprocessing completed successfully. The processed datasets and preprocessing artifacts are ready for integration with the Deep Learning and Model Evaluation modules.
