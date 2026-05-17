# Contextual Expected Goals from Tracking Data in Football

Master's thesis — University of Southern Denmark (SDU), 2026.

This repository contains the code for building a contextual expected goals (xG) model using synchronized tracking data from the Danish Superliga 2024/25 season.

## Project Overview

Classical xG models rely on shot geometry (distance and angle). This thesis extends that by incorporating contextual tracking features such as defensive pressure, goalkeeper positioning, player movement, and ball speed.

Four model classes are compared under identical conditions:
- Logistic Regression
- XGBoost
- TabPFN
- MLP (Neural Network)

Model interpretability is analysed using logistic regression coefficients, SHAP (global), and LIME (local).

## Repository Structure

```
code/
├── data_pipeline/       # Shot extraction and event-tracking alignment
├── features/            # Feature engineering (pressure, goalkeeper, obstruction)
├── models/              # Model training and evaluation notebooks
│   ├── Utils.py         # Shared utilities (features, encoding, evaluation)
│   ├── analysis/        # SHAP and LIME analysis notebooks
│   └── ...
├── figures/             # Figure generation scripts
└── visualization/       # Visualization scripts
```

## Data

The dataset consists of 4,957 shots from 189 Superliga matches. Raw data (Opta events + Second Spectrum tracking) is not included in this repository.

## Dependencies

Main libraries: `xgboost`, `tabpfn`, `shap`, `lime`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`
