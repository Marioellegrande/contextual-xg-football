# Contextual Expected Goals from Tracking Data in Football

Master’s thesis — University of Southern Denmark (SDU), 2026.

This repository contains the code used to build and evaluate a 
contextual expected goals (xG) model using synchronized tracking and 
event data from the Danish Superliga 2024/25 season.

## Project Overview

Traditional xG models mainly rely on shot geometry, such as shooting 
distance and angle. This thesis extends the classical xG framework by 
including contextual tracking-based features such as:

- Defensive pressure
- Goalkeeper positioning
- Player movement
- Ball speed
- Shot obstruction

Four different model classes are compared under identical experimental 
conditions:

- Logistic Regression
- XGBoost
- TabPFN
- MLP (Neural Network)

The models are analysed using both predictive performance metrics and 
interpretability methods. Model explanations are investigated through:

- Logistic regression coefficients
- SHAP (global feature importance)
- LIME (local explanations)

## Repository Structure

```
code/
├── data_pipeline/       # Shot extraction and event-tracking alignment
├── features/            # Feature engineering (pressure, goalkeeper, obstruction)
└── models/              # Model training and evaluation notebooks
    ├── Utils.py         # Shared utilities (features, encoding, evaluation)
    ├── training/        # Model training and evaluation
    ├── analysis/        # SHAP and LIME analysis notebooks
    └── Validering/      # Match and season simulation notebooks
```

## Data

The dataset consists of 5,056 shots from 192 Danish Superliga 2024/25 matches, of which 601 (11.9%) result in goals. Raw data (Opta events + Second Spectrum tracking) is not included in this repository.

## Dependencies

Main libraries: `xgboost`, `tabpfn`, `shap`, `lime`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`
