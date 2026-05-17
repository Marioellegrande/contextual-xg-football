## Feature scripts

This directory contains reusable scripts that compute derived features from the shot tables and tracking.

- `pressure_features.py`: compute pressure features from `*_shots_aligned.csv` and
  `*_SecondSpectrum_Data.jsonl`. Output is a new shot-level CSV with added columns, including
  DataBallPy Adrienko/Herold pressure (`pressure_total_herold`) and proximity summaries
  (nearest-defender distance + defender counts in radii).

