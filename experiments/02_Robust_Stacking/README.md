# Robust Stacking: Purged Cross-Validation

## 🎯 Objective
Reduce variance and overfitting by stacking heterogeneous models (LightGBM, CatBoost) using a Ridge Regression meta-learner.

## 🛡️ Leakage Prevention Strategy
1.  **Purged TimeSeriesSplit:** Applied a `gap` between training and validation folds to prevent information leakage through temporal correlation at the boundaries.
2.  **Strict OOF Generation:** The Meta-Model is trained **exclusively** on Out-Of-Fold predictions. It never sees predictions made by a model on the data it was trained on.