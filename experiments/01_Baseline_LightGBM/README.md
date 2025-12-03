# Baseline Strategy: LightGBM with Strict Temporal Split

## 🎯 Objective
Establish a performance benchmark using a Gradient Boosting Machine (LGBM) while ensuring zero look-ahead bias.

## 🛡️ Leakage Prevention Strategy
1.  **Temporal Split:** Data is split strictly by time index (Train: First 80% / Test: Last 20%). No random shuffling.
2.  **Stateless Feature Engineering:** Features like `Rolling Mean` are calculated using expanding windows or strict shifts to avoid future data inclusion.
3.  **Strict Imputation:** Missing values are filled using the `median` of the **Train Set only**. The Test set sees the Train median, preserving the "unknown" nature of the future.
