# Hybrid Meta-Stacking: LSTM Sequence Modeling

## 🎯 Objective
Capture sequential dependencies in the errors/predictions of base models using an LSTM Meta-Learner.

## 🛡️ Leakage Prevention Strategy
1.  **Causal Windowing:** Input sequences `X[t]` consist strictly of data points `[t-window : t]`. The target `y[t]` is strictly separated from the input window.
2.  **Scaling Hygiene:** Standardization is fitted strictly on OOF predictions (Proxy for Train), then applied to Test predictions.