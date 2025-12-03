# Hull Tactical Market Prediction: Quantitative Alpha Research & Strategy Optimization

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-GPU%20Acceleration-EE4C2C?logo=pytorch&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Focus-Machine%20Learning-orange)
![Quant Finance](https://img.shields.io/badge/Domain-Quant%20Finance-success)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Project Overview

This project focuses on developing a robust **algorithmic trading strategy** to predict short-term market returns (S&P 500) and maximize the **Volatility-Adjusted Sharpe Ratio**.

The research pipeline demonstrates a rigorous evolution from establishing benchmarks with tree-based models to exploring **sequential meta-labeling** and optimizing alphas via **GPU-accelerated rule mining**. A key emphasis is placed on **preventing look-ahead bias** and implementing **dynamic risk management** mechanisms.

---

## 📂 Repository Structure

A modularized structure ensures reproducibility and scalability of the research.

```text
Hull-Tactical-Market-Prediction/
├── data/                   # Dataset documentation (data not included)
├── experiments/            # Sequential research notebooks & scripts
│   ├── 01_Baseline_LightGBM/
│   ├── 02_Robust_Stacking/
│   ├── 03_Hybrid_DL_Meta_Stacking/
│   └── 04_GPU_Alpha_Mining/
├── src/                    # Core source code (Refactored modules)
│   ├── features.py         # Leakage-free feature engineering
│   ├── rules.py            # GPU-accelerated rule mining engine
│   └── metrics.py          # Official evaluation metric implementation
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation