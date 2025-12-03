import sys
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import torch
import warnings
import gc
import time
import pandas.api.types

# --- Path Setup ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJECT_ROOT)

from src.features import create_features_no_leakage
from src.rules import find_rules_gpu, apply_rules_with_prefix
from src.metrics import score

warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'train.csv')
TARGET_COL = 'forward_returns'
TEST_SIZE = 126
MACRO_START_ID = 1000  # For Long-term Rules
TRAIN_START_ID = 7800  # For Short-term Rules & Training

def run_final_strategy():
    print("="*60)
    print(">>> [Experiment 04] GPU Alpha Mining & Smart Leverage Strategy")
    print("="*60)
    total_st = time.time()

    # 1. Load & Feature Engineering
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return
    df = pd.read_csv(DATA_PATH)
    train_df, test_df = create_features_no_leakage(df, test_size=TEST_SIZE)

    # 2. GPU Rule Mining
    macro_rules = find_rules_gpu(train_df, MACRO_START_ID, label="Macro")
    micro_rules = find_rules_gpu(train_df, TRAIN_START_ID, label="Micro")

    train_df = apply_rules_with_prefix(train_df, macro_rules, prefix="M")
    test_df = apply_rules_with_prefix(test_df, macro_rules, prefix="M")
    train_df = apply_rules_with_prefix(train_df, micro_rules, prefix="m")
    test_df = apply_rules_with_prefix(test_df, micro_rules, prefix="m")

    # 3. LGBM Ensemble Training
    print("\n>>> Training LGBM Ensemble...")
    exclude = ['date_id', TARGET_COL, 'risk_free_rate', 'market_forward_excess_returns']
    model_feats = [c for c in train_df.columns if c not in exclude and pd.api.types.is_numeric_dtype(train_df[c])]
    
    real_train_df = train_df[train_df['date_id'] >= TRAIN_START_ID].copy()

    seeds = [42, 2024, 777]
    models = []
    
    for s in seeds:
        model = lgb.LGBMRegressor(
            objective='mse', n_estimators=1500, learning_rate=0.03,
            num_leaves=127, max_depth=12, random_state=s,
            n_jobs=-1, verbose=-1
        )
        model.fit(real_train_df[model_feats], real_train_df[TARGET_COL])
        models.append(model)

    # 4. Z-Score Calibration
    print("\n>>> Calibrating Predictions (Z-Score & Vol Control)...")
    
    train_preds = np.mean([m.predict(real_train_df[model_feats]) for m in models], axis=0)
    train_mean, train_std = np.mean(train_preds), np.std(train_preds)
    
    # Train-based Thresholds
    z_dist = (train_preds - train_mean) / train_std
    z_high = np.percentile(z_dist, 80)
    z_super = np.percentile(z_dist, 95)

    print(f" - Thresholds | High(Top20%): {z_high:.4f}, Super(Top5%): {z_super:.4f}")

    # Test Prediction
    test_preds = np.mean([m.predict(test_df[model_feats]) for m in models], axis=0)
    test_z = (test_preds - train_mean) / train_std

    # 5. Position Sizing
    vol_col = [c for c in test_df.columns if 'std20' in c][0]
    current_vol = test_df[vol_col].values
    avg_vol = np.mean(current_vol)

    final_pos = []
    lev_counts = {1.0: 0, 1.2: 0, 2.0: 0}

    for i, z_val in enumerate(test_z):
        base_pos = 1.0
        vol_ratio = current_vol[i] / (avg_vol + 1e-9)
        
        if z_val >= z_super:
            lev = 2.0 / max(1.0, vol_ratio)
            base_pos = max(1.0, lev)
            lev_counts[2.0] += 1
        elif z_val >= z_high:
            lev = 1.2 / max(1.0, vol_ratio)
            base_pos = max(1.0, lev)
            lev_counts[1.2] += 1
        else:
            base_pos = 1.0
            lev_counts[1.0] += 1
        
        final_pos.append(min(2.0, max(1.0, base_pos)))

    print(f" - Allocation: {lev_counts}")

    # 6. Evaluation
    submission = pd.DataFrame({'prediction': final_pos}, index=test_df.index)
    try:
        print("="*60)
        final_score = score(test_df.copy(), submission, 'date_id')
        print(f" >>> FINAL STRATEGY SCORE: {final_score:.4f}")
        print("="*60)
    except Exception as e:
        print(f"Scoring Error: {e}")

    print(f"Total Time: {time.time() - total_st:.2f}s")

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"★ GPU Enabled: {torch.cuda.get_device_name(0)}")
    run_final_strategy()