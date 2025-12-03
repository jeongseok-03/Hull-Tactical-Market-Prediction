import sys
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
import warnings

# --- Path Setup ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJECT_ROOT)

from src.features import create_features_no_leakage
from src.metrics import score

warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'train.csv')
TEST_SIZE = 126
SEED = 42

def run_stacking():
    print("="*60)
    print(">>> [Experiment 02] Robust Stacking with Purged CV")
    print("="*60)

    # 1. Load Data & Features
    df = pd.read_csv(DATA_PATH)
    train_df, test_df = create_features_no_leakage(df, test_size=TEST_SIZE)
    
    exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
    features = [c for c in train_df.columns if c not in exclude_cols]
    target = 'forward_returns'

    # 2. OOF Prediction (Purged TimeSeriesSplit)
    print("\n>>> Generating OOF Predictions...")
    N_FOLDS = 5
    tscv = TimeSeriesSplit(n_splits=N_FOLDS, gap=50) 
    
    lgbm = lgb.LGBMRegressor(n_estimators=300, random_state=SEED, verbose=-1)
    cat = CatBoostRegressor(iterations=300, random_state=SEED, verbose=0, allow_writing_files=False)

    meta_train = np.zeros((len(train_df), 2)) # [LGBM, CatBoost]
    
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(train_df)):
        X_tr, y_tr = train_df.iloc[tr_idx][features], train_df.iloc[tr_idx][target]
        X_val = train_df.iloc[val_idx][features]
        
        lgbm.fit(X_tr, y_tr)
        cat.fit(X_tr, y_tr)
        
        meta_train[val_idx, 0] = lgbm.predict(X_val)
        meta_train[val_idx, 1] = cat.predict(X_val)
        print(f" - Fold {fold+1}/{N_FOLDS} Completed.")

    valid_mask = np.sum(meta_train, axis=1) != 0
    X_meta = meta_train[valid_mask]
    y_meta = train_df[target].values[valid_mask]

    # 3. Train Meta-Model
    print(">>> Training Meta-Model (Ridge)...")
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(X_meta, y_meta)

    # 4. Retrain Base Models on Full Train
    print(">>> Retraining Base Models...")
    lgbm.fit(train_df[features], train_df[target])
    cat.fit(train_df[features], train_df[target])

    # 5. Final Prediction
    p1 = lgbm.predict(test_df[features])
    p2 = cat.predict(test_df[features])
    final_pred = meta_model.predict(np.column_stack([p1, p2]))
    
    # Strategy: 1.5x Leverage
    allocations = np.where(final_pred > 0, 1.5, 0.0)
    submission = pd.DataFrame({'prediction': allocations}, index=test_df.index)

    # 6. Evaluation
    try:
        final_score = score(test_df.copy(), submission, 'date_id')
        print(f" >>> STACKING STRATEGY SCORE: {final_score:.4f}")
    except Exception as e:
        print(f"Scoring Error: {e}")

if __name__ == "__main__":
    run_stacking()