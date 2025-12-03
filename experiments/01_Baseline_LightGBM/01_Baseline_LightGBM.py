import sys
import os
import pandas as pd
import lightgbm as lgb
import warnings

# --- Path Setup ---
# 프로젝트 루트 경로 설정 (현재 파일 위치에서 두 단계 상위 폴더)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJECT_ROOT)

# src 모듈 임포트
from src.features import create_features_no_leakage
from src.metrics import score

warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'train.csv')
TEST_SIZE = 126
SEED = 42

def run_baseline():
    print("="*60)
    print(">>> [Experiment 01] Baseline LightGBM Strategy")
    print("="*60)

    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return
    df = pd.read_csv(DATA_PATH)

    # 2. Feature Engineering (Strict Mode: No Leakage)
    train_df, test_df = create_features_no_leakage(df, test_size=TEST_SIZE)

    # 3. Model Training
    print("\n>>> Training LightGBM Model...")
    exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
    features = [c for c in train_df.columns if c not in exclude_cols]
    target = 'forward_returns'

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(train_df[features], train_df[target])

    # 4. Prediction & Strategy
    print(">>> Generating Predictions & Allocating Positions...")
    preds = model.predict(test_df[features])
    
    # Strategy: Positive -> 2.0x, Negative -> 0.0x
    allocations = [2.0 if p > 0 else 0.0 for p in preds]
    
    submission = pd.DataFrame({'prediction': allocations}, index=test_df.index)

    # 5. Evaluation
    print("\n>>> Evaluating Baseline Performance...")
    try:
        final_score = score(test_df.copy(), submission, 'date_id')
        print(f" >>> OFFICIAL BASELINE SCORE: {final_score:.4f}")
    except Exception as e:
        print(f"Scoring Error: {e}")

if __name__ == "__main__":
    run_baseline()