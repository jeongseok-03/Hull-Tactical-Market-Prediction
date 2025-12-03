import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import warnings

# --- Path Setup ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJECT_ROOT)

from src.features import create_features_no_leakage
from src.metrics import score

warnings.filterwarnings('ignore')
tf.random.set_seed(42)

DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'train.csv')
TEST_SIZE = 126
LOOKBACK = 10 

def create_sequences(data, lookback):
    Xs = []
    for i in range(len(data) - lookback + 1):
        Xs.append(data[i:(i+lookback)])
    return np.array(Xs)

def run_lstm_hybrid():
    print("="*60)
    print(">>> [Experiment 03] Hybrid Deep Learning (Meta-LSTM)")
    print("="*60)

    # 1. Load & Feature Engineering
    df = pd.read_csv(DATA_PATH)
    train_df, test_df = create_features_no_leakage(df, test_size=TEST_SIZE)
    
    exclude = ['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
    features = [c for c in train_df.columns if c not in exclude]
    target = 'forward_returns'

    # 2. Generate Meta-Features (LGBM OOF)
    print("\n>>> Generating Meta-Features (LGBM OOF)...")
    tscv = TimeSeriesSplit(n_splits=5, gap=20)
    lgbm = lgb.LGBMRegressor(n_estimators=200, verbose=-1, random_state=42)
    
    train_df['lgbm_pred'] = 0.0
    for tr_idx, val_idx in tscv.split(train_df):
        X_tr, y_tr = train_df.iloc[tr_idx][features], train_df.iloc[tr_idx][target]
        X_val = train_df.iloc[val_idx][features]
        lgbm.fit(X_tr, y_tr)
        train_df.iloc[val_idx, train_df.columns.get_loc('lgbm_pred')] = lgbm.predict(X_val)

    valid_idx = train_df['lgbm_pred'] != 0
    meta_train_df = train_df[valid_idx].copy()
    
    # LSTM Features: [LGBM Prediction, Volatility]
    lstm_feats = ['lgbm_pred'] 
    vol_col = [c for c in meta_train_df.columns if 'std20' in c][0]
    lstm_feats.append(vol_col)
    
    scaler = StandardScaler()
    X_train_meta = scaler.fit_transform(meta_train_df[lstm_feats])
    y_train_meta = meta_train_df[target].values
    
    # Prepare Test Data
    lgbm.fit(train_df[features], train_df[target])
    test_df['lgbm_pred'] = lgbm.predict(test_df[features])
    X_test_meta = scaler.transform(test_df[lstm_feats])

    # 3. Create Sequences
    X_seq_train, y_seq_train = [], []
    for i in range(len(X_train_meta) - LOOKBACK):
        X_seq_train.append(X_train_meta[i:i+LOOKBACK])
        y_seq_train.append(y_train_meta[i+LOOKBACK])
    X_seq_train, y_seq_train = np.array(X_seq_train), np.array(y_seq_train)

    concat_meta = np.vstack([X_train_meta[-LOOKBACK:], X_test_meta])
    X_seq_test = create_sequences(concat_meta, LOOKBACK)
    X_seq_test = X_seq_test[-len(test_df):]

    # 4. Train LSTM
    print(">>> Training Meta-LSTM...")
    model = Sequential([
        Input(shape=(LOOKBACK, len(lstm_feats))),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.fit(X_seq_train, y_seq_train, epochs=10, batch_size=64, verbose=0)

    # 5. Predict
    preds = model.predict(X_seq_test, verbose=0).flatten()
    final_pos = [1.0 + min(1.0, p * 100) if p > 0 else 0.0 for p in preds]
    
    submission = pd.DataFrame({'prediction': final_pos}, index=test_df.index)
    
    try:
        final_score = score(test_df.copy(), submission, 'date_id')
        print(f" >>> HYBRID LSTM SCORE: {final_score:.4f}")
    except Exception as e:
        print(f"Scoring Error: {e}")

if __name__ == "__main__":
    run_lstm_hybrid()