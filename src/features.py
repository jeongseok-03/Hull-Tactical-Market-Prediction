# src/features.py
import pandas as pd
import numpy as np
import pandas.api.types
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time

def create_features_no_leakage(df, test_size=126):
    """
    Generates advanced features (MA, STD, RSI, BB, PCA) strictly preventing look-ahead bias.
    Scalers and PCA are fitted ONLY on the Training set.
    """
    print(">>> [Feature Engineering] Generating features (Strict Mode)...")
    st = time.time()
    
    split_idx = len(df) - test_size
    target_col = 'forward_returns'
    exclude = ['date_id', target_col, 'risk_free_rate', 'market_forward_excess_returns']
    
    # Filter numeric base columns
    base_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    
    full_df = df.copy()
    rolling_data = {}
    
    # 1. Rolling Features (MA, STD, Bollinger Bands)
    # Using simple rolling without shifting is safe as long as we don't use bfill with future data
    for col in base_cols:
        # Basic Rolling
        rolling_data[f'{col}_ma5'] = full_df[col].rolling(5).mean()
        rolling_data[f'{col}_ma20'] = full_df[col].rolling(20).mean()
        rolling_data[f'{col}_std20'] = full_df[col].rolling(20).std()
        
        # Bollinger Bands Position (0 to 1)
        upper = rolling_data[f'{col}_ma20'] + 2 * rolling_data[f'{col}_std20']
        lower = rolling_data[f'{col}_ma20'] - 2 * rolling_data[f'{col}_std20']
        denom = (upper - lower).replace(0, 1e-9)
        rolling_data[f'{col}_bb_pos'] = (full_df[col] - lower) / denom

    # 2. RSI (14-day) - Calculating for top 10 important columns only for speed
    for col in base_cols[:10]:
        delta = full_df[col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-9)
        rolling_data[f'{col}_rsi'] = 100 - (100 / (1 + rs))

    # Concat and handle initial NaNs (Fill with 0, NOT backfill)
    rolling_df = pd.DataFrame(rolling_data)
    full_df = pd.concat([full_df, rolling_df], axis=1)
    full_df = full_df.fillna(0)
    
    # 3. Split Train/Test
    train_df = full_df.iloc[:split_idx].copy()
    test_df = full_df.iloc[split_idx:].copy()
    
    # 4. Scaling & PCA (Fit on Train ONLY)
    feat_cols = [c for c in train_df.columns if c not in exclude and pd.api.types.is_numeric_dtype(train_df[c])]
    
    # Handle infinite values
    train_df[feat_cols] = train_df[feat_cols].replace([np.inf, -np.inf], 0)
    test_df[feat_cols] = test_df[feat_cols].replace([np.inf, -np.inf], 0)
    
    # StandardScaler
    scaler = StandardScaler()
    scaler.fit(train_df[feat_cols]) # Strict Fit
    
    train_scaled = scaler.transform(train_df[feat_cols])
    test_scaled = scaler.transform(test_df[feat_cols])
    
    # PCA
    pca = PCA(n_components=10, random_state=42)
    pca.fit(train_scaled) # Strict Fit
    
    train_pca = pca.transform(train_scaled)
    test_pca = pca.transform(test_scaled)
    
    for i in range(10):
        train_df[f'PCA_{i}'] = train_pca[:, i]
        test_df[f'PCA_{i}'] = test_pca[:, i]
        
    print(f"   [Feature Engineering] Completed in {time.time()-st:.2f}s")
    return train_df, test_df