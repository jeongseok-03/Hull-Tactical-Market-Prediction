# src/rules.py
import torch
import numpy as np
import pandas as pd
import pandas.api.types
from itertools import permutations
import time

def find_rules_gpu(train_df, start_id, target_col='forward_returns', 
                   batch_size=50000, min_occurrence=30, min_win_rate=0.60, label="Rules"):
    """
    Mines 3-way interaction rules (A > B > C) using GPU acceleration.
    """
    print(f"\n>>> [{label}] GPU Rule Mining (ID {start_id}+) ...")
    st = time.time()
    
    # Filter data by date_id
    target_df = train_df[train_df['date_id'] >= start_id].copy()
    
    exclude = ['date_id', target_col, 'risk_free_rate', 'market_forward_excess_returns']
    feats = [c for c in target_df.columns if c not in exclude and pd.api.types.is_numeric_dtype(target_df[c])]
    
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   - Device: {device}")
    
    # Prepare Tensors
    X_np = target_df[feats].values.astype(np.float32)
    # Target condition: Returns > 0
    y_bool = torch.tensor((target_df[target_col].values > 0), device=device, dtype=torch.bool).unsqueeze(1)
    X_gpu = torch.tensor(X_np, device=device)
    
    n_feats = len(feats)
    perms = list(permutations(range(n_feats), 3))
    total_perms = len(perms)
    print(f"   - Total Combinations: {total_perms:,}")
    
    valid_rules = []
    
    # Batch Processing
    for i in range(0, total_perms, batch_size):
        batch_indices = perms[i : i + batch_size]
        
        idx_a = [p[0] for p in batch_indices]
        idx_b = [p[1] for p in batch_indices]
        idx_c = [p[2] for p in batch_indices]
        
        col_a = X_gpu[:, idx_a]
        col_b = X_gpu[:, idx_b]
        col_c = X_gpu[:, idx_c]
        
        # Boolean Logic: A > B > C
        mask = (col_a > col_b) & (col_b > col_c)
        
        counts = mask.sum(dim=0)
        wins = (mask & y_bool).sum(dim=0)
        
        counts_cpu = counts.cpu().numpy()
        wins_cpu = wins.cpu().numpy()
        
        # Filtering
        valid_mask = (counts_cpu >= min_occurrence)
        if valid_mask.sum() > 0:
            valid_indices = np.where(valid_mask)[0]
            for idx in valid_indices:
                cnt = counts_cpu[idx]
                wr = wins_cpu[idx] / cnt
                if wr >= min_win_rate:
                    orig_idx = batch_indices[idx]
                    rule_str = f"{feats[orig_idx[0]]} > {feats[orig_idx[1]]} > {feats[orig_idx[2]]}"
                    valid_rules.append((rule_str, wr, cnt))
        
        # Cleanup Memory
        del col_a, col_b, col_c, mask, counts, wins
        if i % 2 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Sort by Win Rate -> Count
    valid_rules.sort(key=lambda x: (x[1], x[2]), reverse=True)
    print(f"   - Found {len(valid_rules)} valid rules.")
    print(f"   [{label} Search] Completed in {time.time()-st:.2f}s")
    
    # Return top 20 rule strings
    return [r[0] for r in valid_rules[:20]]

def apply_rules_with_prefix(df, rules, prefix):
    """
    Applies the mined rules to the DataFrame, creating new boolean features.
    """
    for rule in rules:
        parts = rule.split(' > ')
        col_a, col_b, col_c = parts[0], parts[1], parts[2]
        df[f'{prefix}_{rule}'] = ((df[col_a] > df[col_b]) & (df[col_b] > df[col_c])).astype(int)
    return df
