# ... (GPU Mining Functions) ...

# [Leakage Defense] Z-Score Calculation Logic
def calculate_position_logic(train_preds, test_preds):
    # 1. Derive statistics strictly from TRAIN
    train_mean = np.mean(train_preds)
    train_std = np.std(train_preds)
    
    # 2. Determine Thresholds from TRAIN distribution
    z_scores_train = (train_preds - train_mean) / train_std
    th_high = np.percentile(z_scores_train, 80)
    th_super = np.percentile(z_scores_train, 95)
    
    # 3. Apply to TEST using TRAIN stats
    # We calibrate Test predictions to the Train scale
    z_scores_test = (test_preds - train_mean) / train_std 
    
    final_pos = []
    for z in z_scores_test:
        if z > th_super: pos = 2.0
        elif z > th_high: pos = 1.2
        else: pos = 1.0
        final_pos.append(pos)
        
    return final_pos