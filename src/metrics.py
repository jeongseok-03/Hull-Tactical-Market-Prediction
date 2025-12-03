# src/metrics.py
import pandas as pd
import numpy as np
import pandas.api.types

class ParticipantVisibleError(Exception):
    pass

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Official Evaluation Metric: Volatility-Adjusted Sharpe Ratio.
    Calculates penalties for excess volatility and return gaps compared to the market.
    """
    # 1. Input Validation
    if not pandas.api.types.is_numeric_dtype(submission['prediction']):
        raise ParticipantVisibleError('Predictions must be numeric')

    solution = solution.copy()
    solution['position'] = submission['prediction']

    # 2. Constraint Checks
    if solution['position'].max() > 2:
        raise ParticipantVisibleError('Position exceeds maximum of 2.0')
    if solution['position'].min() < 0:
        raise ParticipantVisibleError('Position below minimum of 0.0')

    # 3. Strategy Performance
    solution['strategy_returns'] = (
        solution['risk_free_rate'] * (1 - solution['position']) +
        solution['position'] * solution['forward_returns']
    )

    strategy_excess_returns = solution['strategy_returns'] - solution['risk_free_rate']
    strategy_mean_excess = (1 + strategy_excess_returns).prod() ** (1 / len(solution)) - 1
    strategy_std = solution['strategy_returns'].std()
    
    # Avoid division by zero
    if strategy_std == 0: return 0.0
    
    trading_days = 252
    sharpe = strategy_mean_excess / strategy_std * np.sqrt(trading_days)
    strategy_vol = float(strategy_std * np.sqrt(trading_days) * 100)

    # 4. Market Benchmarks
    market_excess = solution['forward_returns'] - solution['risk_free_rate']
    market_mean_excess = (1 + market_excess).prod() ** (1 / len(solution)) - 1
    market_std = solution['forward_returns'].std()
    market_vol = float(market_std * np.sqrt(trading_days) * 100)

    if market_vol == 0: return 0.0

    # 5. Penalties Calculation
    # Volatility Penalty
    excess_vol = max(0, strategy_vol / market_vol - 1.2) if market_vol > 0 else 0
    vol_penalty = 1 + excess_vol

    # Return Penalty
    return_gap = max(0, (market_mean_excess - strategy_mean_excess) * 100 * trading_days)
    return_penalty = 1 + (return_gap**2) / 100

    # 6. Final Adjusted Score
    adjusted_sharpe = sharpe / (vol_penalty * return_penalty)
    
    # Optional: Print detailed metrics for debugging
    # print(f"  [Metric] Sharpe: {sharpe:.4f} | VolPen: {vol_penalty:.4f} | RetPen: {return_penalty:.4f}")
    
    return min(float(adjusted_sharpe), 1_000_000)