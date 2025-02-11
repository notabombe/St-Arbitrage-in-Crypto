import numpy as np
import pandas as pd
import scipy.optimize as sco

# 1. Refining Mean Reversion Signal
# You’re using a Z-score to calculate the mean reversion signal, which is a good approach. However, we could improve the calculation by handling edge cases where the rolling standard deviation might be zero (which could happen if the price doesn't change for a long period).
# Improvement: Add a small epsilon value to the denominator to avoid division by zero.
def mean_reversion(df, window):
    """Calculate the mean reversion signal for a given window."""
    rolling_mean = df.rolling(window).mean()
    rolling_std = df.rolling(window).std()
    zscore = (df - rolling_mean) / (rolling_std + 1e-8)  # Adding small epsilon to avoid zero division
    return -zscore

# 2. Momentum Signal Adjustments
# Currently, the momentum signal is calculated as a sum of returns over a rolling window. This is a simple approach, but it can be extended to a more sophisticated version that accounts for volatility or adapts to varying market conditions.
# Improvement: Introduce a volatility-adjusted momentum signal to make the strategy more responsive to market dynamics.
def momentum(df, window):
    """Calculate the volatility-adjusted momentum signal for a given window."""
    returns = df.pct_change()
    rolling_volatility = returns.rolling(window).std()
    momentum_signal = returns.rolling(window).sum() / (rolling_volatility + 1e-8)  # Prevent division by zero
    return momentum_signal

# 3. Signal Combination
# The current approach averages the signals. This works well for some cases, but there are instances where a weighted combination of signals could be more beneficial (e.g., giving more importance to certain signals depending on the market conditions).
# Improvement: Add the option to specify weights for the signals.
def combine_signals(signals, weights=None):
    """Combine multiple trading signals with optional weights."""
    if weights is None:
        weights = np.ones(len(signals)) / len(signals)
    combined_signal = sum(weight * signal for weight, signal in zip(weights, signals))
    return np.sign(combined_signal)

# 4. Portfolio Backtesting Enhancements
# The backtest function can be enhanced by accounting for more factors like slippage or trading frequency. Additionally, calculating transaction costs based on the position size rather than just the change in positions can give more realistic results.
# Improvement: Include slippage and a more nuanced cost model.
def backtest_portfolio(returns, positions, transaction_cost, slippage=0.001):
    """Backtest a portfolio of positions with slippage and more realistic transaction costs."""
    # Calculate slippage-adjusted returns
    adjusted_returns = returns - slippage * np.abs(positions - positions.shift(1))

    # Calculate the portfolio returns with transaction cost
    portfolio_returns = (positions.shift(1) * returns) - (np.abs(positions - positions.shift(1)) * transaction_cost)
    
    # Calculate the cumulative returns
    cumulative_returns = (1 + portfolio_returns).cumprod()

    # Calculate the drawdowns
    previous_peaks = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - previous_peaks) / previous_peaks

    # Calculate the statistics
    sharpe_ratio = np.sqrt(252) * portfolio_returns.mean() / portfolio_returns.std()
    max_drawdown = drawdowns.min()

    return cumulative_returns, sharpe_ratio, max_drawdown

# 5. Optimization
# The optimization process seems fine, but it could be sped up or made more robust by adding constraints for limiting the maximum weight for any asset (to prevent concentration risk).
# Improvement: Add upper bounds for portfolio weights.
def optimize_portfolio(returns, max_weight=0.2):
    """Optimize the portfolio weights to maximize the Sharpe ratio with constraints on asset weights."""
    def objective_function(weights, returns):
        portfolio_returns = np.dot(returns, weights)
        sharpe_ratio = np.sqrt(252) * portfolio_returns.mean() / portfolio_returns.std()
        return -sharpe_ratio

    # Constraints and bounds
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = [(0, max_weight) for _ in range(len(returns.columns))]

    # Initialize weights
    weights = np.ones(len(returns.columns)) / len(returns.columns)

    # Optimize weights
    optimized_weights = sco.minimize(objective_function, weights, args=(returns,), method='SLSQP', bounds=bounds, constraints=constraints)

    return optimized_weights.x

# 6. Consider Portfolio Rebalancing
# The code assumes that the portfolio is rebalanced based on signals. You might want to add a function to explicitly handle periodic rebalancing (e.g., monthly or quarterly).
# Improvement: Add a rebalancing function that takes into account a specified rebalancing frequency.
def rebalance_portfolio(signals, frequency=20):
    """Rebalance the portfolio based on the signals at a specified frequency."""
    rebalance_positions = signals.copy()
    rebalance_positions[::frequency] = signals[::frequency]  # Rebalance at the specified frequency
    return rebalance_positions
