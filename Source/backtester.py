# 성과분석 및 백테스팅 모듈 1

import pandas as pd
import numpy as np

def run_rebalancing_simulation(prices_df, target_weights, initial_investment, frequency, threshold, slippage):
    """리밸런싱과 슬리피지를 적용하여 포트폴리오 가치를 시뮬레이션합니다."""
    target_weights = pd.Series(target_weights, index=prices_df.columns)
    portfolio_log = pd.DataFrame(index=prices_df.index, columns=['total_value'] + list(prices_df.columns))
    portfolio_log.iloc[0] = [initial_investment] + list(initial_investment * target_weights)
    
    rebal_check_dates = pd.to_datetime(prices_df.resample(frequency).last().index)
    rebalance_count = 0

    for i in range(1, len(prices_df)):
        prev_date, curr_date = prices_df.index[i-1], prices_df.index[i]
        daily_returns = prices_df.loc[curr_date] / prices_df.loc[prev_date]
        current_asset_values = portfolio_log.loc[prev_date, prices_df.columns] * daily_returns
        current_total_value = current_asset_values.sum()
        
        if curr_date in rebal_check_dates:
            current_weights = current_asset_values / current_total_value
            if np.abs(current_weights - target_weights).max() > threshold:
                rebalance_count += 1
                turnover = np.abs(current_total_value * target_weights - current_asset_values).sum() / 2
                cost = turnover * slippage
                current_total_value -= cost
                current_asset_values = current_total_value * target_weights

        portfolio_log.loc[curr_date] = [current_total_value] + list(current_asset_values)
        
    print(f"Total rebalances: {rebalance_count}")
    return portfolio_log['total_value'].astype(float)

def calculate_performance_metrics(value_series, risk_free_rate):
    """성과 지표(CAGR, MDD, Sharpe, Calmar)를 계산합니다."""
    # CAGR
    total_years = (value_series.index[-1] - value_series.index[0]).days / 365.25
    cagr = (value_series.iloc[-1] / value_series.iloc[0])**(1/total_years) - 1 if total_years > 0 else 0
    
    # MDD
    peak = value_series.expanding(min_periods=1).max()
    drawdown = (value_series / peak) - 1.0
    mdd = drawdown.min()
    
    # Sharpe
    daily_returns = value_series.pct_change().dropna()
    if daily_returns.std() == 0:
        sharpe = 0
    else:
        excess_returns = daily_returns - ((1 + risk_free_rate)**(1/252) - 1)
        sharpe = (excess_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))
        
    # Calmar
    calmar = cagr / abs(mdd) if mdd < 0 else 0
    
    return cagr, mdd, sharpe, calmar, drawdown
