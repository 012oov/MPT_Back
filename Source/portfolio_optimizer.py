# 최적화를 위한 성과 지표 계산 모듈

import numpy as np
from scipy.optimize import minimize

# --- 포트폴리오 성과 지표 계산 ---
def portfolio_metrics(weights, returns, annualizer=252):
    """포트폴리오의 주요 성과 지표를 계산합니다."""
    portfolio_returns = (returns * weights).sum(axis=1)
    
    annual_return = portfolio_returns.mean() * annualizer
    annual_volatility = portfolio_returns.std() * np.sqrt(annualizer)
    
    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
    
    cumulative_returns = (1 + portfolio_returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min()
    
    num_days = len(cumulative_returns)
    cagr = (cumulative_returns.iloc[-1])**(annualizer/num_days) - 1 if num_days > 0 else 0

    epsilon = 1e-10
    calmar_ratio = cagr / (abs(max_drawdown) + epsilon)
    
    return annual_return, annual_volatility, sharpe_ratio, calmar_ratio

# --- 최적화 목표 함수 (최소화 대상) ---
def neg_sharpe_ratio(weights, returns, annualizer):
    return -portfolio_metrics(weights, returns, annualizer)[2]

def portfolio_volatility(weights, returns, annualizer):
    return portfolio_metrics(weights, returns, annualizer)[1]

def neg_calmar_ratio(weights, returns, annualizer):
    return -portfolio_metrics(weights, returns, annualizer)[3]

# --- 범용 최적화 실행 함수 ---
def run_optimization(objective_func, returns, annualizer, constraints, bounds, init_guess):
    """주어진 목표 함수에 따라 최적화를 수행합니다."""
    result = minimize(objective_func, init_guess, args=(returns, annualizer),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# --- 특정 전략 함수 ---
def get_risk_parity_weights(returns):
    """위험 패리티 포트폴리오의 가중치를 계산합니다."""
    vols = returns.std()
    inv_vols = 1.0 / vols
    return (inv_vols / inv_vols.sum()).values