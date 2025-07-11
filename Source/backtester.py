# 4. 성과분석 및 백테스팅 모듈 1

import pandas as pd
import numpy as np
from typing import Dict, Any

class Backtester:
    def __init__(self, prices: pd.DataFrame, weights: np.ndarray, initial_investment: float, 
                 frequency: str, threshold: float, slippage: float, risk_free_rate: float):
        self.prices = prices
        self.target_weights = pd.Series(weights, index=prices.columns)
        self.initial_investment = initial_investment
        self.frequency = frequency
        self.threshold = threshold
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate
        self.portfolio_log = pd.DataFrame()

    def run(self) -> pd.Series:
        self.portfolio_log = pd.DataFrame(index=self.prices.index, columns=['total_value'] + list(self.prices.columns))
        self.portfolio_log.iloc[0] = [self.initial_investment] + list(self.initial_investment * self.target_weights)
        
        rebal_check_dates = pd.to_datetime(self.prices.resample(self.frequency).last().index)
        rebalance_count = 0

        for i in range(1, len(self.prices)):
            prev_date, curr_date = self.prices.index[i-1], self.prices.index[i]
            daily_returns = self.prices.loc[curr_date] / self.prices.loc[prev_date]
            current_asset_values = self.portfolio_log.loc[prev_date, self.prices.columns] * daily_returns
            current_total_value = current_asset_values.sum()
            
            if curr_date in rebal_check_dates:
                current_weights = current_asset_values / current_total_value
                if np.abs(current_weights - self.target_weights).max() > self.threshold:
                    rebalance_count += 1
                    turnover = np.abs(current_total_value * self.target_weights - current_asset_values).sum() / 2
                    cost = turnover * self.slippage
                    current_total_value -= cost
                    current_asset_values = current_total_value * self.target_weights

            self.portfolio_log.loc[curr_date] = [current_total_value] + list(current_asset_values)
        
        print(f"Total rebalances: {rebalance_count}")
        return self.portfolio_log['total_value'].astype(float)

    @staticmethod
    def calculate_performance_metrics(value_series: pd.Series, risk_free_rate: float) -> Dict[str, Any]:
        """모든 상세 성과 지표를 계산하여 딕셔너리로 반환합니다."""
        # 데이터가 부족하여 계산할 수 없는 경우 기본값 반환
        if value_series.empty or len(value_series) < 2:
            return {
                "CAGR": 0.0, "Annualized Volatility": 0.0, "MDD": 0.0,
                "Sharpe Ratio": 0.0, "Calmar Ratio": 0.0,
                "Final Value": value_series.iloc[-1] if not value_series.empty else 0.0,
                "drawdown_series": pd.Series(dtype=float)
            }
            
        total_years = (value_series.index[-1] - value_series.index[0]).days / 365.25
        cagr = (value_series.iloc[-1] / value_series.iloc[0])**(1/total_years) - 1 if total_years > 0 else 0
        
        peak = value_series.expanding(min_periods=1).max()
        drawdown_series = (value_series / peak) - 1.0
        mdd = drawdown_series.min()
        
        daily_returns = value_series.pct_change().dropna()
        annual_volatility = daily_returns.std() * np.sqrt(252)
        
        if annual_volatility == 0:
            sharpe = 0
        else:
            excess_returns = daily_returns - ((1 + risk_free_rate)**(1/252) - 1)
            sharpe = (excess_returns.mean() * 252) / annual_volatility
            
        calmar = cagr / abs(mdd) if mdd < 0 else 0
        
        return {
            "CAGR": cagr,
            "Annualized Volatility": annual_volatility,
            "MDD": mdd,
            "Sharpe Ratio": sharpe, 
            "Calmar Ratio": calmar,
            "Final Value": value_series.iloc[-1],
            "drawdown_series": drawdown_series
        }
