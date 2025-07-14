# 4. 성과분석 및 백테스팅 모듈 1

import pandas as pd
import numpy as np
from typing import Dict, Any, Callable

class DynamicBacktester:
    """
    롤링 윈도우 기반의 동적 백테스팅을 수행하는 클래스.
    """
    def __init__(self, prices: pd.DataFrame, strategy_func: Callable, strategy_params: Dict,
                 initial_investment: float, window: int, frequency: str, slippage: float):
        self.prices = prices
        self.strategy_func = strategy_func
        self.strategy_params = strategy_params
        self.initial_investment = initial_investment
        self.window = window * 252 # 연 단위 윈도우를 일 단위로 변환
        self.frequency = frequency
        self.slippage = slippage

    def run(self) -> pd.Series:
        """동적 리밸런싱 시뮬레이션을 실행합니다."""
        portfolio_log = pd.DataFrame(index=self.prices.index, columns=['total_value'])
        portfolio_log.iloc[0] = self.initial_investment
        
        rebal_dates = pd.to_datetime(self.prices.resample(self.frequency).last().index)
        current_weights = np.zeros(len(self.prices.columns))
        
        for i in range(1, len(self.prices)):
            prev_date, curr_date = self.prices.index[i-1], self.prices.index[i]
            
            # 리밸런싱 날짜에 도달하면 가중치 재계산
            if curr_date in rebal_dates:
                # 롤링 윈도우에 해당하는 데이터 추출
                window_start_idx = max(0, i - self.window)
                rolling_prices = self.prices.iloc[window_start_idx:i]
                
                if not rolling_prices.empty:
                    print(f"{curr_date.date()}: 가중치 재계산 중...")
                    new_weights = self.strategy_func(rolling_prices, **self.strategy_params)
                    
                    # 거래 비용 계산
                    turnover = np.abs(new_weights - current_weights).sum() / 2
                    cost = portfolio_log.loc[prev_date, 'total_value'] * turnover * self.slippage
                    portfolio_log.loc[prev_date, 'total_value'] -= cost
                    
                    current_weights = new_weights

            # 일별 수익률 계산 및 포트폴리오 가치 업데이트
            daily_returns = (self.prices.loc[curr_date] / self.prices.loc[prev_date] - 1)
            portfolio_return = (daily_returns * current_weights).sum()
            portfolio_log.loc[curr_date, 'total_value'] = portfolio_log.loc[prev_date, 'total_value'] * (1 + portfolio_return)
            
        return portfolio_log['total_value'].dropna().astype(float)

    @staticmethod
    def calculate_performance_metrics(value_series: pd.Series, risk_free_rate: float) -> Dict[str, Any]:
        """모든 상세 성과 지표를 계산하여 딕셔너리로 반환합니다."""
        if value_series.empty or len(value_series) < 2:
            return {"CAGR": 0.0, "Annualized Volatility": 0.0, "MDD": 0.0, "Sharpe Ratio": 0.0, "Calmar Ratio": 0.0, "Final Value": 0.0}
            
        total_years = (value_series.index[-1] - value_series.index[0]).days / 365.25
        cagr = (value_series.iloc[-1] / value_series.iloc[0])**(1/total_years) - 1 if total_years > 0 else 0
        
        peak = value_series.expanding(min_periods=1).max()
        drawdown_series = (value_series / peak) - 1.0
        mdd = drawdown_series.min()
        
        daily_returns = value_series.pct_change().dropna()
        annual_volatility = daily_returns.std() * np.sqrt(252)
        
        sharpe = (cagr - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        calmar = cagr / abs(mdd) if mdd < 0 else 0
        
        return {
            "CAGR": cagr, "Annualized Volatility": annual_volatility, "MDD": mdd,
            "Sharpe Ratio": sharpe, "Calmar Ratio": calmar,
            "Final Value": value_series.iloc[-1], "drawdown_series": drawdown_series
        }