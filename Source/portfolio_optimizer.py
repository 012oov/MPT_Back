# 3. 최적화를 위한 성과 지표 계산 모듈

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import List, Dict, Tuple, Callable

class PortfolioOptimizer:
    def __init__(self, returns: pd.DataFrame, annualizer: int = 252):
        self.returns = returns
        self.num_assets = returns.shape[1]
        self.tickers = returns.columns.tolist()
        self.annualizer = annualizer

    def calculate_metrics(self, weights: np.ndarray) -> Dict[str, float]:
        """포트폴리오의 주요 성과 지표를 계산합니다."""
        weights = np.array(weights)
        portfolio_returns = (self.returns * weights).sum(axis=1)
        
        annual_return = portfolio_returns.mean() * self.annualizer
        annual_volatility = portfolio_returns.std() * np.sqrt(self.annualizer)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        return {"return": annual_return, "volatility": annual_volatility, "sharpe": sharpe_ratio}

    def _get_objective_function(self, name: str) -> Callable:
        """최적화 목표 함수를 반환합니다."""
        if name == 'neg_sharpe_ratio':
            return lambda w: -self.calculate_metrics(w)['sharpe']
        if name == 'volatility':
            return lambda w: self.calculate_metrics(w)['volatility']
        # Calmar Ratio는 백테스팅 단계에서 계산되므로, 여기서는 샤프지수 기반으로 최적화합니다.
        # 더 정교한 Calmar 최적화는 별도 구현이 필요합니다.
        if name == 'neg_calmar_ratio':
             return lambda w: -self.calculate_metrics(w)['sharpe'] # Placeholder

    def run_optimization(self, objective_name: str, constraints: List[Dict], bounds: Tuple) -> np.ndarray:
        """주어진 목표 함수에 따라 최적화를 수행합니다."""
        objective_func = self._get_objective_function(objective_name)
        init_guess = np.array([1/self.num_assets] * self.num_assets)
        result = minimize(objective_func, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x

    def generate_random_portfolios(self, num_portfolios: int) -> pd.DataFrame:
        """무작위 포트폴리오를 생성하여 성과 지표를 계산합니다."""
        results_list = []
        for _ in range(num_portfolios):
            weights = np.random.random(self.num_assets)
            weights /= np.sum(weights)
            metrics = self.calculate_metrics(weights)
            results_list.append(metrics)
        return pd.DataFrame(results_list)

    @staticmethod
    def get_risk_parity_weights(returns: pd.DataFrame) -> np.ndarray:
        vols = returns.std()
        inv_vols = 1.0 / vols
        return (inv_vols / inv_vols.sum()).values
