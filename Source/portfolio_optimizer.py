# 3. 최적화를 위한 성과 지표 계산 모듈

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import List, Dict, Any, Tuple, Callable

class PortfolioOptimizer:
    """
    다양한 전략에 따라 포트폴리오를 최적화하는 클래스.
    """
    def __init__(self, returns: pd.DataFrame, annualizer: int = 252):
        """
        Args:
            returns (pd.DataFrame): 최적화에 사용할 수익률 데이터.
            annualizer (int): 연간화 계수 (daily=252, weekly=52, monthly=12).
        """
        self.returns = returns
        self.num_assets = returns.shape[1]
        self.tickers = returns.columns.tolist()
        self.annualizer = annualizer

    def calculate_metrics(self, weights: np.ndarray) -> Dict[str, float]:
        """
        포트폴리오의 주요 성과 지표를 계산합니다.
        (수정됨: Calmar Ratio 계산 로직 복원)
        """
        weights = np.array(weights)
        portfolio_returns = (self.returns * weights).sum(axis=1)
        
        # 기본 지표 계산
        annual_return = portfolio_returns.mean() * self.annualizer
        annual_volatility = portfolio_returns.std() * np.sqrt(self.annualizer)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Calmar Ratio 계산을 위한 추가 지표
        cumulative_returns = (1 + portfolio_returns).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns / peak) - 1
        max_drawdown = drawdown.min()
        
        num_days = len(cumulative_returns)
        cagr = (cumulative_returns.iloc[-1])**(self.annualizer/num_days) - 1 if num_days > 0 else 0

        epsilon = 1e-10
        calmar_ratio = cagr / (abs(max_drawdown) + epsilon)
        
        return {
            "return": annual_return, 
            "volatility": annual_volatility, 
            "sharpe": sharpe_ratio,
            "calmar": calmar_ratio # 반환값에 calmar 추가
        }

    def _get_objective_function(self, name: str) -> Callable:
        """
        최적화 목표 함수를 반환합니다.
        (수정됨: neg_calmar_ratio가 올바른 값을 사용하도록 변경)
        """
        if name == 'neg_sharpe_ratio':
            return lambda w: -self.calculate_metrics(w)['sharpe']
        if name == 'volatility':
            return lambda w: self.calculate_metrics(w)['volatility']
        if name == 'neg_calmar_ratio':
             return lambda w: -self.calculate_metrics(w)['calmar'] # Placeholder 수정
        raise ValueError(f"'{name}'은(는) 유효한 목표 함수가 아닙니다.")

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
        """위험 패리티 포트폴리오의 가중치를 계산합니다."""
        vols = returns.std()
        inv_vols = 1.0 / vols
        return (inv_vols / inv_vols.sum()).values

    def run_and_save_all_strategies(self, strategies: Dict, weight_constraints: Dict, save_path: 'Path'):
        """
        모든 활성화된 최적화 전략을 실행하고 결과를 Excel 파일로 저장합니다.
        
        Args:
            strategies (Dict): config 파일의 STRATEGIES 딕셔너리.
            weight_constraints (Dict): 최소/최대 가중치 제약.
            save_path (Path): 결과를 저장할 Excel 파일 경로.
        """
        bounds = tuple((weight_constraints['min'], weight_constraints['max']) for _ in range(self.num_assets))
        base_constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        all_weights_dict = {}
        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            for name, params in strategies.items():
                if not params.get('enabled', False):
                    continue

                print(f"  - '{name}' 전략 최적화 중...")
                
                if params.get('is_optimizer') is False: # Risk Parity
                    optimal_weights = self.get_risk_parity_weights(self.returns)
                else:
                    constraints = base_constraints.copy()
                    target_return = params.get('target_return')
                    if target_return is not None:
                         constraints.append({'type': 'ineq', 'fun': lambda w: self.calculate_metrics(w)['return'] - target_return})
                    
                    optimal_weights = self.run_optimization(
                        objective_name=params['objective'],
                        constraints=constraints,
                        bounds=bounds
                    )
                
                weights_df = pd.DataFrame({'Ticker': self.tickers, 'Optimal_Weight': optimal_weights})
                weights_df.to_excel(writer, sheet_name=name, index=False)
                all_weights_dict[name] = weights_df

        print(f"\n최적 가중치가 '{save_path}'에 저장되었습니다.")
        return all_weights_dict
