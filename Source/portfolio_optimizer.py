# 3. 최적화를 위한 성과 지표 계산 모듈

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import List, Dict, Any, Tuple, Callable
import config

class PortfolioOptimizer:
    """포트폴리오 최적화 및 관련 계산을 수행하는 클래스"""
    def __init__(self, returns):
        if not isinstance(returns, pd.DataFrame) or returns.empty:
            raise ValueError("수익률 데이터는 비어 있지 않은 pandas DataFrame이어야 합니다.")
        
        self.returns = returns
        self.cov_matrix = returns.cov() * 252  # 연간 공분산 행렬 계산
        self.num_assets = returns.shape[1]
        self.tickers = returns.columns.tolist()
        self.annualizer = 252 # 기본값 설정

    def calculate_metrics(self, weights, period='daily'):
        """포트폴리오의 주요 지표(수익률, 변동성, MDD)를 계산합니다."""
        portfolio_returns = np.dot(self.returns, weights)
        
        # 기간에 따른 연율화 계수
        annualizing_factor = 252 if period == 'annual' else 1
        
        # 수익률 계산
        expected_return = np.sum(self.returns.mean() * weights) * annualizing_factor
        
        # 변동성 계산
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights))) * np.sqrt(annualizing_factor)
        
        # MDD 계산
        cumulative_returns = (1 + pd.Series(portfolio_returns)).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns / peak) - 1
        mdd = drawdown.min()
        
        return {
            'return': expected_return,
            'volatility': portfolio_volatility,
            'mdd': mdd
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

    def run_optimization(self, objective_func: str, constraints: list, bounds: tuple):
        """
        주어진 목적 함수와 제약 조건으로 포트폴리오 최적화를 실행합니다.
        
        Args:
            objective_func (str): 최적화할 목적 함수의 이름 (e.g., 'sharpe_ratio').
            constraints (list): 제약 조건 딕셔너리의 리스트.
            bounds (tuple): 각 자산의 가중치 범위.

        Returns:
            np.ndarray: 최적화된 포트폴리오 가중치.
        """
        num_assets = len(self.returns.columns)
        initial_weights = np.array([1/num_assets] * num_assets)

        if not hasattr(self, objective_func) or not callable(getattr(self, objective_func)):
             raise ValueError(f"'{objective_func}'는 유효한 최적화 함수가 아닙니다.")
        
        func_to_optimize = getattr(self, objective_func)
        
        # 목적 함수에 따라 추가 인수 설정
        if objective_func == 'neg_sharpe_ratio':
            args = (config.RISK_FREE_RATE_ANNUAL,)
        else:
            args = ()

        result = minimize(
            fun=func_to_optimize,
            x0=initial_weights,
            args=args, 
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            return result.x
        else:
            # 최적화 실패 시 경고 메시지를 출력하고 None을 반환
            print(f"Warning: Optimization failed. Message: {result.message}")
            return None

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
                        objective_func=params['objective'],
                        constraints=constraints,
                        bounds=bounds
                    )
                
                weights_df = pd.DataFrame({'Ticker': self.tickers, 'Optimal_Weight': optimal_weights})
                weights_df.to_excel(writer, sheet_name=name, index=False)
                all_weights_dict[name] = weights_df

        print(f"\n최적 가중치가 '{save_path}'에 저장되었습니다.")
        return all_weights_dict

    def sharpe_ratio(self, weights, risk_free_rate=0.02):
        """포트폴리오의 샤프 지수를 계산합니다."""
        metrics = self.calculate_metrics(weights, 'annual')
        returns = metrics['return']
        volatility = metrics['volatility']
        sharpe = (returns - risk_free_rate) / volatility if volatility > 0 else 0
        return sharpe

    def neg_sharpe_ratio(self, weights, risk_free_rate=0.02):
        """최대화를 위해 샤프 지수에 음수를 취한 값을 반환합니다."""
        return -self.sharpe_ratio(weights, risk_free_rate)

    def calmar_ratio(self, weights):
        """포트폴리오의 칼마 지수를 계산합니다."""
        metrics = self.calculate_metrics(weights, 'annual')
        cagr = metrics['return']
        mdd = metrics['mdd']
        calmar = cagr / abs(mdd) if mdd < 0 else 0
        return calmar

    def neg_calmar_ratio(self, weights):
        """최대화를 위해 칼마 지수에 음수를 취한 값을 반환합니다."""
        return -self.calmar_ratio(weights)

    def portfolio_variance(self, weights):
        """포트폴리오의 분산(변동성)을 계산합니다."""
        return self.calculate_metrics(weights, 'annual')['volatility']**2
