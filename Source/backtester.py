# 4. 성과분석 및 백테스팅 모듈 1

import pandas as pd
import numpy as np
from typing import Dict, Any, Callable
from pathlib import Path

class DynamicBacktester:
    """동적 롤링 윈도우 백테스팅을 수행하는 클래스"""
    
    def __init__(self, prices, strategy_func, strategy_params, initial_investment, window, frequency, slippage):
        self.prices = prices
        self.strategy_func = strategy_func
        self.strategy_params = strategy_params
        self.initial_investment = initial_investment
        self.window = window
        self.frequency = frequency
        self.slippage = slippage
        self.portfolio_value = pd.Series(index=prices.index, dtype=float)
        self.weights_history = []

    def run(self):
        """백테스팅 실행"""
        rebal_dates = pd.to_datetime(self.prices.resample(self.frequency.replace('Y', 'YE')).last().index)
        rebal_dates = rebal_dates[rebal_dates >= self.prices.index[0] + pd.DateOffset(years=self.window)]
        rebal_dates = rebal_dates[rebal_dates <= self.prices.index[-1]]

        last_weights = pd.Series(1 / len(self.prices.columns), index=self.prices.columns)
        self.portfolio_value.iloc[0] = self.initial_investment
        
        for i in range(1, len(self.prices)):
            prev_date, current_date = self.prices.index[i-1], self.prices.index[i]
            
            # 이전 날짜의 포트폴리오 가치로 현재 날짜의 가치를 초기화
            self.portfolio_value.loc[current_date] = self.portfolio_value.loc[prev_date]

            # 리밸런싱 날짜 확인
            if current_date in rebal_dates:
                print(f"{current_date.date()}: 가중치 재계산 중...")
                window_prices = self.prices.loc[:current_date].tail(self.window * 252) # 근사치
                
                new_weights = self.strategy_func(window_prices, **self.strategy_params)
                
                if new_weights is not None:
                    new_weights = pd.Series(new_weights, index=self.prices.columns)
                    
                    # 슬리피지 적용
                    turnover = (new_weights - last_weights).abs().sum() / 2
                    rebalancing_cost = self.portfolio_value.loc[current_date] * turnover * self.slippage
                    self.portfolio_value.loc[current_date] -= rebalancing_cost
                    
                    last_weights = new_weights
                    self.weights_history.append({'date': current_date, 'weights': last_weights})
                else:
                    print(f"{current_date.date()}: 최적화 실패, 이전 가중치 사용.")
                    self.weights_history.append({'date': current_date, 'weights': last_weights})
            
            # 일일 수익률 반영
            daily_returns = self.prices.loc[current_date] / self.prices.loc[prev_date] - 1
            portfolio_daily_return = (daily_returns * last_weights).sum()
            self.portfolio_value.loc[current_date] *= (1 + portfolio_daily_return)

        return self.portfolio_value.dropna()

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

    @classmethod
    def generate_summary_report(cls, portfolio_results: Dict[str, Any], save_path: 'Path'):
        """
        여러 포트폴리오의 성과를 요약하고 CSV 파일로 저장합니다.
        
        Args:
            portfolio_results (Dict[str, Any]): 키는 모델 이름, 값은 성과 딕셔너리.
            save_path (Path): 보고서를 저장할 경로.
        """
        report_data = []
        for name, metrics in portfolio_results.items():
            row = {k: v for k, v in metrics.items() if k not in ['value', 'drawdown_series']}
            row['Model'] = name
            report_data.append(row)
        
        if not report_data:
            print("보고할 데이터가 없습니다.")
            return

        report_df = pd.DataFrame(report_data).set_index('Model')
        
        # 디렉터리 존재 확인 및 생성
        save_path.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(save_path)
        print(f"\n상세 성과 보고서가 저장되었습니다: {save_path}")
        return report_df