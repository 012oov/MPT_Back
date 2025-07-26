import pandas as pd
import numpy as np
import config
from data_manager import DataManager
from portfolio_optimizer import PortfolioOptimizer
from backtester import DynamicBacktester
from visualizer import Visualizer
from utils import format_model_name

def main():
    """메인 동적 백테스팅 실행 함수"""
    # --- 1. 데이터 준비 ---
    dm = DataManager()
    stock_data = dm.get_data(config.STOCKS, config.START_DATE, config.END_DATE, save_path=config.RAW_DATA_PATH)

    # --- 2. 전략별 백테스팅 실행 ---
    portfolio_results = {}
    weights_history_by_strategy = {}

    def strategy_wrapper(prices, **params):
        """최적화 로직을 백테스터에 전달하기 위한 래퍼 함수"""
        returns = dm.calculate_returns(prices, 'daily')
        optimizer = PortfolioOptimizer(returns)
        
        # Risk Parity는 별도 처리
        if params.get('is_risk_parity', False):
            return optimizer.get_risk_parity_weights(returns)
        
        objective_name = params.get('objective')
            
        # 최적화 함수 이름 매핑 (config.py의 objective 값과 일치시켜야 함)
        func_map = {
            'neg_sharpe_ratio': 'neg_sharpe_ratio',
            'portfolio_variance': 'portfolio_variance',
            'neg_calmar_ratio': 'neg_calmar_ratio',
        }
        
        # Target Return의 경우, objective는 portfolio_variance를 사용
        if params.get('target_return') is not None:
            objective_name = 'portfolio_variance'

        if objective_name not in func_map:
            raise ValueError(f"'{objective_name}'에 해당하는 최적화 함수를 찾을 수 없습니다.")
        
        objective_func_name = func_map[objective_name]

        # 나머지 최적화 전략
        num_assets = len(prices.columns)
        bounds = tuple(((config.WEIGHT_CONSTRAINTS['min'], config.WEIGHT_CONSTRAINTS['max'])) for _ in range(num_assets))
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        target_return = params.get('target_return')
        if target_return is not None:
            # 목표 수익률 제약조건 추가
            constraints.append({
                'type': 'eq', 
                'fun': lambda w: optimizer.calculate_metrics(w)['return'] - target_return
            })
            
        return optimizer.run_optimization(objective_func_name, constraints=constraints, bounds=bounds)

    # 활성화된 전략 실행
    for name, params in config.STRATEGIES.items():
        if not params.get('enabled', False):
            continue
        
        model_name = format_model_name(name)
        print(f"\n--- '{model_name}' 모델 동적 백테스팅 시작 ---")
        
        backtester = DynamicBacktester(
            prices=stock_data,
            strategy_func=strategy_wrapper,
            strategy_params=params,
            initial_investment=config.INITIAL_INVESTMENT_USD,
            window=config.ROLLING_WINDOW_YEARS,
            frequency=config.REBALANCE_FREQUENCY,
            slippage=config.SLIPPAGE_PCT
        )
        portfolio_value = backtester.run()
        
        performance = DynamicBacktester.calculate_performance_metrics(portfolio_value, config.RISK_FREE_RATE_ANNUAL)
        portfolio_results[model_name] = {'value': portfolio_value, **performance}
        weights_history_by_strategy[model_name] = backtester.weights_history

    # --- 3. 벤치마크 성과 계산 ---
    benchmarks = {'SPY': 'SPY', 'BTC-USD': 'BTC-USD'}
    for bm_name, bm_ticker in benchmarks.items():
        if bm_ticker in stock_data.columns:
            bm_price_data = stock_data[[bm_ticker]].dropna()
            bm_returns = bm_price_data.pct_change().dropna()
            portfolio_value = (config.INITIAL_INVESTMENT_USD * (1 + bm_returns).cumprod()).iloc[:, 0]
            performance = DynamicBacktester.calculate_performance_metrics(portfolio_value, config.RISK_FREE_RATE_ANNUAL)
            portfolio_results[f"Benchmark ({bm_name})"] = {'value': portfolio_value, **performance}

    # --- 4. 최종 결과 요약 및 저장 ---
    if portfolio_results:
        print("\n\n--- 최종 성과 요약 ---")
        for name, metrics in portfolio_results.items():
            print(f"  - {name}: CAGR={metrics['CAGR']:.2%}, MDD={metrics['MDD']:.2%}, Sharpe={metrics['Sharpe Ratio']:.2f}, Calmar={metrics['Calmar Ratio']:.2f}")
            if name in weights_history_by_strategy and weights_history_by_strategy[name]:
                print(f"    - '{name}' 모델 리밸런싱 비중 내역:")
                for rebalance_info in weights_history_by_strategy[name]:
                    rebalance_date = rebalance_info['date']
                    weights = rebalance_info['weights']
                    print(f"      - 리밸런싱 날짜: {rebalance_date.date()}")

                    sorted_weights = weights.sort_values(ascending=False)

                    for asset, weight in sorted_weights.items():
                        if weight > 1e-4:
                            print(f"        - {asset}: {weight:.2%}")
                print()  # 모델별 구분을 위한 빈 줄 추가

        # backtester 모듈을 사용하여 보고서 생성 및 저장
        report_df = DynamicBacktester.generate_summary_report(portfolio_results, config.PERFORMANCE_REPORT_SAVE_PATH)
        if report_df is not None:
            print("\n--- 최종 리포트 ---")

            # Pandas 디스플레이 옵션 설정으로 터미널에서 잘리는 문제 해결
            pd.set_option('display.width', None)
            pd.set_option('display.max_columns', None)

            # 터미널 출력용으로 데이터프레임 복사 및 서식 지정
            display_df = report_df.copy()
            formatters = {
                'Final Value': '{:,.0f}'.format,
                'CAGR': '{:.2%}'.format,
                'MDD': '{:.2%}'.format,
                'Annualized Volatility': '{:.2%}'.format,
                'Sharpe Ratio': '{:.2f}'.format,
                'Calmar Ratio': '{:.2f}'.format
            }
            for col, formatter in formatters.items():
                if col in display_df.columns:
                    display_df[col] = display_df[col].map(formatter)
            
            print(display_df)

        # --- 5. 시각화 ---
        visualizer = Visualizer(portfolio_results)
        visualizer.plot_performance_comparison()

if __name__ == '__main__':
    main() 