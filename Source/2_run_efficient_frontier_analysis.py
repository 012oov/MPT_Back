import pandas as pd
import config
from data_manager import DataManager
from portfolio_optimizer import PortfolioOptimizer
from visualizer import Visualizer
from utils import format_model_name

def run_analysis():
    """
    정적 최적화부터 효율적 투자선 시각화까지 모든 분석을 한 번에 실행합니다.
    """
    print("--- 효율적 투자선 분석 전체 프로세스 시작 ---")

    # --- 1. 데이터 준비 ---
    print("\n[1/4] 데이터 로드 중...")
    dm = DataManager()
    stock_data = dm.get_data(config.STOCKS, config.START_DATE, config.END_DATE, save_path=config.RAW_DATA_PATH)
    daily_returns = dm.calculate_returns(stock_data, 'daily')

    # --- 2. 포트폴리오 최적화 및 결과 저장 ---
    print("\n[2/4] 포트폴리오 최적화 및 가중치 계산 중...")
    optimizer = PortfolioOptimizer(daily_returns)
    all_weights_dict = optimizer.run_and_save_all_strategies(
        strategies=config.STRATEGIES,
        weight_constraints=config.WEIGHT_CONSTRAINTS,
        save_path=config.OPTIMAL_WEIGHTS_SAVE_PATH
    )

    # --- 3. 무작위 포트폴리오 생성 및 최적 지점 계산 ---
    print("\n[3/4] 시각화를 위한 데이터 준비 중...")
    # 무작위 포트폴리오 생성
    random_ports = optimizer.generate_random_portfolios(config.NUM_RANDOM_PORTFOLIOS)
    print(f"  - {config.NUM_RANDOM_PORTFOLIOS}개의 무작위 포트폴리오 생성 완료.")

    # 저장된 최적 가중치를 기반으로 성과 지표 계산
    optimal_points = {}
    for name, df in all_weights_dict.items():
        weights = df['Optimal_Weight'].values
        metrics = optimizer.calculate_metrics(weights)
        model_name = format_model_name(name)
        optimal_points[model_name] = metrics
    
    print("  - 최적화된 포트폴리오 지점 계산 완료.")
    for name, point in optimal_points.items():
        print(f"    * {name}: Return={point['return']:.2%}, Volatility={point['volatility']:.2%}, Sharpe={point['sharpe']:.2f}")

    # --- 4. 시각화 ---
    print("\n[4/4] 효율적 투자선 시각화 중...")
    visualizer = Visualizer(results={}) # 시각화 클래스 초기화
    visualizer.plot_efficient_frontier(random_ports, optimal_points)

    print("\n--- 모든 작업이 성공적으로 완료되었습니다. ---")

if __name__ == '__main__':
    run_analysis() 