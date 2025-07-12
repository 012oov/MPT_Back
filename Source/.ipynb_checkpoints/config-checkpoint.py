# 1. 사용자가 이 파일만 수정하여 모든 변수를 제어 가능. 여타 프로그램에서 "일반설정" 세팅이라고 생각하면 됨

from datetime import datetime, timedelta
from pathlib import Path

# --- 기본 경로 설정 (자동, 수정 불필요) ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd().parent

DATA_DIR = PROJECT_ROOT / "Data"
RESULTS_DIR = PROJECT_ROOT / "Results"
RAW_DATA_PATH = DATA_DIR / "Raw data" / "downloaded_stock_prices.csv"
OPTIMAL_WEIGHTS_SAVE_PATH = RESULTS_DIR / "optimized_weights_all_periods.xlsx"
PERFORMANCE_REPORT_SAVE_PATH = RESULTS_DIR / "performance_report.csv"

# ==============================================================================
# --- 사용자 설정 영역 ---
# ==============================================================================

# 1. 종목 및 기간 설정
STOCKS = [
    'SPY',  # SPDR S&P 500 ETF Trust
    'QQQ',  # Invesco QQQ Trust
    'IWM',  # iShares Russell 2000 ETF
    'VTI',  # Vanguard Total Stock Market ETF
    'AGG',  # iShares Core U.S. Aggregate Bond ETF

    # 섹터 대표 ETF
    'XLK',  # Technology Select Sector SPDR Fund
    'XLV',  # Health Care Select Sector SPDR Fund
    'XLF',  # Financial Select Sector SPDR Fund
    'XLE',  # Energy Select Sector SPDR Fund
    'XLP',  # Consumer Staples Select Sector SPDR Fund

    # 기타 전략 ETF
    'VIG',  # Vanguard Dividend Appreciation ETF
    'GLD',  # SPDR Gold Shares

    # 주요 암호화폐 (Yahoo Finance 티커 형식)
    'BTC-USD', # Bitcoin
    'ETH-USD', # Ethereum
    'SOL-USD', # Solana
    'XRP-USD', # Ripple
    'ADA-USD', # Cardano
    'AVAX-USD',# Avalanche
    'LINK-USD',# Chainlink
    'DOT-USD'  # Polkadot
]
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=5*365)

# 2. 최적화 전략별 파라미터 설정
TARGET_RETURN = 0.18  # '목표 수익률' 모델을 위한 연간 목표 수익률
NUM_RANDOM_PORTFOLIOS = 25000 # ✨ 효율적 투자선 계산 시 생성할 포트폴리오 수 추가 ✨

# 3. 범용 투자 비중 제약 설정
WEIGHT_CONSTRAINTS = {
    'min': 0.01,  # 개별 자산의 최소 보유 비중 (예: 1%)
    'max': 0.30   # 개별 자산의 최대 보유 비중 (예: 30%)
}

# 4. 리밸런싱 및 성과 분석 설정
INITIAL_INVESTMENT_USD = 100_000_000
REBALANCING_FREQUENCY = 'M'
REBALANCING_THRESHOLD = 0.05
SLIPPAGE_PCT = 0.001
RISK_FREE_RATE_ANNUAL = 0.00

# 5. 실행할 최적화 전략 목록
STRATEGIES = {
    'max_calmar':       {'objective': 'neg_calmar_ratio', 'enabled': True},
    'max_sharpe':       {'objective': 'neg_sharpe_ratio', 'enabled': True},
    'risk_parity':      {'is_optimizer': False, 'enabled': True},
    'min_variance':     {'objective': 'volatility', 'enabled': True},
    'target_return':    {'objective': 'volatility', 'target_return': TARGET_RETURN, 'enabled': True},
}
