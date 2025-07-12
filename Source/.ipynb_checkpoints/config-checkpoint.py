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
    'SPY', 'QQQ', 'EFA', 'EEM', 'SSO', 'TQQQ', # 주식
    'TLT', 'IEF', 'SHY',       # 채권
    'GLD', 'DBC',              # 원자재
    'BTC-USD', 'ETH-USD', 'XRP-USD'                  # 암호화폐
]
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=5*365) # 분석 기간을 5년으로 조정

# 2. 최적화 전략별 파라미터 설정
TARGET_RETURN = 0.18
WEIGHT_CAP = 0.30
NUM_RANDOM_PORTFOLIOS = 25000 # 효율적 투자선 계산 시 생성할 포트폴리오 수

# 3. 리밸런싱 및 성과 분석 설정
INITIAL_INVESTMENT_USD = 100_000_000
REBALANCING_FREQUENCY = 'M'
REBALANCING_THRESHOLD = 0.05
SLIPPAGE_PCT = 0.001
RISK_FREE_RATE_ANNUAL = 0.00

# 4. 실행할 최적화 전략 목록
# (노트북을 수정할 필요 없이 여기에서 실행 여부를 제어합니다)
STRATEGIES = {
    'max_calmar':       {'objective': 'neg_calmar_ratio', 'enabled': True},
    'risk_parity':      {'is_optimizer': False, 'enabled': True},
    'min_variance':     {'objective': 'volatility', 'enabled': True},
    'daily_30_cap':     {'objective': 'neg_sharpe_ratio', 'bounds_cap': WEIGHT_CAP, 'enabled': True},
    'target_return':    {'objective': 'volatility', 'target_return': TARGET_RETURN, 'enabled': True},
    # 'max_sharpe':       {'objective': 'neg_sharpe_ratio', 'enabled': False}, # 필요 시 주석 해제하여 사용
}
