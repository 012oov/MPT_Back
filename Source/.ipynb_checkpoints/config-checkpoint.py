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
START_DATE = END_DATE - timedelta(days=10*365)

# 2. 동적 백테스팅 설정
ROLLING_WINDOW_YEARS = 3  # 최적화에 사용할 과거 데이터 기간 (예: Y)
REBALANCE_FREQUENCY = 'Y' # 리밸런싱 및 가중치 재계산 주기 ('Y': 연별, 'Q': 분기별)

# 3. 최적화 전략별 파라미터 설정
TARGET_RETURN = 0.18
WEIGHT_CONSTRAINTS = {'min': 0.01, 'max': 0.30}

# 4. 리밸런싱 및 성과 분석 설정
INITIAL_INVESTMENT_USD = 100_000_000
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
