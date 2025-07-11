# 사용자가 이 파일만 수정하여 모든 변수를 제어 가능. 여타 프로그램에서 "일반설정" 세팅이라고 생각하면 됨

from datetime import datetime, timedelta
from pathlib import Path

# --- 기본 경로 설정 (자동, 수정 불필요) ---
try:
    # .py 파일로 실행할 때를 위한 경로
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    # Jupyter Notebook 환경을 위한 경로
    PROJECT_ROOT = Path.cwd().parent

DATA_DIR = PROJECT_ROOT / "Data"
RESULTS_DIR = PROJECT_ROOT / "Results"
RAW_DATA_PATH = DATA_DIR / "Raw data" / "downloaded_stock_prices.csv"
OPTIMAL_WEIGHTS_SAVE_PATH = RESULTS_DIR / "optimized_weights_all_periods.xlsx"


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


# 2. 최적화 전략별 파라미터 설정
TARGET_RETURN = 0.18  # '목표 수익률' 모델을 위한 연간 목표 수익률
WEIGHT_CAP = 0.30     # '최대 비중 제한' 모델을 위한 개별 자산 최대 보유 비중


# 3. 리밸런싱 및 성과 분석 설정
INITIAL_INVESTMENT_USD = 100_000_000
REBALANCING_FREQUENCY = 'Y' # 리밸런싱 주기 ('M': 월별, 'Q': 분기별, 'Y': 연별)
REBALANCING_THRESHOLD = 0.05 # 목표 비중에서 이 값(5%) 이상 벗어나면 리밸런싱 실행
SLIPPAGE_PCT = 0.001 # 거래비용 (0.1%)
RISK_FREE_RATE_ANNUAL = 0.00 # 연간 무위험 수익률