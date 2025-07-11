# 데이터를 다운로드하고, 수익률을 계산하는것 까지 포함된 모듈

import yfinance as yf
import pandas as pd
import os

def get_data(tickers, start, end, save_path=None):
    """지정된 티커의 주가 데이터를 다운로드하고, 'Adj Close'를 사용하며, 컬럼 순서를 재정렬합니다."""
    print(f"'{tickers}' 종목 데이터 다운로드 시도 중...")
    data = yf.download(tickers, start=start, end=end, auto_adjust=False)
    if data.empty: raise ValueError("데이터 다운로드 실패")
    
    price_data = data['Adj Close'].dropna()
    present_tickers = [ticker for ticker in tickers if ticker in price_data.columns]
    price_data = price_data[present_tickers]
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        price_data.to_csv(save_path)
        print(f"수정 종가 데이터가 로컬에 저장되었습니다: {save_path}")
    return price_data

def calculate_returns(data, period='daily'):
    """주가 데이터를 기반으로 수익률을 계산합니다."""
    if period == 'weekly': return data.resample('W-FRI').last().pct_change().dropna()
    if period == 'monthly': return data.resample('M').last().pct_change().dropna()
    return data.pct_change().dropna()