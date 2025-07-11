# 2. 데이터를 다운로드하고, 수익률을 계산하는것 까지 포함된 모듈

import yfinance as yf
import pandas as pd
import os
from pathlib import Path
from typing import List, Optional

class DataManager:
    """
    주가 데이터를 다운로드하고 수익률을 계산하는 클래스.
    """
    def get_data(
        self,
        tickers: List[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        save_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        지정된 티커의 주가 데이터를 다운로드하고 'Adj Close'를 사용합니다.

        Args:
            tickers (List[str]): 티커 심볼 리스트.
            start (pd.Timestamp): 데이터 시작일.
            end (pd.Timestamp): 데이터 종료일.
            save_path (Optional[Path]): 데이터를 저장할 CSV 파일 경로.

        Returns:
            pd.DataFrame: 처리된 주가 데이터프레임.
        """
        print(f"'{tickers}' 종목 데이터 다운로드 시도 중...")
        data = yf.download(tickers, start=start, end=end, auto_adjust=False)
        if data.empty:
            raise ValueError("데이터 다운로드에 실패했거나 데이터가 없습니다.")
        
        price_data = data['Adj Close'].dropna()
        present_tickers = [ticker for ticker in tickers if ticker in price_data.columns]
        price_data = price_data[present_tickers]
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            price_data.to_csv(save_path)
            print(f"수정 종가 데이터가 로컬에 저장되었습니다: {save_path}")
            
        return price_data

    def calculate_returns(self, data: pd.DataFrame, period: str = 'daily') -> pd.DataFrame:
        """
        주가 데이터를 기반으로 수익률을 계산합니다.

        Args:
            data (pd.DataFrame): 주가 데이터프레임.
            period (str): 수익률 계산 주기 ('daily', 'weekly', 'monthly').

        Returns:
            pd.DataFrame: 계산된 수익률 데이터프레임.
        """
        if period == 'weekly':
            return data.resample('W-FRI').last().pct_change().dropna()
        if period == 'monthly':
            return data.resample('M').last().pct_change().dropna()
        return data.pct_change().dropna()