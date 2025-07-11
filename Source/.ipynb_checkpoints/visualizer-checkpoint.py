# 5. 시각화 모듈

import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any

class Visualizer:
    def __init__(self, results: Dict[str, Any]):
        self.results = results

    def _format_model_name(self, sheet_name: str) -> str:
        """Excel 시트 이름을 그래프에 표시할 이름으로 변환합니다."""
        name = sheet_name.replace('_weights', '')
        if 'target_return' in name: return f"Target Return ({name.split('_')[-1]}%)"
        name_map = {
            'max_calmar': 'Max Calmar Ratio', 'risk_parity': 'Risk Parity',
            'min_variance': 'Minimum Variance', 'daily_30_cap': 'Daily (30% Cap)',
            'daily_max_sharpe': 'Daily (Max Sharpe)',
        }
        return name_map.get(name, name.replace('_', ' ').title())

    def plot_performance_comparison(self):
        """여러 포트폴리오의 성과와 MDD를 비교하는 그래프를 그립니다."""
        formatted_results = {self._format_model_name(k): v for k, v in self.results.items()}

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        colors = plt.colormaps['tab10'].colors
        
        for i, (name, data) in enumerate(formatted_results.items()):
            color = colors[i % len(colors)]
            ax1.plot(data['value'].index, data['value'], label=f"{name} (CAGR: {data['CAGR']*100:.2f}%)", color=color, linewidth=2 if 'Benchmark' in name else 1.5)
            ax2.plot(data['drawdown_series'].index, data['drawdown_series'] * 100, label=f"{name} (MDD: {data['MDD']*100:.2f}%)", color=color, alpha=0.8, linewidth=2 if 'Benchmark' in name else 1.5)

        ax1.set_title('Portfolio Value Comparison with Rebalancing & Slippage', fontsize=16)
        ax1.set_ylabel('Portfolio Value (USD)')
        ax1.legend(loc='upper left', fontsize='small')
        ax1.ticklabel_format(style='plain', axis='y')
        ax1.grid(True)
        
        ax2.set_title('Portfolio Drawdown (MDD) Comparison', fontsize=14)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.axhline(0, color='grey', linestyle='--', linewidth=0.8)
        ax2.legend(loc='lower left', fontsize='small')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

    def plot_efficient_frontier(
        self, 
        random_port_results: pd.DataFrame, 
        max_sharpe_port: Dict[str, float], 
        min_vol_port: Dict[str, float]
    ):
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(12, 8))
        
        plt.scatter(
            random_port_results.volatility, 
            random_port_results['return'], 
            c=random_port_results.sharpe, 
            cmap='viridis', marker='o', s=10, alpha=0.5
        )
        plt.colorbar(label='Sharpe Ratio')
        
        plt.scatter(max_sharpe_port['volatility'], max_sharpe_port['return'], 
                    marker='*', color='r', s=250, label='Maximum Sharpe Ratio')
        
        plt.scatter(min_vol_port['volatility'], min_vol_port['return'], 
                    marker='*', color='b', s=250, label='Minimum Volatility')
        
        plt.title('Efficient Frontier', fontsize=16)
        plt.xlabel('Annualized Volatility')
        plt.ylabel('Annualized Return')
        plt.legend(labelspacing=0.8)
        plt.grid(True)
        plt.show()