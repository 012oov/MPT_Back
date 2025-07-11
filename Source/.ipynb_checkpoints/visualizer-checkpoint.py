# 시각화 모듈

import matplotlib.pyplot as plt

def format_model_name(sheet_name):
    """Excel 시트 이름을 그래프에 표시할 이름으로 변환합니다."""
    name = sheet_name.replace('_weights', '')
    if 'target_return' in name: return f"Target Return ({name.split('_')[-1]}%)"
    name_map = {
        'max_calmar': 'Max Calmar Ratio',
        'risk_parity': 'Risk Parity',
        'min_variance': 'Minimum Variance',
        'daily_30_cap': 'Daily (30% Cap)',
        'daily_max_sharpe': 'Daily (Max Sharpe)',
    }
    return name_map.get(name, name.replace('_', ' ').title())

def plot_performance_comparison(portfolio_results):
    """여러 포트폴리오의 성과와 MDD를 비교하는 그래프를 그립니다."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    colors = plt.colormaps['tab10'].colors
    
    for i, (name, data) in enumerate(portfolio_results.items()):
        color = colors[i % len(colors)]
        # 포트폴리오 가치
        ax1.plot(data['value'].index, data['value'], label=f"{name} (CAGR: {data['cagr']*100:.2f}%)", color=color, linewidth=2 if 'Benchmark' in name else 1.5)
        # MDD
        ax2.plot(data['drawdown'].index, data['drawdown'] * 100, label=f"{name} (MDD: {data['mdd']*100:.2f}%)", color=color, alpha=0.8, linewidth=2 if 'Benchmark' in name else 1.5)

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
