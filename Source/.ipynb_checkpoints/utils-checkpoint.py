# Utils

def format_model_name(sheet_name: str) -> str:
    """Excel 시트 이름을 그래프에 표시할 이름으로 변환합니다."""
    name = sheet_name.replace('_weights', '')
    if 'target_return' in name:
        # target_return_18 -> Target Return (18%)
        rate = name.split('_')[-1]
        return f"Target Return ({rate}%)"
    
    # max_calmar -> Max Calmar Ratio
    name_map = {
        'max_calmar': 'Max Calmar Ratio',
        'risk_parity': 'Risk Parity',
        'min_variance': 'Minimum Variance',
        'daily_30_cap': 'Daily (30% Cap)',
        'max_sharpe': 'Max Sharpe Ratio',
    }
    return name_map.get(name, name.replace('_', ' ').title())
