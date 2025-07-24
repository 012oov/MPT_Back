# Utils

def format_model_name(sheet_name: str) -> str:
    """Excel 시트 이름을 그래프에 표시할 이름으로 변환합니다."""
    name = sheet_name.replace('_weights', '')
    if 'target_return' in name:
        rate = name.split('_')[-1]
        return f"Target Return ({rate}%)"
    
    name_map = {
        'max_calmar': 'Max Calmar Ratio',
        'max_sharpe': 'Max Sharpe Ratio',
        'risk_parity': 'Risk Parity',
        'min_variance': 'Minimum Variance',
    }
    return name_map.get(name, name.replace('_', ' ').title())


from pathlib import Path

def ensure_dir_exists(file_path: Path):
    """파일 경로를 받아, 해당 파일이 위치할 디렉터리가 존재하지 않으면 생성합니다."""
    directory = file_path.parent
    directory.mkdir(parents=True, exist_ok=True)
