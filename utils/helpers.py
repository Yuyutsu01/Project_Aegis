from typing import Any
from datetime import datetime

def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def calculate_percentage_change(old_val: float, new_val: float) -> float:
    """Calculate percentage change"""
    if old_val == 0:
        return 0.0
    return (new_val - old_val) / old_val * 100

def format_timestamp(timestamp: datetime = None) -> str:
    """Format timestamp for display"""
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")

def normalize_value(value: float, min_val: float = 0, max_val: float = 1) -> float:
    """Normalize value to 0-1 range"""
    return max(min_val, min(max_val, value))

def calculate_kelly_fraction(win_prob: float, win_loss_ratio: float = 2.0) -> float:
    """Calculate Kelly Criterion position size"""
    if win_prob <= 0 or win_prob >= 1:
        return 0.0
    kelly = win_prob - (1 - win_prob) / win_loss_ratio
    return max(0.0, min(0.25, kelly))  # Conservative cap at 25%