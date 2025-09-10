"""
Configuration settings for the trading system
"""

import os
from datetime import time

# API Configuration
UPSTOX_API_KEY = os.environ.get("UPSTOX_API_KEY", "")
UPSTOX_ACCESS_TOKEN = os.environ.get("UPSTOX_ACCESS_TOKEN", "")

# Trading Parameters
TRADING_CONFIG = {
    'max_signals_per_day': 5,
    'max_concurrent_positions': 2,
    'max_risk_per_trade': 0.02,  # 2% of capital
    'max_daily_loss': 0.06,      # 6% of capital
    'initial_capital': 100000,    # Starting capital
    'min_signal_strength': 60,    # Minimum strength for signal execution
}

# Technical Indicators
INDICATOR_CONFIG = {
    'rsi_period': 14,
    'rsi_smoothing': True,
    'vwap_window': 300,
    'supertrend_atr_period': 10,
    'supertrend_multiplier': 3.0,
    'volatility_window': 20,
}

# Options Parameters
OPTIONS_CONFIG = {
    'risk_free_rate': 0.065,      # 6.5% annual
    'default_iv': 0.15,           # 15% implied volatility
    'delta_threshold': 0.4,       # Minimum delta for entry
    'strike_gap': 50,             # Strike price intervals
    'lot_size': 75,               # NIFTY lot size
    'volatility_stop_multiplier': 2.0,
    'time_based_exit_hours': 2,   # Exit after 2 hours of no movement
}

# Market Hours (IST)
MARKET_HOURS = {
    'pre_open_start': time(9, 0),
    'pre_open_end': time(9, 15),
    'normal_open': time(9, 15),
    'normal_close': time(15, 30),
    'post_close_end': time(16, 0),
}

# Signal Conditions
SIGNAL_CONFIG = {
    'ce_signal': {
        'vwap_buffer': 1.001,     # Price > VWAP * 1.001
        'rsi_min': 55,
        'rsi_max': 70,
        'max_volatility': 0.25,
    },
    'pe_signal': {
        'vwap_buffer': 0.999,     # Price < VWAP * 0.999
        'rsi_min': 30,
        'rsi_max': 45,
        'max_volatility': 0.25,
    }
}

# Risk Management
RISK_CONFIG = {
    'position_sizing_method': 'fixed_risk',  # or 'fixed_units'
    'stop_loss_methods': ['volatility', 'time_decay', 'percentage'],
    'min_stop_loss': 0.5,         # 50% minimum stop
    'target_multipliers': {
        'high_probability': 2.0,   # Delta > 0.6
        'medium_probability': 1.5, # Delta > 0.4
        'low_probability': 1.2,    # Delta < 0.4
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_format': '%(asctime)s - %(levelname)s - %(message)s',
    'file_rotation': 'daily',
    'max_log_files': 30,
}

# Report Configuration
REPORT_CONFIG = {
    'generate_html': True,
    'generate_pdf': False,
    'generate_charts': True,
    'email_reports': False,
    'chart_style': 'seaborn',
}

# Notification Configuration (optional)
NOTIFICATION_CONFIG = {
    'enable_notifications': False,
    'notification_channels': ['email', 'discord'],
    'notify_on_signal': True,
    'notify_on_trade_close': True,
    'notify_on_error': True,
}

# Holiday Calendar (add Indian market holidays)
MARKET_HOLIDAYS = [
    '2025-01-26',  # Republic Day
    '2025-03-10',  # Holi
    '2025-04-14',  # Ram Navami
    '2025-04-18',  # Good Friday
    '2025-05-01',  # Maharashtra Day
    '2025-08-15',  # Independence Day
    '2025-10-02',  # Gandhi Jayanti
    '2025-10-24',  # Dussehra
    '2025-11-12',  # Diwali
    '2025-11-13',  # Diwali (Balipratipada)
    # Add more holidays as needed
]

# Environment Detection
IS_GITHUB_ACTIONS = os.environ.get('GITHUB_ACTIONS', False)
IS_DEVELOPMENT = os.environ.get('ENVIRONMENT', 'production') == 'development'

# Override for development
if IS_DEVELOPMENT:
    TRADING_CONFIG['max_signals_per_day'] = 10
    TRADING_CONFIG['initial_capital'] = 10000  # Smaller capital for testing

def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check API credentials
    if not UPSTOX_API_KEY:
        errors.append("UPSTOX_API_KEY not set")
    if not UPSTOX_ACCESS_TOKEN:
        errors.append("UPSTOX_ACCESS_TOKEN not set")
    
    # Validate numeric ranges
    if not 0 < TRADING_CONFIG['max_risk_per_trade'] <= 0.1:
        errors.append("max_risk_per_trade should be between 0 and 10%")
    
    if not 0 < OPTIONS_CONFIG['delta_threshold'] <= 1:
        errors.append("delta_threshold should be between 0 and 1")
    
    return errors

# Validate on import
config_errors = validate_config()
if config_errors and not IS_DEVELOPMENT:
    raise ValueError(f"Configuration errors: {', '.join(config_errors)}")
