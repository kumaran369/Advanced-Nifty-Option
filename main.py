#!/usr/bin/env python3 -u
"""
FIXED SCALPING NIFTY Option Single Lot Trading System
10% Profit Target, 25% Stop Loss
All logic errors corrected
"""

import time
import numpy as np
import pandas as pd
from collections import deque
import upstox_client
from upstox_client.rest import ApiException
import sys
from datetime import datetime, timedelta
import logging
import os
from math import log, sqrt, exp
from scipy.stats import norm
from typing import Dict, Optional, Tuple
import requests
import json

# Force unbuffered output for GitHub Actions
if os.environ.get('GITHUB_ACTIONS', 'false') == 'true':
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

# -------- SCALPING SETTINGS ----------
# Detect if running in GitHub Actions
IS_GITHUB_ACTIONS = os.environ.get('GITHUB_ACTIONS', 'false') == 'true'

# Discord webhook URL
DISCORD_WEBHOOK_URL = os.environ.get('DISCORD_WEBHOOK_URL', '')

print(f"[STARTUP] AGGRESSIVE SCALPING - 10% Target, Continuous Trading - Started at {datetime.now()}", flush=True)

# Read token from file
def get_token():
    token_file = "token.txt"
    if os.path.exists(token_file):
        with open(token_file, 'r') as f:
            return f.read().strip()
    return os.environ.get("UPSTOX_ACCESS_TOKEN", "")

API_KEY = "a17874e5-9c5b-45d1-aa21-04ddd1f34c67"
ACCESS_TOKEN = get_token()

# SCALPING Trading parameters
VWAP_WINDOW = 300
RSI_PERIOD = 14
RSI_WARMUP_PERIODS = 20  # Very fast warmup for quick signals
SUPER_ATR_PERIOD = 10
SUPER_MULTIPLIER = 3.0
MAX_SIGNALS_PER_DAY = 10  # Allow many trades for continuous scalping
SIGNAL_COOLDOWN_SECONDS = 300  # 5 minutes between signals
MIN_TICKS_BEFORE_SIGNAL = 50  # Minimal data for fast signals
MIN_SIGNAL_GAP_SECONDS = 60  # Minimum 1 minute between signals even after target

# Fixed position size
POSITION_SIZE = 75  # Single lot only
MAX_POSITIONS = 1   # Only one position at a time

# Options parameters
RISK_FREE_RATE = 0.065
DEFAULT_IV = 0.15
DELTA_THRESHOLD = 0.25  # Lower threshold for better liquidity
MIN_PREMIUM = 5.0  # Minimum acceptable premium
MAX_SPREAD_PERCENT = 0.15  # Maximum 15% bid-ask spread

# Risk Management - SCALPING
STOP_LOSS_PERCENTAGE = 0.25   # 25% stop loss
TARGET_PERCENTAGE = 0.10      # 10% target (scalping)
TIME_BASED_EXIT_HOURS = 1.0   # Exit after 1 hour max
MIN_STOP_LOSS = 5.0  # Minimum stop loss in rupees

# Signal Thresholds - AGGRESSIVE SCALPING
BASE_BULL_VWAP_THRESHOLD = 1.0010  # 0.10% above VWAP
BASE_BEAR_VWAP_THRESHOLD = 0.9990  # 0.10% below VWAP
MIN_SIGNAL_STRENGTH = 40           # Lower threshold for frequent signals

# Market timing helpers
def get_market_hours():
    """Get today's market hours dynamically"""
    now = datetime.now()
    return (
        now.replace(hour=9, minute=15, second=0, microsecond=0),
        now.replace(hour=15, minute=30, second=0, microsecond=0)
    )

def is_market_open():
    """Check if market is currently open"""
    now = datetime.now()
    market_open, market_close = get_market_hours()
    return market_open <= now <= market_close

# Logging configuration
class GitHubActionsLogger:
    def __init__(self):
        self.last_tick_time = 0
        self.tick_interval = 10
        
    def should_log_tick(self):
        current_time = time.time()
        if current_time - self.last_tick_time >= self.tick_interval:
            self.last_tick_time = current_time
            return True
        return False

gh_logger = GitHubActionsLogger()

class OptionsDataManager:
    """Real-time options data fetching using Upstox REST API"""
    
    def __init__(self, access_token):
        self.access_token = access_token
        self.option_chain_cache = {}
        self.chain_cache_expiry = 15  # Reduced to 15 seconds for scalping
        
        # NIFTY symbol constants
        self.nifty_symbol = "NSE_INDEX|Nifty 50"
        self.nifty_instrument_key = "NSE_INDEX|Nifty 50"
        
        # Use direct REST API calls
        self.base_url = "https://api.upstox.com/v2"
        self.headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {access_token}'
        }
        
    def get_option_chain_data(self, spot_price, expiry_date):
        """Get Put/Call option chain data with enhanced validation"""
        try:
            import requests
            
            expiry_str = expiry_date.strftime('%Y-%m-%d') if hasattr(expiry_date, 'strftime') else str(expiry_date)
            
            cache_key = f"option_chain_{expiry_str}"
            current_time = time.time()
            
            # Check cache
            if (cache_key in self.option_chain_cache and 
                current_time - self.option_chain_cache[cache_key]['timestamp'] < self.chain_cache_expiry):
                cached_data = self.option_chain_cache[cache_key]['data']
                print(f"[OPTIONS API] Using cached option chain data", flush=True)
                return cached_data
            
            url = f"{self.base_url}/option/chain"
            params = {
                'instrument_key': self.nifty_instrument_key,
                'expiry_date': expiry_str
            }
            
            print(f"[OPTIONS API] Fetching fresh option chain for {expiry_str}", flush=True)
            response = requests.get(url, headers=self.headers, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    option_data = data.get('data', [])
                    
                    # Validate data quality
                    if not self._validate_option_data(option_data):
                        print(f"[OPTIONS API] Invalid option chain data received", flush=True)
                        return []
                    
                    # Cache the result
                    self.option_chain_cache[cache_key] = {
                        'data': option_data,
                        'timestamp': current_time
                    }
                    
                    print(f"[OPTIONS API] Fetched {len(option_data)} strikes", flush=True)
                    return option_data
                else:
                    print(f"[OPTIONS API] API error: {data}", flush=True)
                    return []
            else:
                print(f"[OPTIONS API] HTTP {response.status_code}: {response.text[:200]}", flush=True)
                return []
                
        except Exception as e:
            print(f"[OPTIONS API] Exception: {e}", flush=True)
            return []
    
    def _validate_option_data(self, option_data):
        """Validate option chain data quality"""
        if not option_data or len(option_data) == 0:
            return False
        
        # Check if we have valid strikes with both CE and PE data
        valid_strikes = 0
        for option in option_data:
            if (option.get('strike_price', 0) > 0 and 
                'call_options' in option and 
                'put_options' in option):
                
                call_ltp = option['call_options'].get('market_data', {}).get('ltp', 0)
                put_ltp = option['put_options'].get('market_data', {}).get('ltp', 0)
                
                if call_ltp > 0 and put_ltp > 0:
                    valid_strikes += 1
        
        # Need at least 10 valid strikes for reliable data
        is_valid = valid_strikes >= 10
        print(f"[OPTIONS API] Validation: {valid_strikes} valid strikes, quality={'GOOD' if is_valid else 'POOR'}", flush=True)
        return is_valid
    
    def get_option_premium(self, strike, option_type, expiry_date, spot_price):
        """Get option premium with enhanced validation"""
        chain_data = self.get_option_chain_data(spot_price, expiry_date)
        
        if not chain_data:
            print(f"[OPTIONS API] No option chain data available", flush=True)
            return None
        
        try:
            for option_data in chain_data:
                strike_price = float(option_data.get('strike_price', 0))
                
                if abs(strike_price - strike) < 0.1:  # Match strike
                    
                    if option_type == 'CE' and 'call_options' in option_data:
                        call_option = option_data['call_options']
                        market_data = call_option.get('market_data', {})
                        option_greeks = call_option.get('option_greeks', {})
                        
                        ltp = float(market_data.get('ltp', 0))
                        bid = float(market_data.get('bid_price', 0))
                        ask = float(market_data.get('ask_price', 0))
                        volume = int(market_data.get('volume', 0))
                        oi = int(market_data.get('oi', 0))
                        
                        # Validate data quality
                        if not self._validate_premium_data(ltp, bid, ask, volume, oi):
                            return None
                        
                        delta = float(option_greeks.get('delta', 0))
                        theta = float(option_greeks.get('theta', 0))
                        iv = float(option_greeks.get('iv', 0))
                        
                        return {
                            'premium': ltp,
                            'instrument_key': call_option.get('instrument_key', ''),
                            'lot_size': 75,
                            'bid_price': bid,
                            'ask_price': ask,
                            'volume': volume,
                            'oi': oi,
                            'delta': delta,
                            'theta': theta,
                            'iv': iv,
                            'spread': ask - bid if ask > bid else 0
                        }
                    
                    elif option_type == 'PE' and 'put_options' in option_data:
                        put_option = option_data['put_options']
                        market_data = put_option.get('market_data', {})
                        option_greeks = put_option.get('option_greeks', {})
                        
                        ltp = float(market_data.get('ltp', 0))
                        bid = float(market_data.get('bid_price', 0))
                        ask = float(market_data.get('ask_price', 0))
                        volume = int(market_data.get('volume', 0))
                        oi = int(market_data.get('oi', 0))
                        
                        # Validate data quality
                        if not self._validate_premium_data(ltp, bid, ask, volume, oi):
                            return None
                        
                        delta = float(option_greeks.get('delta', 0))
                        theta = float(option_greeks.get('theta', 0))
                        iv = float(option_greeks.get('iv', 0))
                        
                        return {
                            'premium': ltp,
                            'instrument_key': put_option.get('instrument_key', ''),
                            'lot_size': 75,
                            'bid_price': bid,
                            'ask_price': ask,
                            'volume': volume,
                            'oi': oi,
                            'delta': delta,
                            'theta': theta,
                            'iv': iv,
                            'spread': ask - bid if ask > bid else 0
                        }
            
            print(f"[OPTIONS API] Strike {strike} not found in chain", flush=True)
            return None
            
        except Exception as e:
            print(f"[OPTIONS API] Exception in premium lookup: {e}", flush=True)
            return None
    
    def _validate_premium_data(self, ltp, bid, ask, volume, oi):
        """Validate premium data quality"""
        # Basic validation
        if ltp <= 0:
            print(f"[OPTIONS API] Invalid LTP: {ltp}", flush=True)
            return False
        
        # Check for minimum premium
        if ltp < MIN_PREMIUM:
            print(f"[OPTIONS API] Premium too low: {ltp} < {MIN_PREMIUM}", flush=True)
            return False
        
        # Check for reasonable bid-ask spread
        if ask > bid > 0:
            spread_pct = (ask - bid) / ltp
            if spread_pct > MAX_SPREAD_PERCENT:
                print(f"[OPTIONS API] Wide spread: {spread_pct*100:.1f}% > {MAX_SPREAD_PERCENT*100:.1f}%", flush=True)
                return False
        
        # Check for minimum liquidity
        if volume == 0 and oi < 100:
            print(f"[OPTIONS API] Poor liquidity: Vol={volume}, OI={oi}", flush=True)
            return False
        
        return True
    
    def get_next_expiry_date(self):
        """Get next Thursday expiry for NIFTY"""
        try:
            # Get from option chain data
            chain_data = self.get_option_chain_data(25000, None)
            
            if chain_data:
                expiry_dates = set()
                for option_data in chain_data:
                    expiry_str = option_data.get('expiry', '')
                    if expiry_str:
                        try:
                            expiry_date = datetime.strptime(expiry_str, '%Y-%m-%d')
                            if expiry_date >= datetime.now():
                                expiry_dates.add(expiry_date)
                        except ValueError:
                            continue
                
                if expiry_dates:
                    next_expiry = min(expiry_dates)
                    print(f"[OPTIONS API] Next expiry: {next_expiry.strftime('%Y-%m-%d')}", flush=True)
                    return next_expiry
            
            print(f"[OPTIONS API] No expiry data from API", flush=True)
            return None
            
        except Exception as e:
            print(f"[OPTIONS API] Error getting expiry: {e}", flush=True)
            return None

class ScalpingSignalGenerator:
    def __init__(self):
        self.vwap_cumulative = {'sum_pv': 0, 'sum_v': 0, 'day_start': None}
        self.rsi_state = {'gains': [], 'losses': [], 'avg_gain': None, 'avg_loss': None}
        self.volatility_estimator = VolatilityEstimator()
        self.signal_debug_counter = 0
        self.rsi_data_points = 0
        self.last_signal_time = 0  # Start with 0
        self.total_ticks_processed = 0
        self.prev_price = None
        
    def update_vwap(self, price, volume, timestamp):
        """VWAP calculation with proper day validation"""
        current_date = datetime.fromtimestamp(timestamp).date()
        current_weekday = current_date.weekday()
        
        # Only reset on trading days (Mon-Fri) and during market hours
        if (self.vwap_cumulative['day_start'] != current_date and 
            current_weekday < 5 and is_market_open()):
            
            self.vwap_cumulative = {
                'sum_pv': price * volume,
                'sum_v': volume,
                'day_start': current_date
            }
        else:
            self.vwap_cumulative['sum_pv'] += price * volume
            self.vwap_cumulative['sum_v'] += volume
        
        if self.vwap_cumulative['sum_v'] > 0:
            return self.vwap_cumulative['sum_pv'] / self.vwap_cumulative['sum_v']
        return price
    
    def update_rsi(self, price, period=14):
        """Enhanced RSI with proper initialization"""
        if self.prev_price is None:
            self.prev_price = price
            return None
        
        change = price - self.prev_price
        gain = max(change, 0)
        loss = max(-change, 0)
        
        self.rsi_data_points += 1
        
        if self.rsi_state['avg_gain'] is None:
            self.rsi_state['gains'].append(gain)
            self.rsi_state['losses'].append(loss)
            
            if len(self.rsi_state['gains']) >= period:
                self.rsi_state['avg_gain'] = sum(self.rsi_state['gains']) / period
                self.rsi_state['avg_loss'] = sum(self.rsi_state['losses']) / period
                self.rsi_state['gains'] = []
                self.rsi_state['losses'] = []
        else:
            # Wilder's smoothing
            self.rsi_state['avg_gain'] = (self.rsi_state['avg_gain'] * (period - 1) + gain) / period
            self.rsi_state['avg_loss'] = (self.rsi_state['avg_loss'] * (period - 1) + loss) / period
        
        self.prev_price = price
        
        if self.rsi_state['avg_gain'] is not None and self.rsi_state['avg_loss'] > 0:
            rs = self.rsi_state['avg_gain'] / self.rsi_state['avg_loss']
            return 100 - (100 / (1 + rs))
        elif self.rsi_state['avg_gain'] is not None and self.rsi_state['avg_loss'] == 0:
            return 100
        
        return None
    
    def get_dynamic_thresholds(self, volatility):
        """Aggressive scalping: Very relaxed thresholds for frequent signals"""
        if volatility < 0.10:  # Low volatility
            vol_multiplier = 0.5  # Very relaxed
        elif volatility < 0.15:  # Normal volatility
            vol_multiplier = 0.7  # Relaxed
        elif volatility < 0.20:  # High volatility
            vol_multiplier = 0.9  # Slightly relaxed
        else:  # Very high volatility
            vol_multiplier = 1.1  # Still tradeable
        
        bull_threshold = 1 + ((BASE_BULL_VWAP_THRESHOLD - 1) * vol_multiplier)
        bear_threshold = 1 - ((1 - BASE_BEAR_VWAP_THRESHOLD) * vol_multiplier)
        
        return bull_threshold, bear_threshold
    
    def generate_signal(self, market_data: Dict) -> Optional[Dict]:
        """Scalping signal generation with relaxed conditions"""
        price = market_data['price']
        vwap = market_data['vwap']
        rsi = market_data['rsi']
        supertrend = market_data['supertrend']
        volatility = market_data['volatility']
        
        self.total_ticks_processed += 1
        self.signal_debug_counter += 1
        
        # Cooldown check
        current_time = time.time()
        time_since_last_signal = current_time - self.last_signal_time
        if time_since_last_signal < SIGNAL_COOLDOWN_SECONDS:
            return None
        
        # Warmup checks (reduced for scalping)
        if self.total_ticks_processed < MIN_TICKS_BEFORE_SIGNAL:
            return None
        
        if self.rsi_data_points < RSI_WARMUP_PERIODS:
            return None
        
        # Data availability check
        if not vwap or not rsi:
            return None
        
        # Market hours check
        if not is_market_open():
            return None
        
        # Get dynamic thresholds
        bull_threshold, bear_threshold = self.get_dynamic_thresholds(volatility)
        
        # Debug output every 20 checks
        if self.signal_debug_counter % 20 == 0:
            print(f"[DEBUG] {datetime.now().strftime('%H:%M:%S')} | Price: {price:.2f} | VWAP: {vwap:.2f} | RSI: {rsi:.1f} | Vol: {volatility*100:.1f}% | Bull>{bull_threshold:.4f} | Bear<{bear_threshold:.4f}", flush=True)
        
        signal = None
        price_vwap_ratio = price / vwap
        
        # BULLISH CONDITIONS - Very relaxed for frequent scalping
        bull_vwap_condition = price_vwap_ratio > bull_threshold
        bull_supertrend_condition = supertrend.get('is_uptrend', False) if supertrend['value'] else False
        bull_rsi_condition = 40 < rsi < 80  # Very wide range
        bull_vol_condition = volatility < 0.35  # Accept higher volatility
        
        if all([bull_vwap_condition, bull_supertrend_condition, bull_rsi_condition, bull_vol_condition]):
            strength = self._calculate_signal_strength(price, vwap, rsi, True, volatility)
            if strength >= MIN_SIGNAL_STRENGTH:
                signal = {
                    'type': 'CE',
                    'strength': strength
                }
                print(f"[SIGNAL] {datetime.now().strftime('%H:%M:%S')} | BULLISH | Strength: {strength:.1f} | Price: {price:.2f} > VWAP: {vwap:.2f} | RSI: {rsi:.1f}", flush=True)
                self.last_signal_time = current_time
        
        # BEARISH CONDITIONS - Very relaxed for frequent scalping
        bear_vwap_condition = price_vwap_ratio < bear_threshold
        bear_supertrend_condition = not supertrend.get('is_uptrend', True) if supertrend['value'] else False
        bear_rsi_condition = 20 < rsi < 60  # Very wide range
        bear_vol_condition = volatility < 0.35  # Accept higher volatility
        
        if not signal and all([bear_vwap_condition, bear_supertrend_condition, bear_rsi_condition, bear_vol_condition]):
            strength = self._calculate_signal_strength(price, vwap, rsi, False, volatility)
            if strength >= MIN_SIGNAL_STRENGTH:
                signal = {
                    'type': 'PE',
                    'strength': strength
                }
                print(f"[SIGNAL] {datetime.now().strftime('%H:%M:%S')} | BEARISH | Strength: {strength:.1f} | Price: {price:.2f} < VWAP: {vwap:.2f} | RSI: {rsi:.1f}", flush=True)
                self.last_signal_time = current_time
        
        return signal
    
    def _calculate_signal_strength(self, price, vwap, rsi, is_bullish, volatility):
        """Signal strength calculation optimized for frequent scalping"""
        vwap_distance = abs(price - vwap) / vwap * 100
        
        # Base strength from VWAP distance (very sensitive)
        vwap_component = min(50, vwap_distance * 5000)
        
        # RSI strength component (more lenient)
        if is_bullish:
            rsi_strength = max(0, min(30, (rsi - 40) * 3))
        else:
            rsi_strength = max(0, min(30, (60 - rsi) * 3))
        
        # Volatility penalty (minimal for scalping)
        vol_penalty = max(0, (volatility - 0.25) * 20) if volatility > 0.25 else 0
        
        # Trend confirmation bonus
        trend_bonus = 10
        
        final_strength = vwap_component + rsi_strength + trend_bonus - vol_penalty
        
        return max(0, min(100, final_strength))
    
    def reset_for_next_signal(self):
        """Properly reset signal timer after target hit"""
        # Set last signal time to allow minimum gap
        self.last_signal_time = time.time() - SIGNAL_COOLDOWN_SECONDS + MIN_SIGNAL_GAP_SECONDS

class VolatilityEstimator:
    def __init__(self, window=20):
        self.returns = deque(maxlen=window)
        self.last_price = None
        
    def update(self, price):
        if self.last_price is not None and self.last_price > 0:
            ret = log(price / self.last_price)
            self.returns.append(ret)
        self.last_price = price
        
    def get_volatility(self):
        if len(self.returns) < 5:
            return DEFAULT_IV
        
        std = np.std(self.returns)
        # Corrected annualization factor
        annual_vol = std * sqrt(252)  # Daily returns annualized
        
        return max(0.08, min(0.40, annual_vol))

class ScalpingRiskManager:
    def __init__(self):
        self.daily_pnl = 0
        self.max_daily_loss = 10000  # Increased for continuous trading
        self.trades_today = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.target_reached_today = False
        self.pending_pnl = 0  # Track P&L being closed
    
    def calculate_stops_targets(self, premium):
        """Scalping percentage-based stops and targets"""
        stop_loss = premium * (1 - STOP_LOSS_PERCENTAGE)
        target = premium * (1 + TARGET_PERCENTAGE)
        
        return {
            'stop_loss': max(MIN_STOP_LOSS, round(stop_loss, 1)),
            'target': round(target, 1),
            'risk_reward': round(TARGET_PERCENTAGE / STOP_LOSS_PERCENTAGE, 2)
        }
    
    def can_take_position(self, potential_loss=None):
        """Check if we can take a new position considering daily limits"""
        # Check if we would exceed daily loss limit
        if potential_loss:
            projected_loss = self.daily_pnl - potential_loss
            return projected_loss > -self.max_daily_loss
        
        # Basic check
        return self.daily_pnl > -self.max_daily_loss
    
    def reset_for_next_trade(self):
        """Reset cooldown for immediate next trade after target hit"""
        self.target_reached_today = True
    
    def get_win_rate(self):
        """Calculate win rate safely"""
        total_trades = self.winning_trades + self.losing_trades
        if total_trades == 0:
            return 0
        return (self.winning_trades / total_trades) * 100

class DiscordNotifier:
    def __init__(self, webhook_url):
        self.webhook_url = webhook_url
        self.enabled = bool(webhook_url)
        
    def send_trade_signal(self, position_data):
        if not self.enabled:
            return
            
        try:
            color = 0x00ff00 if position_data['type'] == 'CE' else 0xff0000
            
            embed = {
                "title": f"⚡ SCALP {position_data['type']} SIGNAL",
                "color": color,
                "timestamp": datetime.now().isoformat(),
                "fields": [
                    {
                        "name": "Strike",
                        "value": f"{position_data['strike']}",
                        "inline": True
                    },
                    {
                        "name": "Entry Premium",
                        "value": f"₹{position_data['premium']:.2f}",
                        "inline": True
                    },
                    {
                        "name": "Spot Price",
                        "value": f"₹{position_data['spot_entry']:.2f}",
                        "inline": True
                    },
                    {
                        "name": "Target (10%)",
                        "value": f"₹{position_data['target']:.2f}",
                        "inline": True
                    },
                    {
                        "name": "Stop Loss (25%)",
                        "value": f"₹{position_data['stop_loss']:.2f}",
                        "inline": True
                    },
                    {
                        "name": "Risk/Reward",
                        "value": f"1:{position_data['risk_reward']:.2f}",
                        "inline": True
                    }
                ]
            }
            
            content = {
                "username": "NIFTY Scalping Bot",
                "content": f"**Scalp Trade Alert!** {position_data['type']} {position_data['strike']} @ ₹{position_data['premium']:.2f}",
                "embeds": [embed]
            }
            
            requests.post(self.webhook_url, json=content, timeout=5)
            
        except Exception as e:
            print(f"Discord notification error: {e}", flush=True)

class ScalpingTrader:
    def __init__(self):
        self.signal_generator = ScalpingSignalGenerator()
        self.risk_manager = ScalpingRiskManager()
        self.discord_notifier = DiscordNotifier(DISCORD_WEBHOOK_URL)
        
        # Initialize options data manager
        if ACCESS_TOKEN:
            self.options_data_manager = OptionsDataManager(ACCESS_TOKEN)
            print("[INIT] Options data manager initialized", flush=True)
        else:
            print("[INIT] No access token available", flush=True)
            self.options_data_manager = None
        
        self.active_position = None  # Only one position
        self.tick_buffer = deque(maxlen=2000)  # Reduced for better memory usage
        self.tick_count = 0
        self.last_processed_time = 0
        self.last_pnl_log_time = 0  # Track last P&L log time
        self.pnl_log_frequency = 1.0  # Log every 1 second by default
        self.position_monitor_fail_count = 0  # Track monitoring failures
        
    def process_tick(self, tick_data):
        try:
            if not self._validate_tick(tick_data):
                return
            
            timestamp = time.time()
            price = float(tick_data.get("last_price", tick_data.get("ltp", 0)))
            volume = max(1, float(tick_data.get("volume", 1)))  # Ensure positive volume
            
            # Throttle processing
            if timestamp - self.last_processed_time < 0.2:
                return
            
            self.last_processed_time = timestamp
            self.tick_buffer.append((timestamp, price, volume))
            self.tick_count += 1
            
            market_data = self._update_market_data(price, volume, timestamp)
            
            # Only show status when NO position is active (clean display)
            if not self.active_position and market_data['vwap'] and market_data['rsi'] and self.tick_count % 10 == 0:
                trend = "UP" if market_data['supertrend'].get('is_uptrend', False) else "DN"
                if self.risk_manager.target_reached_today:
                    position_info = "🔍 SCANNING"
                else:
                    position_info = "WAITING"
                    
                win_rate = self.risk_manager.get_win_rate()
                
                status_line = (f"[{datetime.now().strftime('%H:%M:%S')}] "
                              f"NIFTY {price:,.0f} | VWAP {market_data['vwap']:,.0f} | "
                              f"RSI {market_data['rsi']:.0f} | {trend} | "
                              f"Trades {self.risk_manager.trades_today} | "
                              f"Win% {win_rate:.0f} | "
                              f"Daily P&L: Rs{self.risk_manager.daily_pnl:+.0f} | "
                              f"Status: {position_info}")
                
                print(f"[STATUS] {status_line}", flush=True)
            
            # Manage existing position
            if self.active_position:
                self._manage_position(market_data)
            
            # Check for new signals
            elif self.risk_manager.can_take_position():
                signal = self.signal_generator.generate_signal(market_data)
                if signal:
                    self._execute_signal(signal, market_data)
                    
        except Exception as e:
            print(f"\n[ERROR] Tick processing: {e}", flush=True)
            
    def _validate_tick(self, tick_data):
        """Enhanced tick validation"""
        try:
            price = float(tick_data.get("last_price", tick_data.get("ltp", 0)))
            volume = float(tick_data.get("volume", 1))
            
            # Basic price validation
            if not (15000 < price < 35000):  # Reasonable NIFTY range
                return False
            
            # Volume validation
            if volume < 0:
                return False
                
            return True
        except (ValueError, TypeError):
            return False
    
    def _update_market_data(self, price, volume, timestamp):
        """Update all market indicators"""
        self.signal_generator.volatility_estimator.update(price)
        
        vwap = self.signal_generator.update_vwap(price, volume, timestamp)
        rsi = self.signal_generator.update_rsi(price)
        
        # Build candles for SuperTrend
        candles = self._build_candles()
        supertrend = self._calculate_supertrend(candles)
        
        return {
            'price': price,
            'volume': volume,
            'vwap': vwap,
            'rsi': rsi,
            'supertrend': supertrend,
            'volatility': self.signal_generator.volatility_estimator.get_volatility(),
            'timestamp': timestamp
        }
    
    def _execute_signal(self, signal, market_data):
        """Execute signal with proper validation"""
        try:
            if not self.options_data_manager:
                print("[TRADE] No options data manager - skipping", flush=True)
                return
            
            spot_price = market_data['price']
            option_type = signal['type']
            strike = self._get_atm_strike(spot_price)
            
            # Get expiry
            expiry = self.options_data_manager.get_next_expiry_date()
            if not expiry:
                print("[TRADE] No expiry available - skipping", flush=True)
                return
            
            # Get option premium
            option_data = self.options_data_manager.get_option_premium(
                strike, option_type, expiry, spot_price
            )
            
            if not option_data:
                print(f"[TRADE] No premium data for {option_type} {strike} - skipping", flush=True)
                return
            
            premium = option_data['premium']
            
            # Validate premium
            if premium <= 0:
                print(f"[TRADE] Invalid premium {premium} - skipping", flush=True)
                return
            
            if premium < MIN_PREMIUM:
                print(f"[TRADE] Premium {premium} below minimum {MIN_PREMIUM} - skipping", flush=True)
                return
            
            delta = abs(option_data.get('delta', 0))
            
            # Validate delta threshold
            if delta < DELTA_THRESHOLD:
                print(f"[TRADE] Delta {delta:.3f} below threshold {DELTA_THRESHOLD} - skipping", flush=True)
                return
            
            # Validate spread (liquidity)
            spread = option_data.get('spread', 0)
            if spread > premium * MAX_SPREAD_PERCENT:
                print(f"[TRADE] Wide spread {spread:.2f} ({spread/premium*100:.1f}%) - skipping", flush=True)
                return
            
            # Calculate stops and targets
            stops_targets = self.risk_manager.calculate_stops_targets(premium)
            
            # Check if we can afford the potential loss
            potential_loss = (premium - stops_targets['stop_loss']) * POSITION_SIZE
            if not self.risk_manager.can_take_position(potential_loss):
                print(f"[TRADE] Would exceed daily loss limit - skipping", flush=True)
                return
            
            # Create position
            position = {
                'type': option_type,
                'strike': strike,
                'premium': premium,
                'target': stops_targets['target'],
                'stop_loss': stops_targets['stop_loss'],
                'risk_reward': stops_targets['risk_reward'],
                'expiry': expiry,
                'spot_entry': spot_price,
                'entry_time': datetime.now(),
                'signal_strength': signal['strength'],
                'instrument_key': option_data['instrument_key'],
                'delta': delta,
                'theta': option_data.get('theta', 0)
            }
            
            # Single line trade entry log
            print(f"[TRADE] {datetime.now().strftime('%H:%M:%S')} | OPEN {option_type} {strike} @ {premium:.2f} | Target: {stops_targets['target']:.2f} (+{TARGET_PERCENTAGE*100:.0f}%) | Stop: {stops_targets['stop_loss']:.2f} (-{STOP_LOSS_PERCENTAGE*100:.0f}%) | Strength: {signal['strength']:.0f}", flush=True)
            
            # Update counters
            self.active_position = position
            self.risk_manager.trades_today += 1
            self.position_monitor_fail_count = 0  # Reset fail counter
            
            # Send notification
            self.discord_notifier.send_trade_signal(position)
            
        except Exception as e:
            print(f"[ERROR] Signal execution: {e}", flush=True)
    
    def _manage_position(self, market_data):
        """Manage the active position with continuous P&L logging"""
        if not self.active_position:
            return
        
        try:
            position = self.active_position
            time_elapsed = (datetime.now() - position['entry_time']).total_seconds()
            
            # Check expiry
            if datetime.now() >= position['expiry']:
                # Try to get last premium, otherwise use 1% of entry
                last_premium = position['premium'] * 0.01
                print(f"[WARNING] Option expired - using estimated exit premium: {last_premium:.2f}", flush=True)
                self._close_position(last_premium, "EXPIRED")
                return
            
            # Get current premium
            current_premium_data = self.options_data_manager.get_option_premium(
                position['strike'], position['type'], position['expiry'], market_data['price']
            )
            
            if not current_premium_data:
                self.position_monitor_fail_count += 1
                print(f"[WARNING] Cannot fetch premium for {position['type']} {position['strike']} - monitoring interrupted (fail #{self.position_monitor_fail_count})", flush=True)
                
                # If monitoring fails too many times, close with estimated price
                if self.position_monitor_fail_count >= 5:
                    estimated_premium = position['premium']  # Use entry price as estimate
                    print(f"[WARNING] Too many monitoring failures - closing at entry price", flush=True)
                    self._close_position(estimated_premium, "MONITOR_FAIL")
                return
            
            # Reset fail counter on success
            self.position_monitor_fail_count = 0
            
            current_premium = current_premium_data['premium']
            pnl = (current_premium - position['premium']) * POSITION_SIZE
            pnl_pct = ((current_premium / position['premium']) - 1) * 100 if position['premium'] > 0 else 0
            
            # Dynamic log frequency based on P&L proximity to targets
            current_time = time.time()
            if abs(pnl_pct) >= 8:  # Near target or stop loss
                self.pnl_log_frequency = 0.2  # Log every 0.2 seconds (fast)
            elif abs(pnl_pct) >= 5:
                self.pnl_log_frequency = 0.5  # Log every 0.5 seconds
            else:
                self.pnl_log_frequency = 1.0  # Log every 1 second
            
            # CONTINUOUS P&L LOG - Print based on frequency
            if current_time - self.last_pnl_log_time >= self.pnl_log_frequency:
                status_emoji = "🟢" if pnl >= 0 else "🔴"
                if pnl_pct >= 8:
                    status_emoji = "🎯"  # Near target
                elif pnl_pct <= -20:
                    status_emoji = "⚠️"  # Near stop loss
                
                target_distance = ((position['target'] - current_premium) / current_premium * 100) if current_premium > 0 else 0
                stop_distance = ((current_premium - position['stop_loss']) / current_premium * 100) if current_premium > 0 else 0
                
                print(f"[P&L] {datetime.now().strftime('%H:%M:%S.%f')[:-4]} | {position['type']} {position['strike']} | Premium: {current_premium:.2f} | P&L: {pnl:+.0f} ({pnl_pct:+.1f}%) {status_emoji} | Target: {target_distance:.1f}% away | Stop: {stop_distance:.1f}% away | Spot: {market_data['price']:.0f}", flush=True)
                self.last_pnl_log_time = current_time
            
            # Exit conditions
            if current_premium <= position['stop_loss']:
                self._close_position(current_premium, "STOP_LOSS")
            elif current_premium >= position['target']:
                self._close_position(current_premium, "TARGET_HIT")
            elif time_elapsed > TIME_BASED_EXIT_HOURS * 3600:
                self._close_position(current_premium, "TIME_EXIT")
            elif self.risk_manager.daily_pnl + pnl <= -self.risk_manager.max_daily_loss:
                self._close_position(current_premium, "DAILY_LOSS_LIMIT")
            
        except Exception as e:
            print(f"[ERROR] Position management: {e}", flush=True)
    
    def _close_position(self, exit_price, reason):
        """Close the active position"""
        if not self.active_position:
            return
        
        position = self.active_position
        pnl = (exit_price - position['premium']) * POSITION_SIZE
        pnl_pct = ((exit_price / position['premium']) - 1) * 100 if position['premium'] > 0 else 0
        duration_mins = (datetime.now() - position['entry_time']).total_seconds() / 60
        
        # Update P&L BEFORE printing
        self.risk_manager.daily_pnl += pnl
        
        # Update win/loss counters
        if pnl > 0:
            self.risk_manager.winning_trades += 1
            self.risk_manager.consecutive_wins += 1
            self.risk_manager.consecutive_losses = 0
            status = "✅ PROFIT"
        else:
            self.risk_manager.losing_trades += 1
            self.risk_manager.consecutive_losses += 1
            self.risk_manager.consecutive_wins = 0
            status = "❌ LOSS"
        
        # Single line position close log with UPDATED daily P&L
        status_icon = "✅" if pnl > 0 else "❌"
        total_trades = self.risk_manager.winning_trades + self.risk_manager.losing_trades
        print(f"[TRADE] {datetime.now().strftime('%H:%M:%S')} | CLOSE {reason} {status_icon} | {position['type']} {position['strike']} | Entry: {position['premium']:.2f} Exit: {exit_price:.2f} | P&L: {pnl:+.0f} ({pnl_pct:+.1f}%) | Daily: {self.risk_manager.daily_pnl:+.0f} | Win Rate: {self.risk_manager.winning_trades}/{total_trades}", flush=True)
        
        self.active_position = None
        
        # IMPORTANT: Reset signal cooldown if target was hit for immediate next trade
        if reason == "TARGET_HIT" and pnl > 0:
            print(f"[SYSTEM] {datetime.now().strftime('%H:%M:%S')} | Target hit! Resetting cooldown for next trade...", flush=True)
            self.signal_generator.reset_for_next_signal()
            self.risk_manager.reset_for_next_trade()
        
        # Send Discord notification
        if self.discord_notifier.enabled:
            try:
                color = 0x00ff00 if pnl > 0 else 0xff0000
                emoji = "✅" if pnl > 0 else "❌"
                
                embed = {
                    "title": f"{emoji} SCALP CLOSED - {reason}",
                    "color": color,
                    "timestamp": datetime.now().isoformat(),
                    "fields": [
                        {"name": "Type", "value": f"{position['type']} {position['strike']}", "inline": True},
                        {"name": "P&L", "value": f"₹{pnl:+.0f} ({pnl_pct:+.1f}%)", "inline": True},
                        {"name": "Duration", "value": f"{duration_mins:.0f} mins", "inline": True},
                        {"name": "Daily Total", "value": f"₹{self.risk_manager.daily_pnl:+.0f}", "inline": True},
                        {"name": "Win Streak", "value": f"{self.risk_manager.consecutive_wins}", "inline": True},
                        {"name": "Next Trade", "value": "Scanning..." if reason == "TARGET_HIT" else "Waiting", "inline": True}
                    ]
                }
                
                content = {
                    "username": "NIFTY Scalping Bot",
                    "content": f"**Scalp Closed: {emoji} ₹{pnl:+.0f}**" + (" - Scanning for next trade!" if reason == "TARGET_HIT" else ""),
                    "embeds": [embed]
                }
                
                requests.post(self.discord_notifier.webhook_url, json=content, timeout=5)
                
            except Exception as e:
                print(f"[ERROR] Discord notification: {e}", flush=True)
    
    def _get_atm_strike(self, spot_price, gap=50):
        """Get ATM strike with proper rounding"""
        return round(spot_price / gap) * gap
    
    def _build_candles(self):
        """Build 1-minute candles from tick data"""
        if len(self.tick_buffer) < 20:
            return pd.DataFrame()
        
        try:
            df = pd.DataFrame(list(self.tick_buffer), columns=['ts', 'price', 'vol'])
            df['datetime'] = pd.to_datetime(df['ts'], unit='s')
            
            # 1-minute candles
            df['minute'] = df['datetime'].dt.floor('1min')
            
            candles = df.groupby('minute').agg({
                'price': ['first', 'max', 'min', 'last', 'count'],
                'vol': 'sum'
            }).reset_index()
            
            candles.columns = ['minute', 'open', 'high', 'low', 'close', 'count', 'volume']
            
            # Keep candles with at least 5 ticks
            candles = candles[candles['count'] >= 5].drop('count', axis=1)
            
            return candles
            
        except Exception as e:
            print(f"[ERROR] Candle building: {e}", flush=True)
            return pd.DataFrame()
    
    def _calculate_supertrend(self, df, period=10, multiplier=3.0):
        """Calculate SuperTrend indicator"""
        if df.empty or len(df) < period:
            return {'value': None, 'is_uptrend': None}
        
        try:
            df = df.copy()
            
            # True Range calculation
            df['prev_close'] = df['close'].shift(1)
            df['tr1'] = df['high'] - df['low']
            df['tr2'] = abs(df['high'] - df['prev_close'])
            df['tr3'] = abs(df['low'] - df['prev_close'])
            df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            
            # ATR calculation
            df['atr'] = df['tr'].rolling(window=period, min_periods=period//2).mean()
            
            # Basic bands
            hl2 = (df['high'] + df['low']) / 2
            df['basic_upper'] = hl2 + multiplier * df['atr']
            df['basic_lower'] = hl2 - multiplier * df['atr']
            
            # Initialize arrays
            df['upper'] = df['basic_upper'].copy()
            df['lower'] = df['basic_lower'].copy()
            df['supertrend'] = 0.0
            df['in_uptrend'] = True
            
            # Calculate SuperTrend
            for i in range(1, len(df)):
                # Upper band
                if (df['basic_upper'].iloc[i] < df['upper'].iloc[i-1] or 
                    df['close'].iloc[i-1] > df['upper'].iloc[i-1]):
                    df.iloc[i, df.columns.get_loc('upper')] = df['basic_upper'].iloc[i]
                else:
                    df.iloc[i, df.columns.get_loc('upper')] = df['upper'].iloc[i-1]
                
                # Lower band
                if (df['basic_lower'].iloc[i] > df['lower'].iloc[i-1] or 
                    df['close'].iloc[i-1] < df['lower'].iloc[i-1]):
                    df.iloc[i, df.columns.get_loc('lower')] = df['basic_lower'].iloc[i]
                else:
                    df.iloc[i, df.columns.get_loc('lower')] = df['lower'].iloc[i-1]
                
                # Trend determination
                if df['in_uptrend'].iloc[i-1]:
                    if df['close'].iloc[i] <= df['lower'].iloc[i]:
                        df.iloc[i, df.columns.get_loc('in_uptrend')] = False
                        df.iloc[i, df.columns.get_loc('supertrend')] = df['upper'].iloc[i]
                    else:
                        df.iloc[i, df.columns.get_loc('supertrend')] = df['lower'].iloc[i]
                else:
                    if df['close'].iloc[i] > df['upper'].iloc[i]:
                        df.iloc[i, df.columns.get_loc('in_uptrend')] = True
                        df.iloc[i, df.columns.get_loc('supertrend')] = df['lower'].iloc[i]
                    else:
                        df.iloc[i, df.columns.get_loc('supertrend')] = df['upper'].iloc[i]
            
            return {
                'value': float(df['supertrend'].iloc[-1]),
                'is_uptrend': bool(df['in_uptrend'].iloc[-1])
            }
            
        except Exception as e:
            print(f"[ERROR] SuperTrend calculation: {e}", flush=True)
            return {'value': None, 'is_uptrend': None}

class StreamManager:
    """WebSocket stream manager"""
    
    def __init__(self, trader):
        self.trader = trader
        self.api_client = self._setup_client()
        self.is_connected = False
        self.reconnect_delay = 5
        
    def _setup_client(self):
        configuration = upstox_client.Configuration()
        configuration.access_token = ACCESS_TOKEN
        return upstox_client.ApiClient(configuration)
    
    def connect(self):
        try:
            instrument_keys = ["NSE_INDEX|Nifty 50"]
            
            def on_message(message):
                try:
                    if "feeds" in message:
                        for inst_key, feed_data in message["feeds"].items():
                            ltpc_data = feed_data.get("ltpc", {})
                            if ltpc_data:
                                tick_data = {
                                    "last_price": ltpc_data.get("ltp", 0),
                                    "ltp": ltpc_data.get("ltp", 0),
                                    "volume": ltpc_data.get("ltq", 1)
                                }
                                self.trader.process_tick(tick_data)
                except Exception as e:
                    print(f"[ERROR] Message processing: {e}", flush=True)
            
            def on_error(error):
                print(f"[ERROR] WebSocket: {error}", flush=True)
                self.is_connected = False
                self._reconnect()
            
            def on_close():
                print("[INFO] WebSocket connection closed", flush=True)
                self.is_connected = False
                self._reconnect()
            
            self.streamer = upstox_client.MarketDataStreamerV3(
                self.api_client, instrument_keys, "ltpc"
            )
            
            self.streamer.on("message", on_message)
            self.streamer.on("error", on_error)
            self.streamer.on("close", on_close)
            
            print("[INFO] Connecting to market data stream...", flush=True)
            self.streamer.connect()
            self.is_connected = True
            
        except Exception as e:
            print(f"[ERROR] Connection failed: {e}", flush=True)
            self._reconnect()
    
    def _reconnect(self):
        if not self.is_connected:
            print(f"[INFO] Reconnecting in {self.reconnect_delay} seconds...", flush=True)
            time.sleep(self.reconnect_delay)
            self.reconnect_delay = min(self.reconnect_delay * 2, 60)
            self.connect()

def validate_token():
    """Validate access token"""
    if not ACCESS_TOKEN or len(ACCESS_TOKEN) < 50:
        print("[ERROR] Invalid or missing access token", flush=True)
        return False
    print("[INFO] Access token validated", flush=True)
    return True

def main():
    """Main trading function"""
    try:
        # Setup logging
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(f"logs/scalping_{datetime.now().strftime('%Y%m%d')}.log", encoding='utf-8'),
                logging.StreamHandler() if not IS_GITHUB_ACTIONS else logging.NullHandler()
            ]
        )
        
        print(f"[STARTUP] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | NIFTY AGGRESSIVE SCALPING | 10% Target | 25% Stop | Continuous Trading")
        print(f"[CONFIG] Target: {TARGET_PERCENTAGE*100:.0f}% | Stop: {STOP_LOSS_PERCENTAGE*100:.0f}% | Cooldown: {SIGNAL_COOLDOWN_SECONDS//60}min | VWAP: {(BASE_BULL_VWAP_THRESHOLD-1)*100:.2f}% | Signal: {MIN_SIGNAL_STRENGTH}")
        
        # Validate token
        if not validate_token():
            sys.exit(1)
        
        # Check market hours
        if not is_market_open():
            force_run = os.environ.get('FORCE_RUN', 'false').lower() == 'true'
            if not force_run:
                market_open, market_close = get_market_hours()
                print(f"[INFO] Market hours: {market_open.strftime('%H:%M')} - {market_close.strftime('%H:%M')}", flush=True)
                print("[INFO] Outside market hours. Set FORCE_RUN=true to override.", flush=True)
                return
        
        # Initialize system
        trader = ScalpingTrader()
        stream_manager = StreamManager(trader)
        
        print(f"[INIT] Access token validated | Options API ready | Max loss: {trader.risk_manager.max_daily_loss}")
        print(f"[SYSTEM] {datetime.now().strftime('%H:%M:%S')} | Scalping mode ACTIVE | Connecting to market stream...")
        
        # Connect and trade
        stream_manager.connect()
        
        # Run until market close
        start_time = datetime.now()
        
        while is_market_open():
            time.sleep(1)
        
        # Session summary
        total_trades = trader.risk_manager.winning_trades + trader.risk_manager.losing_trades
        win_rate = trader.risk_manager.get_win_rate()
        pnl_status = "PROFIT" if trader.risk_manager.daily_pnl > 0 else "LOSS" if trader.risk_manager.daily_pnl < 0 else "FLAT"
        
        print(f"[SESSION] {datetime.now().strftime('%H:%M:%S')} | COMPLETE | Status: {pnl_status} | Trades: {trader.risk_manager.trades_today} | Win/Loss: {trader.risk_manager.winning_trades}/{trader.risk_manager.losing_trades} | Win Rate: {win_rate:.0f}% | Daily P&L: Rs{trader.risk_manager.daily_pnl:+.0f}")
        
    except KeyboardInterrupt:
        print(f"\n[SYSTEM] {datetime.now().strftime('%H:%M:%S')} | Manual shutdown requested | Final P&L: Rs{trader.risk_manager.daily_pnl:+.0f}" if 'trader' in locals() else "[SYSTEM] Shutdown")
    except Exception as e:
        print(f"\n[FATAL] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()