"""
Nifty Option 1-lot Scalping Signal Generator (no order placement) - WINDOWS VERSION
Author: Enhanced educational script with proper options pricing and error handling
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
import json
from typing import Dict, Optional, Tuple

# -------- SETTINGS ----------
# Load credentials from environment or config file
API_KEY = os.environ.get("UPSTOX_API_KEY", "YOUR_API_KEY")
ACCESS_TOKEN = os.environ.get("UPSTOX_ACCESS_TOKEN", "YOUR_ACCESS_TOKEN")

# Trading parameters
VWAP_WINDOW = 300        # number of ticks for VWAP
RSI_PERIOD = 14
RSI_SMOOTHING = True     # Use proper Wilder's smoothing
SUPER_ATR_PERIOD = 10
SUPER_MULTIPLIER = 3.0
MAX_SIGNALS_PER_DAY = 5

# Options parameters
RISK_FREE_RATE = 0.065   # 6.5% risk-free rate
DEFAULT_IV = 0.15        # 15% implied volatility as default
DELTA_THRESHOLD = 0.4    # Minimum delta for trade entry

# Risk Management
MAX_RISK_PER_TRADE = 0.02  # 2% of capital per trade
VOLATILITY_STOP_MULTIPLIER = 2.0  # Stop loss at 2x IV move
TIME_BASED_EXIT_HOURS = 2  # Exit if no movement in 2 hours

# Market Hours
MARKET_OPEN = datetime.now().replace(hour=9, minute=15, second=0, microsecond=0)
MARKET_CLOSE = datetime.now().replace(hour=15, minute=30, second=0, microsecond=0)
# ----------------------------

# Error handling decorator
def safe_execute(func):
    """Decorator for error handling with retry logic"""
    def wrapper(*args, **kwargs):
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                retry_count += 1
                logging.error(f"Error in {func.__name__}: {e}. Retry {retry_count}/{max_retries}")
                if retry_count >= max_retries:
                    logging.error(f"Max retries reached for {func.__name__}")
                    return None
                time.sleep(2 ** retry_count)  # Exponential backoff
        return None
    return wrapper

class BlackScholesCalculator:
    """Proper options pricing using Black-Scholes model"""
    
    @staticmethod
    def calculate_option_price(S, K, T, r, sigma, option_type='CE'):
        """
        Black-Scholes option pricing
        S: Spot price
        K: Strike price
        T: Time to expiry (in years)
        r: Risk-free rate
        sigma: Implied volatility
        """
        if T <= 0:
            return max(0, S - K) if option_type == 'CE' else max(0, K - S)
        
        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        
        if option_type == 'CE':
            price = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
        else:  # PE
            price = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return round(price, 2)
    
    @staticmethod
    def calculate_delta(S, K, T, r, sigma, option_type='CE'):
        """Calculate option delta"""
        if T <= 0:
            return 1.0 if (S > K and option_type == 'CE') else 0.0
        
        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        
        if option_type == 'CE':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1
    
    @staticmethod
    def calculate_gamma(S, K, T, r, sigma):
        """Calculate option gamma"""
        if T <= 0:
            return 0.0
        
        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        return norm.pdf(d1) / (S * sigma * sqrt(T))
    
    @staticmethod
    def calculate_theta(S, K, T, r, sigma, option_type='CE'):
        """Calculate option theta (per day)"""
        if T <= 0:
            return 0.0
        
        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        
        if option_type == 'CE':
            theta = (-S * norm.pdf(d1) * sigma / (2 * sqrt(T)) 
                    - r * K * exp(-r * T) * norm.cdf(d2))
        else:
            theta = (-S * norm.pdf(d1) * sigma / (2 * sqrt(T)) 
                    + r * K * exp(-r * T) * norm.cdf(-d2))
        
        return theta / 365  # Convert to per day

class ImprovedSignalGenerator:
    """Enhanced signal generation with proper option analysis"""
    
    def __init__(self):
        self.vwap_cumulative = {'sum_pv': 0, 'sum_v': 0, 'day_start': None}
        self.rsi_state = {'gains': [], 'losses': [], 'avg_gain': None, 'avg_loss': None}
        self.volatility_estimator = VolatilityEstimator()
        
    def update_vwap(self, price, volume, timestamp):
        """Calculate cumulative VWAP (resets daily)"""
        current_day = datetime.fromtimestamp(timestamp).date()
        
        if self.vwap_cumulative['day_start'] != current_day:
            # Reset for new day
            self.vwap_cumulative = {
                'sum_pv': price * volume,
                'sum_v': volume,
                'day_start': current_day
            }
        else:
            self.vwap_cumulative['sum_pv'] += price * volume
            self.vwap_cumulative['sum_v'] += volume
        
        if self.vwap_cumulative['sum_v'] > 0:
            return self.vwap_cumulative['sum_pv'] / self.vwap_cumulative['sum_v']
        return price
    
    def update_rsi(self, price, period=14):
        """Calculate RSI with proper Wilder's smoothing"""
        if not hasattr(self, 'prev_price'):
            self.prev_price = price
            return None
        
        change = price - self.prev_price
        gain = change if change > 0 else 0
        loss = -change if change < 0 else 0
        
        if self.rsi_state['avg_gain'] is None:
            # Initialize
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
        
        if self.rsi_state['avg_gain'] is not None and self.rsi_state['avg_loss'] != 0:
            rs = self.rsi_state['avg_gain'] / self.rsi_state['avg_loss']
            return 100 - (100 / (1 + rs))
        
        return None
    
    def generate_signal(self, market_data: Dict) -> Optional[Dict]:
        """Generate trading signal with proper option analysis"""
        price = market_data['price']
        vwap = market_data['vwap']
        rsi = market_data['rsi']
        supertrend = market_data['supertrend']
        volatility = market_data['volatility']
        
        if not all([vwap, rsi, supertrend['value']]):
            return None
        
        # Market hours check
        current_time = datetime.now()
        if not (MARKET_OPEN <= current_time <= MARKET_CLOSE):
            return None
        
        # Enhanced signal logic with multiple confirmations
        signal = None
        
        # Bullish signal
        if (price > vwap * 1.001 and  # Price above VWAP with buffer
            supertrend['is_uptrend'] and
            55 < rsi < 70 and  # Not overbought
            volatility < 0.25):  # Not too volatile
            
            signal = {
                'type': 'CE',
                'strength': self._calculate_signal_strength(price, vwap, rsi, True)
            }
        
        # Bearish signal
        elif (price < vwap * 0.999 and  # Price below VWAP with buffer
              not supertrend['is_uptrend'] and
              30 < rsi < 45 and  # Not oversold
              volatility < 0.25):
            
            signal = {
                'type': 'PE',
                'strength': self._calculate_signal_strength(price, vwap, rsi, False)
            }
        
        return signal
    
    def _calculate_signal_strength(self, price, vwap, rsi, is_bullish):
        """Calculate signal strength (0-100)"""
        vwap_distance = abs(price - vwap) / vwap * 100
        rsi_strength = (rsi - 50) / 50 * 100 if is_bullish else (50 - rsi) / 50 * 100
        
        # Weighted average
        strength = (vwap_distance * 0.4 + abs(rsi_strength) * 0.6)
        return min(100, max(0, strength))

class VolatilityEstimator:
    """Estimate real-time volatility for option pricing"""
    
    def __init__(self, window=20):
        self.returns = deque(maxlen=window)
        self.garch_params = {'omega': 0.00001, 'alpha': 0.1, 'beta': 0.85}
        
    def update(self, price):
        """Update volatility estimate with new price"""
        if hasattr(self, 'last_price'):
            ret = log(price / self.last_price)
            self.returns.append(ret)
        self.last_price = price
        
    def get_volatility(self):
        """Get annualized volatility estimate"""
        if len(self.returns) < 5:
            return DEFAULT_IV
        
        # Simple historical volatility
        std = np.std(self.returns)
        annual_vol = std * sqrt(252 * 375)  # 375 minutes per trading day
        
        # Cap volatility between reasonable bounds
        return max(0.10, min(0.50, annual_vol))

class RiskManager:
    """Advanced risk management for options trading"""
    
    def __init__(self, initial_capital=100000):
        self.capital = initial_capital
        self.open_positions = {}
        self.daily_pnl = 0
        self.max_daily_loss = initial_capital * 0.06  # 6% daily loss limit
        
    def calculate_position_size(self, option_price, stop_loss_price):
        """Calculate position size based on risk per trade"""
        risk_per_share = abs(option_price - stop_loss_price)
        max_risk_amount = self.capital * MAX_RISK_PER_TRADE
        
        if risk_per_share > 0:
            shares = int(max_risk_amount / risk_per_share)
            # Round to nearest lot (75 for Nifty)
            lots = max(1, shares // 75)
            return lots * 75
        return 75  # Default 1 lot
    
    def calculate_dynamic_stops(self, option_details, market_volatility):
        """Calculate dynamic stop loss and target based on Greeks"""
        premium = option_details['premium']
        delta = option_details['delta']
        gamma = option_details['gamma']
        theta = option_details['theta']
        
        # Volatility-based stop
        vol_stop = premium * (1 - VOLATILITY_STOP_MULTIPLIER * market_volatility)
        
        # Time decay consideration
        time_stop = premium - abs(theta * 2)  # 2 days of theta
        
        # Use the higher stop for safety
        stop_loss = max(vol_stop, time_stop, premium * 0.5)  # Min 50% stop
        
        # Target based on risk-reward and probability
        risk = premium - stop_loss
        prob_profit = abs(delta)  # Rough probability estimate
        
        # Adjust target based on probability
        if prob_profit > 0.6:
            target = premium + risk * 2.0  # 2:1 RR
        elif prob_profit > 0.4:
            target = premium + risk * 1.5  # 1.5:1 RR
        else:
            target = premium + risk * 1.2  # 1.2:1 RR
        
        return {
            'stop_loss': round(stop_loss, 2),
            'target': round(target, 2),
            'risk_reward': round((target - premium) / (premium - stop_loss), 2)
        }

class OptionsTrader:
    """Main trading system with proper error handling"""
    
    def __init__(self):
        self.signal_generator = ImprovedSignalGenerator()
        self.risk_manager = RiskManager()
        self.bs_calculator = BlackScholesCalculator()
        self.active_positions = {}
        self.tick_buffer = deque(maxlen=5000)
        self.last_processed_time = 0
        self.reconnect_attempts = 0
        self.setup_logging()
        
    def setup_logging(self):
        """Setup comprehensive logging for Windows compatibility"""
        log_filename = f"options_trader_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Clear any existing handlers
        logger = logging.getLogger()
        logger.handlers = []
        
        # Create formatters
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File handler with UTF-8 encoding
        file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        
        # Console handler - no emojis for Windows compatibility
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Configure logger
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    @safe_execute
    def process_tick(self, tick_data):
        """Process incoming tick with error handling"""
        try:
            # Validate tick data
            if not self._validate_tick(tick_data):
                return
            
            timestamp = time.time()
            price = float(tick_data.get("last_price", tick_data.get("ltp", 0)))
            volume = float(tick_data.get("volume", 1))
            
            # Skip if duplicate or stale
            if timestamp - self.last_processed_time < 0.1:  # 100ms debounce
                return
            
            self.last_processed_time = timestamp
            self.tick_buffer.append((timestamp, price, volume))
            
            # Update indicators
            market_data = self._update_market_data(price, volume, timestamp)
            
            # Check existing positions
            self._manage_positions(market_data)
            
            # Generate new signals
            if len(self.active_positions) < 2:  # Max 2 concurrent positions
                signal = self.signal_generator.generate_signal(market_data)
                if signal and signal['strength'] > 60:  # Min 60% signal strength
                    self._execute_signal(signal, market_data)
                    
        except Exception as e:
            logging.error(f"Error processing tick: {e}")
            
    def _validate_tick(self, tick_data):
        """Validate incoming tick data"""
        required_fields = ["last_price", "ltp", "volume"]
        if not any(field in tick_data for field in required_fields[:2]):
            return False
            
        price = float(tick_data.get("last_price", tick_data.get("ltp", 0)))
        if price <= 0 or price > 100000:  # Sanity check
            return False
            
        return True
    
    def _update_market_data(self, price, volume, timestamp):
        """Update all market indicators"""
        # Update volatility
        self.signal_generator.volatility_estimator.update(price)
        
        # Update indicators
        vwap = self.signal_generator.update_vwap(price, volume, timestamp)
        rsi = self.signal_generator.update_rsi(price)
        
        # Calculate supertrend
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
        """Execute trading signal with proper option pricing"""
        try:
            spot_price = market_data['price']
            option_type = signal['type']
            
            # Get ATM strike
            strike = self._get_atm_strike(spot_price)
            
            # Calculate time to expiry (weekly options)
            expiry = self._get_next_expiry()
            time_to_expiry = (expiry - datetime.now()).total_seconds() / (365 * 24 * 3600)
            
            # Calculate option premium using Black-Scholes
            iv = market_data['volatility']
            premium = self.bs_calculator.calculate_option_price(
                spot_price, strike, time_to_expiry, RISK_FREE_RATE, iv, option_type
            )
            
            # Calculate Greeks
            delta = self.bs_calculator.calculate_delta(
                spot_price, strike, time_to_expiry, RISK_FREE_RATE, iv, option_type
            )
            gamma = self.bs_calculator.calculate_gamma(
                spot_price, strike, time_to_expiry, RISK_FREE_RATE, iv
            )
            theta = self.bs_calculator.calculate_theta(
                spot_price, strike, time_to_expiry, RISK_FREE_RATE, iv, option_type
            )
            
            # Check delta threshold
            if abs(delta) < DELTA_THRESHOLD:
                logging.info(f"Signal rejected: Delta {delta:.3f} below threshold")
                return
            
            # Prepare option details
            option_details = {
                'strike': strike,
                'type': option_type,
                'premium': premium,
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'iv': iv,
                'expiry': expiry,
                'spot_entry': spot_price
            }
            
            # Calculate risk management levels
            stops_targets = self.risk_manager.calculate_dynamic_stops(
                option_details, market_data['volatility']
            )
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                premium, stops_targets['stop_loss']
            )
            
            # Create position
            position = {
                **option_details,
                **stops_targets,
                'quantity': position_size,
                'entry_time': datetime.now(),
                'signal_strength': signal['strength']
            }
            
            # Log signal
            self._log_signal(position)
            
            # Store position
            position_id = f"{option_type}_{strike}_{int(time.time())}"
            self.active_positions[position_id] = position
            
        except Exception as e:
            logging.error(f"Error executing signal: {e}")
    
    def _manage_positions(self, market_data):
        """Manage existing positions with real-time P&L"""
        for position_id, position in list(self.active_positions.items()):
            try:
                # Recalculate option price
                time_elapsed = (datetime.now() - position['entry_time']).total_seconds()
                time_to_expiry = ((position['expiry'] - datetime.now()).total_seconds() / 
                                 (365 * 24 * 3600))
                
                if time_to_expiry <= 0:
                    # Option expired
                    self._close_position(position_id, 0, "EXPIRED")
                    continue
                
                # Current option price
                current_premium = self.bs_calculator.calculate_option_price(
                    market_data['price'],
                    position['strike'],
                    time_to_expiry,
                    RISK_FREE_RATE,
                    market_data['volatility'],
                    position['type']
                )
                
                # Calculate P&L
                pnl_per_lot = (current_premium - position['premium']) * 75
                total_pnl = pnl_per_lot * (position['quantity'] / 75)
                pnl_percentage = ((current_premium - position['premium']) / 
                                position['premium'] * 100)
                
                # Display P&L
                self._display_pnl(position, current_premium, total_pnl, pnl_percentage)
                
                # Check exit conditions
                if current_premium <= position['stop_loss']:
                    self._close_position(position_id, current_premium, "STOP_LOSS")
                elif current_premium >= position['target']:
                    self._close_position(position_id, current_premium, "TARGET")
                elif time_elapsed > TIME_BASED_EXIT_HOURS * 3600:
                    # Time-based exit
                    self._close_position(position_id, current_premium, "TIME_EXIT")
                    
            except Exception as e:
                logging.error(f"Error managing position {position_id}: {e}")
    
    def _close_position(self, position_id, exit_price, reason):
        """Close position and log results"""
        position = self.active_positions[position_id]
        pnl = (exit_price - position['premium']) * position['quantity']
        
        logging.info(f"""
        POSITION CLOSED - {reason}
        Type: {position['type']} {position['strike']}
        Entry: Rs.{position['premium']:.2f} | Exit: Rs.{exit_price:.2f}
        P&L: Rs.{pnl:+,.0f} ({pnl/position['quantity']*100/position['premium']:+.1f}%)
        Duration: {datetime.now() - position['entry_time']}
        """)
        
        # Update risk manager
        self.risk_manager.daily_pnl += pnl
        
        # Remove position
        del self.active_positions[position_id]
    
    def _display_pnl(self, position, current_premium, total_pnl, pnl_percentage):
        """Display live P&L with Greeks"""
        pnl_indicator = "[+]" if total_pnl >= 0 else "[-]"
        
        print(f"\r{pnl_indicator} {position['type']} {position['strike']} | "
              f"Entry: Rs.{position['premium']:.2f} | Current: Rs.{current_premium:.2f} | "
              f"P&L: Rs.{total_pnl:+,.0f} ({pnl_percentage:+.1f}%) | "
              f"Theta: {position['theta']:.2f}", end='', flush=True)
    
    def _log_signal(self, position):
        """Log trading signal with full details"""
        logging.info(f"""
        ========== NEW SIGNAL ==========
        Type: {position['type']} (Strike: {position['strike']})
        Spot Price: Rs.{position['spot_entry']:.2f}
        Premium: Rs.{position['premium']:.2f}
        Quantity: {position['quantity']} ({position['quantity']//75} lots)
        Target: Rs.{position['target']:.2f} | Stop Loss: Rs.{position['stop_loss']:.2f}
        Greeks - Delta: {position['delta']:.3f} | Gamma: {position['gamma']:.4f} | Theta: {position['theta']:.2f}
        IV: {position['iv']*100:.1f}% | R:R: {position['risk_reward']:.1f}
        Signal Strength: {position['signal_strength']:.0f}%
        ================================
        """)
    
    def _get_atm_strike(self, spot_price, gap=50):
        """Get ATM strike price"""
        return round(spot_price / gap) * gap
    
    def _get_next_expiry(self):
        """Get next weekly expiry (Thursday)"""
        today = datetime.now()
        days_ahead = 3 - today.weekday()  # Thursday is 3
        if days_ahead <= 0:
            days_ahead += 7
        return today + timedelta(days=days_ahead)
    
    def _build_candles(self):
        """Build minute candles from ticks"""
        if len(self.tick_buffer) < 10:
            return pd.DataFrame()
            
        df = pd.DataFrame(list(self.tick_buffer), columns=['ts', 'price', 'vol'])
        df['minute'] = pd.to_datetime(df['ts'], unit='s').dt.floor('T')
        
        candles = df.groupby('minute').agg({
            'price': ['first', 'max', 'min', 'last'],
            'vol': 'sum'
        })
        
        candles.columns = ['open', 'high', 'low', 'close', 'volume']
        return candles.reset_index()
    
    def _calculate_supertrend(self, df, period=10, multiplier=3):
        """Calculate SuperTrend indicator"""
        if len(df) < period + 2:
            return {'value': None, 'is_uptrend': None}
            
        # ATR calculation
        df['tr'] = df[['high', 'low', 'close']].apply(
            lambda x: max(x['high'] - x['low'], 
                         abs(x['high'] - x['close']), 
                         abs(x['low'] - x['close'])), axis=1
        )
        
        df['atr'] = df['tr'].rolling(period).mean()
        
        # Basic bands
        hl2 = (df['high'] + df['low']) / 2
        df['basic_upper'] = hl2 + multiplier * df['atr']
        df['basic_lower'] = hl2 - multiplier * df['atr']
        
        # SuperTrend calculation
        df['upper'] = df['basic_upper']
        df['lower'] = df['basic_lower']
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] <= df['upper'].iloc[i-1]:
                df.loc[df.index[i], 'upper'] = min(df['basic_upper'].iloc[i], 
                                                   df['upper'].iloc[i-1])
            
            if df['close'].iloc[i] >= df['lower'].iloc[i-1]:
                df.loc[df.index[i], 'lower'] = max(df['basic_lower'].iloc[i], 
                                                   df['lower'].iloc[i-1])
        
        # Determine trend
        last_close = df['close'].iloc[-1]
        last_lower = df['lower'].iloc[-1]
        last_upper = df['upper'].iloc[-1]
        
        if last_close > last_lower:
            return {'value': last_lower, 'is_uptrend': True}
        else:
            return {'value': last_upper, 'is_uptrend': False}

# WebSocket connection manager with reconnection logic
class StreamManager:
    """Manage WebSocket connection with automatic reconnection"""
    
    def __init__(self, trader):
        self.trader = trader
        self.api_client = self._setup_client()
        self.streamer = None
        self.is_connected = False
        self.reconnect_delay = 5
        
    def _setup_client(self):
        """Setup Upstox API client"""
        configuration = upstox_client.Configuration()
        configuration.access_token = ACCESS_TOKEN
        return upstox_client.ApiClient(configuration)
    
    @safe_execute
    def connect(self):
        """Connect to market data stream with error handling"""
        try:
            instrument_keys = ["NSE_INDEX|Nifty 50"]
            
            def on_message(message):
                try:
                    if message.get("type") == "market_info":
                        self._handle_market_status(message)
                    elif "feeds" in message:
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
                    logging.error(f"Error processing message: {e}")
            
            def on_error(error):
                logging.error(f"WebSocket error: {error}")
                self.is_connected = False
                self._reconnect()
            
            def on_close():
                logging.info("WebSocket connection closed")
                self.is_connected = False
                self._reconnect()
            
            # Create streamer
            self.streamer = upstox_client.MarketDataStreamerV3(
                self.api_client,
                instrument_keys,
                "ltpc"
            )
            
            # Set handlers
            self.streamer.on("message", on_message)
            self.streamer.on("error", on_error)
            self.streamer.on("close", on_close)
            
            # Connect
            logging.info("Connecting to market data stream...")
            self.streamer.connect()
            self.is_connected = True
            self.reconnect_delay = 5  # Reset delay on successful connection
            
        except Exception as e:
            logging.error(f"Connection failed: {e}")
            self._reconnect()
    
    def _reconnect(self):
        """Reconnect with exponential backoff"""
        if self.is_connected:
            return
            
        logging.info(f"Reconnecting in {self.reconnect_delay} seconds...")
        time.sleep(self.reconnect_delay)
        self.reconnect_delay = min(self.reconnect_delay * 2, 300)  # Max 5 minutes
        self.connect()
    
    def _handle_market_status(self, message):
        """Handle market status updates"""
        market_status = message.get('marketInfo', {}).get('segmentStatus', {})
        nse_status = market_status.get('NSE_INDEX', 'UNKNOWN')
        
        if nse_status in ['NORMAL_OPEN', 'PRE_OPEN']:
            logging.info("[MARKET OPEN] Trading Active")
        elif nse_status in ['CLOSING_END', 'NORMAL_CLOSE']:
            logging.info("[MARKET CLOSED]")
            # Close all positions at market close
            for position_id in list(self.trader.active_positions.keys()):
                self.trader._close_position(position_id, 0, "MARKET_CLOSE")
        else:
            logging.info(f"[MARKET STATUS] {nse_status}")

def main():
    """Main entry point with error handling"""
    try:
        # Validate credentials
        if API_KEY == "YOUR_API_KEY" or ACCESS_TOKEN == "YOUR_ACCESS_TOKEN":
            logging.error("Please set UPSTOX_API_KEY and UPSTOX_ACCESS_TOKEN environment variables")
            sys.exit(1)
        
        # Initialize trader
        trader = OptionsTrader()
        
        # Initialize stream manager
        stream_manager = StreamManager(trader)
        
        # Start trading
        logging.info("Starting Nifty Options Trading System...")
        logging.info(f"Risk per trade: {MAX_RISK_PER_TRADE*100}%")
        logging.info(f"Max daily signals: {MAX_SIGNALS_PER_DAY}")
        
        # Connect to market
        stream_manager.connect()
        
        # Keep running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logging.info("Shutting down trading system...")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
