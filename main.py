"""
Nifty Option 1-lot Scalping Signal Generator - SIMPLIFIED VERSION
For GitHub Actions automated trading
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

# -------- SETTINGS ----------
# Read token from file
def get_token():
    token_file = "token.txt"
    if os.path.exists(token_file):
        with open(token_file, 'r') as f:
            return f.read().strip()
    return os.environ.get("UPSTOX_ACCESS_TOKEN", "")

API_KEY = "a17874e5-9c5b-45d1-aa21-04ddd1f34c67"
ACCESS_TOKEN = get_token()

# Trading parameters
VWAP_WINDOW = 300
RSI_PERIOD = 14
SUPER_ATR_PERIOD = 10
SUPER_MULTIPLIER = 3.0
MAX_SIGNALS_PER_DAY = 5

# Options parameters
RISK_FREE_RATE = 0.065
DEFAULT_IV = 0.15
DELTA_THRESHOLD = 0.4

# Risk Management
MAX_RISK_PER_TRADE = 0.02
VOLATILITY_STOP_MULTIPLIER = 2.0
TIME_BASED_EXIT_HOURS = 2

# Market Hours
MARKET_OPEN = datetime.now().replace(hour=9, minute=15, second=0, microsecond=0)
MARKET_CLOSE = datetime.now().replace(hour=15, minute=30, second=0, microsecond=0)

# Setup simple logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/trading_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)

# Copy all the classes from your original file
class BlackScholesCalculator:
    """Proper options pricing using Black-Scholes model"""
    
    @staticmethod
    def calculate_option_price(S, K, T, r, sigma, option_type='CE'):
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
        if T <= 0:
            return 1.0 if (S > K and option_type == 'CE') else 0.0
        
        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        
        if option_type == 'CE':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1
    
    @staticmethod
    def calculate_gamma(S, K, T, r, sigma):
        if T <= 0:
            return 0.0
        
        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        return norm.pdf(d1) / (S * sigma * sqrt(T))
    
    @staticmethod
    def calculate_theta(S, K, T, r, sigma, option_type='CE'):
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
        
        return theta / 365

class ImprovedSignalGenerator:
    def __init__(self):
        self.vwap_cumulative = {'sum_pv': 0, 'sum_v': 0, 'day_start': None}
        self.rsi_state = {'gains': [], 'losses': [], 'avg_gain': None, 'avg_loss': None}
        self.volatility_estimator = VolatilityEstimator()
        
    def update_vwap(self, price, volume, timestamp):
        current_day = datetime.fromtimestamp(timestamp).date()
        
        if self.vwap_cumulative['day_start'] != current_day:
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
        if not hasattr(self, 'prev_price'):
            self.prev_price = price
            return None
        
        change = price - self.prev_price
        gain = change if change > 0 else 0
        loss = -change if change < 0 else 0
        
        if self.rsi_state['avg_gain'] is None:
            self.rsi_state['gains'].append(gain)
            self.rsi_state['losses'].append(loss)
            
            if len(self.rsi_state['gains']) >= period:
                self.rsi_state['avg_gain'] = sum(self.rsi_state['gains']) / period
                self.rsi_state['avg_loss'] = sum(self.rsi_state['losses']) / period
                self.rsi_state['gains'] = []
                self.rsi_state['losses'] = []
        else:
            self.rsi_state['avg_gain'] = (self.rsi_state['avg_gain'] * (period - 1) + gain) / period
            self.rsi_state['avg_loss'] = (self.rsi_state['avg_loss'] * (period - 1) + loss) / period
        
        self.prev_price = price
        
        if self.rsi_state['avg_gain'] is not None and self.rsi_state['avg_loss'] != 0:
            rs = self.rsi_state['avg_gain'] / self.rsi_state['avg_loss']
            return 100 - (100 / (1 + rs))
        
        return None
    
    def generate_signal(self, market_data: Dict) -> Optional[Dict]:
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
        
        signal = None
        
        # Bullish signal
        if (price > vwap * 1.001 and
            supertrend['is_uptrend'] and
            55 < rsi < 70 and
            volatility < 0.25):
            
            signal = {
                'type': 'CE',
                'strength': self._calculate_signal_strength(price, vwap, rsi, True)
            }
        
        # Bearish signal
        elif (price < vwap * 0.999 and
              not supertrend['is_uptrend'] and
              30 < rsi < 45 and
              volatility < 0.25):
            
            signal = {
                'type': 'PE',
                'strength': self._calculate_signal_strength(price, vwap, rsi, False)
            }
        
        return signal
    
    def _calculate_signal_strength(self, price, vwap, rsi, is_bullish):
        vwap_distance = abs(price - vwap) / vwap * 100
        rsi_strength = (rsi - 50) / 50 * 100 if is_bullish else (50 - rsi) / 50 * 100
        strength = (vwap_distance * 0.4 + abs(rsi_strength) * 0.6)
        return min(100, max(0, strength))

class VolatilityEstimator:
    def __init__(self, window=20):
        self.returns = deque(maxlen=window)
        self.last_price = None
        
    def update(self, price):
        if self.last_price is not None:
            ret = log(price / self.last_price)
            self.returns.append(ret)
        self.last_price = price
        
    def get_volatility(self):
        if len(self.returns) < 5:
            return DEFAULT_IV
        
        std = np.std(self.returns)
        annual_vol = std * sqrt(252 * 375)
        
        return max(0.10, min(0.50, annual_vol))

class RiskManager:
    def __init__(self, initial_capital=100000):
        self.capital = initial_capital
        self.open_positions = {}
        self.daily_pnl = 0
        self.max_daily_loss = initial_capital * 0.06
        
    def calculate_position_size(self, option_price, stop_loss_price):
        risk_per_share = abs(option_price - stop_loss_price)
        max_risk_amount = self.capital * MAX_RISK_PER_TRADE
        
        if risk_per_share > 0:
            shares = int(max_risk_amount / risk_per_share)
            lots = max(1, shares // 75)
            return lots * 75
        return 75
    
    def calculate_dynamic_stops(self, option_details, market_volatility):
        premium = option_details['premium']
        delta = option_details['delta']
        theta = option_details['theta']
        
        vol_stop = premium * (1 - VOLATILITY_STOP_MULTIPLIER * market_volatility)
        time_stop = premium - abs(theta * 2)
        
        stop_loss = max(vol_stop, time_stop, premium * 0.5)
        
        risk = premium - stop_loss
        prob_profit = abs(delta)
        
        if prob_profit > 0.6:
            target = premium + risk * 2.0
        elif prob_profit > 0.4:
            target = premium + risk * 1.5
        else:
            target = premium + risk * 1.2
        
        return {
            'stop_loss': round(stop_loss, 2),
            'target': round(target, 2),
            'risk_reward': round((target - premium) / (premium - stop_loss), 2)
        }

class OptionsTrader:
    def __init__(self):
        self.signal_generator = ImprovedSignalGenerator()
        self.risk_manager = RiskManager()
        self.bs_calculator = BlackScholesCalculator()
        self.active_positions = {}
        self.tick_buffer = deque(maxlen=5000)
        self.last_processed_time = 0
        self.signals_today = 0
        
    def process_tick(self, tick_data):
        try:
            if not self._validate_tick(tick_data):
                return
            
            timestamp = time.time()
            price = float(tick_data.get("last_price", tick_data.get("ltp", 0)))
            volume = float(tick_data.get("volume", 1))
            
            if timestamp - self.last_processed_time < 0.1:
                return
            
            self.last_processed_time = timestamp
            self.tick_buffer.append((timestamp, price, volume))
            
            market_data = self._update_market_data(price, volume, timestamp)
            
            # Simple console output
            if market_data['vwap'] and market_data['rsi']:
                print(f"Price: {price:.2f} | VWAP: {market_data['vwap']:.2f} | RSI: {market_data['rsi']:.2f}")
            
            self._manage_positions(market_data)
            
            if len(self.active_positions) < 2 and self.signals_today < MAX_SIGNALS_PER_DAY:
                signal = self.signal_generator.generate_signal(market_data)
                if signal and signal['strength'] > 60:
                    self._execute_signal(signal, market_data)
                    
        except Exception as e:
            logging.error(f"Error processing tick: {e}")
            
    def _validate_tick(self, tick_data):
        price = float(tick_data.get("last_price", tick_data.get("ltp", 0)))
        return 0 < price < 100000
    
    def _update_market_data(self, price, volume, timestamp):
        self.signal_generator.volatility_estimator.update(price)
        
        vwap = self.signal_generator.update_vwap(price, volume, timestamp)
        rsi = self.signal_generator.update_rsi(price)
        
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
        try:
            spot_price = market_data['price']
            option_type = signal['type']
            
            strike = self._get_atm_strike(spot_price)
            expiry = self._get_next_expiry()
            time_to_expiry = (expiry - datetime.now()).total_seconds() / (365 * 24 * 3600)
            
            iv = market_data['volatility']
            premium = self.bs_calculator.calculate_option_price(
                spot_price, strike, time_to_expiry, RISK_FREE_RATE, iv, option_type
            )
            
            delta = self.bs_calculator.calculate_delta(
                spot_price, strike, time_to_expiry, RISK_FREE_RATE, iv, option_type
            )
            
            if abs(delta) < DELTA_THRESHOLD:
                logging.info(f"Signal rejected: Delta {delta:.3f} below threshold")
                return
            
            gamma = self.bs_calculator.calculate_gamma(
                spot_price, strike, time_to_expiry, RISK_FREE_RATE, iv
            )
            theta = self.bs_calculator.calculate_theta(
                spot_price, strike, time_to_expiry, RISK_FREE_RATE, iv, option_type
            )
            
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
            
            stops_targets = self.risk_manager.calculate_dynamic_stops(
                option_details, market_data['volatility']
            )
            
            position_size = self.risk_manager.calculate_position_size(
                premium, stops_targets['stop_loss']
            )
            
            position = {
                **option_details,
                **stops_targets,
                'quantity': position_size,
                'entry_time': datetime.now(),
                'signal_strength': signal['strength']
            }
            
            # Log signal
            logging.info(f"""
========== NEW SIGNAL ==========
Type: {position['type']} (Strike: {position['strike']})
Spot: Rs.{position['spot_entry']:.2f}
Premium: Rs.{position['premium']:.2f}
Qty: {position['quantity']} ({position['quantity']//75} lots)
Target: Rs.{position['target']:.2f} | SL: Rs.{position['stop_loss']:.2f}
Delta: {position['delta']:.3f} | IV: {position['iv']*100:.1f}%
================================
            """)
            
            position_id = f"{option_type}_{strike}_{int(time.time())}"
            self.active_positions[position_id] = position
            self.signals_today += 1
            
        except Exception as e:
            logging.error(f"Error executing signal: {e}")
    
    def _manage_positions(self, market_data):
        for position_id, position in list(self.active_positions.items()):
            try:
                time_elapsed = (datetime.now() - position['entry_time']).total_seconds()
                time_to_expiry = ((position['expiry'] - datetime.now()).total_seconds() / 
                                 (365 * 24 * 3600))
                
                if time_to_expiry <= 0:
                    self._close_position(position_id, 0, "EXPIRED")
                    continue
                
                current_premium = self.bs_calculator.calculate_option_price(
                    market_data['price'],
                    position['strike'],
                    time_to_expiry,
                    RISK_FREE_RATE,
                    market_data['volatility'],
                    position['type']
                )
                
                pnl = (current_premium - position['premium']) * position['quantity']
                pnl_pct = ((current_premium - position['premium']) / position['premium'] * 100)
                
                # Simple P&L display
                print(f"{position['type']} {position['strike']} | P&L: Rs.{pnl:+,.0f} ({pnl_pct:+.1f}%)")
                
                if current_premium <= position['stop_loss']:
                    self._close_position(position_id, current_premium, "STOP_LOSS")
                elif current_premium >= position['target']:
                    self._close_position(position_id, current_premium, "TARGET")
                elif time_elapsed > TIME_BASED_EXIT_HOURS * 3600:
                    self._close_position(position_id, current_premium, "TIME_EXIT")
                    
            except Exception as e:
                logging.error(f"Error managing position: {e}")
    
    def _close_position(self, position_id, exit_price, reason):
        position = self.active_positions[position_id]
        pnl = (exit_price - position['premium']) * position['quantity']
        
        logging.info(f"""
POSITION CLOSED - {reason}
Type: {position['type']} {position['strike']}
Entry: Rs.{position['premium']:.2f} | Exit: Rs.{exit_price:.2f}
P&L: Rs.{pnl:+,.0f}
        """)
        
        self.risk_manager.daily_pnl += pnl
        del self.active_positions[position_id]
    
    def _get_atm_strike(self, spot_price, gap=50):
        return round(spot_price / gap) * gap
    
    def _get_next_expiry(self):
        today = datetime.now()
        days_ahead = 3 - today.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        return today + timedelta(days=days_ahead)
    
    def _build_candles(self):
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
        if len(df) < period + 2:
            return {'value': None, 'is_uptrend': None}
            
        df['tr'] = df[['high', 'low', 'close']].apply(
            lambda x: max(x['high'] - x['low'], 
                         abs(x['high'] - x['close']), 
                         abs(x['low'] - x['close'])), axis=1
        )
        
        df['atr'] = df['tr'].rolling(period).mean()
        
        hl2 = (df['high'] + df['low']) / 2
        df['basic_upper'] = hl2 + multiplier * df['atr']
        df['basic_lower'] = hl2 - multiplier * df['atr']
        
        df['upper'] = df['basic_upper']
        df['lower'] = df['basic_lower']
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] <= df['upper'].iloc[i-1]:
                df.loc[df.index[i], 'upper'] = min(df['basic_upper'].iloc[i], 
                                                   df['upper'].iloc[i-1])
            
            if df['close'].iloc[i] >= df['lower'].iloc[i-1]:
                df.loc[df.index[i], 'lower'] = max(df['basic_lower'].iloc[i], 
                                                   df['lower'].iloc[i-1])
        
        last_close = df['close'].iloc[-1]
        last_lower = df['lower'].iloc[-1]
        last_upper = df['upper'].iloc[-1]
        
        if last_close > last_lower:
            return {'value': last_lower, 'is_uptrend': True}
        else:
            return {'value': last_upper, 'is_uptrend': False}

class StreamManager:
    def __init__(self, trader):
        self.trader = trader
        self.api_client = self._setup_client()
        self.streamer = None
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
            
            self.streamer = upstox_client.MarketDataStreamerV3(
                self.api_client,
                instrument_keys,
                "ltpc"
            )
            
            self.streamer.on("message", on_message)
            self.streamer.on("error", on_error)
            self.streamer.on("close", on_close)
            
            logging.info("Connecting to market data stream...")
            self.streamer.connect()
            self.is_connected = True
            self.reconnect_delay = 5
            
        except Exception as e:
            logging.error(f"Connection failed: {e}")
            self._reconnect()
    
    def _reconnect(self):
        if self.is_connected:
            return
            
        logging.info(f"Reconnecting in {self.reconnect_delay} seconds...")
        time.sleep(self.reconnect_delay)
        self.reconnect_delay = min(self.reconnect_delay * 2, 300)
        self.connect()
    
    def _handle_market_status(self, message):
        market_status = message.get('marketInfo', {}).get('segmentStatus', {})
        nse_status = market_status.get('NSE_INDEX', 'UNKNOWN')
        
        if nse_status in ['NORMAL_OPEN', 'PRE_OPEN']:
            logging.info("[MARKET OPEN] Trading Active")
        elif nse_status in ['CLOSING_END', 'NORMAL_CLOSE']:
            logging.info("[MARKET CLOSED]")
            for position_id in list(self.trader.active_positions.keys()):
                self.trader._close_position(position_id, 0, "MARKET_CLOSE")
        else:
            logging.info(f"[MARKET STATUS] {nse_status}")

def validate_token():
    """Check if token is valid by attempting a simple API call"""
    if not ACCESS_TOKEN or ACCESS_TOKEN == "":
        logging.error("No access token found in token.txt")
        return False
    
    # Basic token validation
    if len(ACCESS_TOKEN) < 100:
        logging.error("Invalid token format")
        return False
    
    logging.info("Token loaded successfully")
    return True

def main():
    try:
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Validate token
        if not validate_token():
            sys.exit(1)
        
        # Check market hours
        now = datetime.now()
        if now.hour < 9 or now.hour >= 16:
            logging.info("Outside market hours. Current time: " + now.strftime("%H:%M"))
            
        # Initialize trader
        trader = OptionsTrader()
        stream_manager = StreamManager(trader)
        
        # Start trading
        logging.info("Starting Nifty Options Trading System...")
        logging.info(f"Risk per trade: {MAX_RISK_PER_TRADE*100}%")
        logging.info(f"Max daily signals: {MAX_SIGNALS_PER_DAY}")
        
        # Connect to market
        stream_manager.connect()
        
        # Run for market hours or until interrupted
        end_time = datetime.now().replace(hour=15, minute=30, second=0)
        while datetime.now() < end_time:
            time.sleep(1)
            
        logging.info("Market closed. Shutting down...")
            
    except KeyboardInterrupt:
        logging.info("Shutting down trading system...")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
