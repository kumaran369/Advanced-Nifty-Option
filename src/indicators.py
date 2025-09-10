"""
Technical Indicators Module
Separate module for all indicator calculations
"""

import numpy as np
from collections import deque
from datetime import datetime
from math import log, sqrt

class ImprovedSignalGenerator:
    """Enhanced signal generation with proper option analysis"""
    
    def __init__(self):
        self.vwap_cumulative = {'sum_pv': 0, 'sum_v': 0, 'day_start': None}
        self.rsi_state = {'gains': [], 'losses': [], 'avg_gain': None, 'avg_loss': None}
        self.volatility_estimator = VolatilityEstimator()
        self.prev_price = None
        
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
        if not hasattr(self, 'prev_price') or self.prev_price is None:
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
        
        if self.rsi_state['avg_gain'] is not None:
            if self.rsi_state['avg_loss'] != 0:
                rs = self.rsi_state['avg_gain'] / self.rsi_state['avg_loss']
                return 100 - (100 / (1 + rs))
            else:
                return 100 if self.rsi_state['avg_gain'] > 0 else 50
        
        return None
    
    def generate_signal(self, market_data):
        """Generate trading signal with proper option analysis"""
        price = market_data.get('price')
        vwap = market_data.get('vwap')
        rsi = market_data.get('rsi')
        supertrend = market_data.get('supertrend')
        volatility = market_data.get('volatility')
        
        if not all([vwap, rsi, supertrend]):
            return None
        
        if not supertrend.get('value'):
            return None
        
        # Market hours check
        from datetime import datetime
        current_time = datetime.now()
        market_open = current_time.replace(hour=9, minute=15, second=0)
        market_close = current_time.replace(hour=15, minute=30, second=0)
        
        if not (market_open <= current_time <= market_close):
            return None
        
        # Enhanced signal logic with multiple confirmations
        signal = None
        
        # Bullish signal
        if (price > vwap * 1.001 and  # Price above VWAP with buffer
            supertrend.get('is_uptrend', False) and
            55 < rsi < 70 and  # Not overbought
            volatility < 0.25):  # Not too volatile
            
            signal = {
                'type': 'CE',
                'strength': self._calculate_signal_strength(price, vwap, rsi, True)
            }
        
        # Bearish signal
        elif (price < vwap * 0.999 and  # Price below VWAP with buffer
              not supertrend.get('is_uptrend', True) and
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
        self.last_price = None
        
    def update(self, price):
        """Update volatility estimate with new price"""
        if self.last_price is not None and self.last_price > 0:
            try:
                ret = log(price / self.last_price)
                self.returns.append(ret)
            except:
                pass
        self.last_price = price
        
    def get_volatility(self):
        """Get annualized volatility estimate"""
        if len(self.returns) < 5:
            return 0.15  # Default 15% volatility
        
        try:
            # Simple historical volatility
            std = np.std(self.returns)
            annual_vol = std * sqrt(252 * 375)  # 375 minutes per trading day
            
            # Cap volatility between reasonable bounds
            return max(0.10, min(0.50, annual_vol))
        except:
            return 0.15
