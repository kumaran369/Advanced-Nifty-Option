"""
Risk Management Module
Handles position sizing, stops, and risk limits
"""

import logging
from datetime import datetime

class RiskManager:
    """Advanced risk management for options trading"""
    
    def __init__(self, initial_capital=100000):
        self.capital = initial_capital
        self.open_positions = {}
        self.daily_pnl = 0
        self.max_daily_loss = initial_capital * 0.06  # 6% daily loss limit
        self.max_risk_per_trade = 0.02  # 2% per trade
        self.volatility_stop_multiplier = 2.0
        
    def calculate_position_size(self, option_price, stop_loss_price):
        """Calculate position size based on risk per trade"""
        if option_price <= 0 or stop_loss_price <= 0:
            return 75  # Default 1 lot
            
        risk_per_share = abs(option_price - stop_loss_price)
        max_risk_amount = self.capital * self.max_risk_per_trade
        
        if risk_per_share > 0:
            shares = int(max_risk_amount / risk_per_share)
            # Round to nearest lot (75 for Nifty)
            lots = max(1, shares // 75)
            return lots * 75
        return 75  # Default 1 lot
    
    def calculate_dynamic_stops(self, option_details, market_volatility):
        """Calculate dynamic stop loss and target based on Greeks"""
        premium = option_details.get('premium', 100)
        delta = option_details.get('delta', 0.5)
        gamma = option_details.get('gamma', 0.001)
        theta = option_details.get('theta', -1)
        
        # Volatility-based stop
        vol_stop = premium * (1 - self.volatility_stop_multiplier * market_volatility)
        
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
            'risk_reward': round((target - premium) / (premium - stop_loss), 2) if stop_loss < premium else 1.0
        }
    
    def can_take_trade(self, potential_loss):
        """Check if we can take a new trade based on risk limits"""
        # Check daily loss limit
        if self.daily_pnl <= -self.max_daily_loss:
            logging.warning("Daily loss limit reached. No new trades allowed.")
            return False
        
        # Check if potential loss would exceed daily limit
        if self.daily_pnl + potential_loss <= -self.max_daily_loss:
            logging.warning("Trade would exceed daily loss limit. Skipping.")
            return False
        
        # Check maximum positions (risk concentration)
        if len(self.open_positions) >= 3:
            logging.warning("Maximum concurrent positions reached.")
            return False
        
        return True
    
    def update_daily_pnl(self, pnl):
        """Update daily P&L"""
        self.daily_pnl += pnl
        logging.info(f"Daily P&L updated: Rs.{self.daily_pnl:,.2f}")
        
        # Check if we've hit daily loss limit
        if self.daily_pnl <= -self.max_daily_loss:
            logging.error("DAILY LOSS LIMIT REACHED! No new positions allowed.")
            return False
        
        return True
    
    def add_position(self, position_id, position_data):
        """Add new position to tracking"""
        self.open_positions[position_id] = {
            **position_data,
            'open_time': datetime.now()
        }
        logging.info(f"Position added: {position_id}")
    
    def remove_position(self, position_id):
        """Remove closed position"""
        if position_id in self.open_positions:
            del self.open_positions[position_id]
            logging.info(f"Position removed: {position_id}")
    
    def get_total_exposure(self):
        """Calculate total market exposure"""
        total_exposure = sum(
            pos.get('quantity', 0) * pos.get('premium', 0)
            for pos in self.open_positions.values()
        )
        return total_exposure
    
    def get_risk_metrics(self):
        """Get current risk metrics"""
        return {
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': (self.daily_pnl / self.capital) * 100,
            'open_positions': len(self.open_positions),
            'total_exposure': self.get_total_exposure(),
            'exposure_pct': (self.get_total_exposure() / self.capital) * 100,
            'can_trade': self.daily_pnl > -self.max_daily_loss,
            'remaining_loss_capacity': self.max_daily_loss + self.daily_pnl
        }
    
    def reset_daily_metrics(self):
        """Reset daily metrics (call at start of each day)"""
        self.daily_pnl = 0
        logging.info("Daily metrics reset for new trading day")
