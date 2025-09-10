"""
Nifty Options Trading System - GitHub Actions Version
Runs daily during market hours with automated reporting
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the trading system
from src.options_trader import OptionsTrader, StreamManager
from src.report_generator import ReportGenerator

class DailyTradingSession:
    """Manages a complete trading session with reporting"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.session_data = {
            'date': self.start_time.strftime('%Y-%m-%d'),
            'start_time': self.start_time.isoformat(),
            'signals': [],
            'trades': [],
            'errors': [],
            'market_data': []
        }
        self.setup_directories()
        
    def setup_directories(self):
        """Ensure all required directories exist"""
        dirs = ['logs', 'reports', 'data']
        for dir_name in dirs:
            Path(dir_name).mkdir(exist_ok=True)
    
    def run_session(self, duration_hours=6):
        """Run trading session for specified hours"""
        try:
            # Initialize components
            trader = OptionsTrader()
            stream_manager = StreamManager(trader)
            report_generator = ReportGenerator()
            
            # Set up session monitoring
            trader.on_signal = self.record_signal
            trader.on_trade_close = self.record_trade
            trader.on_error = self.record_error
            
            logging.info(f"Starting trading session for {duration_hours} hours")
            
            # Connect to market
            stream_manager.connect()
            
            # Run for specified duration or until market close
            end_time = datetime.now() + timedelta(hours=duration_hours)
            market_close = datetime.now().replace(hour=15, minute=30, second=0)
            actual_end = min(end_time, market_close)
            
            while datetime.now() < actual_end:
                time.sleep(60)  # Check every minute
                
                # Periodic status update
                if datetime.now().minute % 15 == 0:
                    self.log_status(trader)
            
            # Session complete
            self.finalize_session(trader, report_generator)
            
        except Exception as e:
            logging.error(f"Session error: {e}")
            self.record_error(str(e))
            self.emergency_shutdown()
    
    def record_signal(self, signal_data):
        """Record generated signals"""
        self.session_data['signals'].append({
            'timestamp': datetime.now().isoformat(),
            **signal_data
        })
        
    def record_trade(self, trade_data):
        """Record completed trades"""
        self.session_data['trades'].append({
            'timestamp': datetime.now().isoformat(),
            **trade_data
        })
        
    def record_error(self, error_msg):
        """Record errors"""
        self.session_data['errors'].append({
            'timestamp': datetime.now().isoformat(),
            'error': error_msg
        })
    
    def log_status(self, trader):
        """Log periodic status"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'active_positions': len(trader.active_positions),
            'daily_pnl': trader.risk_manager.daily_pnl,
            'signals_today': len(self.session_data['signals'])
        }
        logging.info(f"Status Update: {status}")
        
    def finalize_session(self, trader, report_generator):
        """Finalize session and generate reports"""
        # Add final data
        self.session_data['end_time'] = datetime.now().isoformat()
        self.session_data['duration'] = str(datetime.now() - self.start_time)
        self.session_data['final_pnl'] = trader.risk_manager.daily_pnl
        
        # Calculate statistics
        self.calculate_statistics()
        
        # Save session data
        session_file = f"data/session_{self.session_data['date']}.json"
        with open(session_file, 'w') as f:
            json.dump(self.session_data, f, indent=2)
        
        # Generate reports
        report_generator.generate_daily_report(self.session_data)
        report_generator.generate_trade_log(self.session_data)
        
        # Create summary for GitHub Actions
        self.create_github_summary()
        
    def calculate_statistics(self):
        """Calculate session statistics"""
        trades = self.session_data['trades']
        if trades:
            wins = [t for t in trades if t['pnl'] > 0]
            losses = [t for t in trades if t['pnl'] < 0]
            
            self.session_data['statistics'] = {
                'total_trades': len(trades),
                'wins': len(wins),
                'losses': len(losses),
                'win_rate': len(wins) / len(trades) * 100 if trades else 0,
                'total_pnl': sum(t['pnl'] for t in trades),
                'average_win': sum(t['pnl'] for t in wins) / len(wins) if wins else 0,
                'average_loss': sum(t['pnl'] for t in losses) / len(losses) if losses else 0,
                'largest_win': max((t['pnl'] for t in wins), default=0),
                'largest_loss': min((t['pnl'] for t in losses), default=0)
            }
        else:
            self.session_data['statistics'] = {
                'total_trades': 0,
                'message': 'No trades executed'
            }
    
    def create_github_summary(self):
        """Create summary for GitHub Actions"""
        summary = f"""# Trading Session Summary - {self.session_data['date']}

## Performance
- Total Signals: {len(self.session_data['signals'])}
- Total Trades: {self.session_data['statistics'].get('total_trades', 0)}
- Final P&L: ₹{self.session_data.get('final_pnl', 0):,.2f}
- Win Rate: {self.session_data['statistics'].get('win_rate', 0):.1f}%

## Session Details
- Start: {self.session_data['start_time']}
- End: {self.session_data.get('end_time', 'In Progress')}
- Duration: {self.session_data.get('duration', 'N/A')}
- Errors: {len(self.session_data['errors'])}

## Actions Taken
- Generated {len(self.session_data['signals'])} trading signals
- Executed {self.session_data['statistics'].get('total_trades', 0)} trades
- Saved session data and reports
"""
        
        # Write to GitHub Actions summary
        summary_file = os.environ.get('GITHUB_STEP_SUMMARY')
        if summary_file:
            with open(summary_file, 'w') as f:
                f.write(summary)
        
        # Also save locally
        with open('reports/latest_summary.md', 'w') as f:
            f.write(summary)
    
    def emergency_shutdown(self):
        """Emergency shutdown procedure"""
        logging.error("Emergency shutdown initiated")
        self.session_data['emergency_shutdown'] = True
        self.session_data['shutdown_time'] = datetime.now().isoformat()
        
        # Save whatever data we have
        emergency_file = f"data/emergency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(emergency_file, 'w') as f:
            json.dump(self.session_data, f, indent=2)

def main():
    """Main entry point for GitHub Actions"""
    # Setup logging
    log_file = f"logs/trading_{datetime.now().strftime('%Y%m%d')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logging.info("="*50)
    logging.info("NIFTY OPTIONS TRADING SYSTEM")
    logging.info(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    logging.info(f"Running in: {'GitHub Actions' if os.environ.get('GITHUB_ACTIONS') else 'Local'}")
    logging.info("="*50)
    
    # Check market hours
    current_hour = datetime.now().hour
    current_minute = datetime.now().minute
    
    # Allow some flexibility for GitHub Actions scheduling
    if not (9 <= current_hour <= 15):
        logging.info("Outside market hours. Skipping session.")
        sys.exit(0)
    
    # Check if today is a weekend
    if datetime.now().weekday() >= 5:  # Saturday = 5, Sunday = 6
        logging.info("Weekend. Markets closed.")
        sys.exit(0)
    
    # Run trading session
    session = DailyTradingSession()
    
    # Calculate remaining market hours
    market_close = datetime.now().replace(hour=15, minute=30, second=0)
    remaining_hours = (market_close - datetime.now()).seconds / 3600
    
    if remaining_hours > 0:
        session.run_session(duration_hours=remaining_hours)
    else:
        logging.info("Market already closed for the day.")
    
    logging.info("Session complete. Check reports directory for results.")

if __name__ == "__main__":
    main()
