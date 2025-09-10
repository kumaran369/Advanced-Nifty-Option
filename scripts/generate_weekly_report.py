#!/usr/bin/env python3
"""
Generate weekly consolidated report from daily trading sessions
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

def generate_weekly_report():
    """Generate weekly summary from daily session files"""
    
    # Get all session files from the past week
    data_dir = Path("artifacts/session-data-*")
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    
    # Calculate date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=7)
    
    all_trades = []
    all_signals = []
    daily_pnl = {}
    
    # Process each day's data
    for data_file in Path(".").glob("artifacts/*/data/session_*.json"):
        try:
            with open(data_file, 'r') as f:
                session_data = json.load(f)
            
            date = session_data['date']
            
            # Collect trades
            for trade in session_data.get('trades', []):
                trade['date'] = date
                all_trades.append(trade)
            
            # Collect signals
            for signal in session_data.get('signals', []):
                signal['date'] = date
                all_signals.append(signal)
            
            # Daily P&L
            daily_pnl[date] = session_data.get('final_pnl', 0)
            
        except Exception as e:
            print(f"Error processing {data_file}: {e}")
    
    # Create summary statistics
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        
        summary_stats = {
            'week_ending': end_date.isoformat(),
            'total_trades': len(trades_df),
            'total_signals': len(all_signals),
            'total_pnl': sum(daily_pnl.values()),
            'winning_trades': len(trades_df[trades_df['pnl'] > 0]),
            'losing_trades': len(trades_df[trades_df['pnl'] < 0]),
            'win_rate': len(trades_df[trades_df['pnl'] > 0]) / len(trades_df) * 100,
            'best_day': max(daily_pnl.items(), key=lambda x: x[1]) if daily_pnl else ('N/A', 0),
            'worst_day': min(daily_pnl.items(), key=lambda x: x[1]) if daily_pnl else ('N/A', 0),
            'average_daily_pnl': sum(daily_pnl.values()) / len(daily_pnl) if daily_pnl else 0
        }
        
        # Generate charts
        generate_weekly_charts(trades_df, daily_pnl, report_dir)
        
        # Create HTML report
        create_weekly_html_report(summary_stats, trades_df, daily_pnl, report_dir)
        
        # Save summary as JSON
        with open(report_dir / 'weekly_summary.json', 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"Weekly report generated: {report_dir / 'weekly_report.html'}")
        
    else:
        print("No trades found for the week")

def generate_weekly_charts(trades_df, daily_pnl, report_dir):
    """Generate weekly charts"""
    
    # Daily P&L bar chart
    plt.figure(figsize=(10, 6))
    dates = list(daily_pnl.keys())
    pnl_values = list(daily_pnl.values())
    colors = ['green' if x >= 0 else 'red' for x in pnl_values]
    
    plt.bar(dates, pnl_values, color=colors, alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Daily P&L - Week Summary')
    plt.xlabel('Date')
    plt.ylabel('P&L (₹)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(report_dir / 'weekly_daily_pnl.png', dpi=150)
    plt.close()
    
    # Cumulative P&L line chart
    plt.figure(figsize=(10, 6))
    cumulative = []
    current = 0
    for date in sorted(daily_pnl.keys()):
        current += daily_pnl[date]
        cumulative.append(current)
    
    plt.plot(sorted(daily_pnl.keys()), cumulative, marker='o', linewidth=2)
    plt.fill_between(sorted(daily_pnl.keys()), 0, cumulative, alpha=0.3)
    plt.title('Cumulative P&L - Week')
    plt.xlabel('Date')
    plt.ylabel('Cumulative P&L (₹)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(report_dir / 'weekly_cumulative_pnl.png', dpi=150)
    plt.close()

def create_weekly_html_report(stats, trades_df, daily_pnl, report_dir):
    """Create weekly HTML report"""
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Weekly Trading Report - {stats['week_ending']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: auto; background: white; padding: 20px; border-radius: 10px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #007bff; }}
        .metric-value {{ font-size: 24px; font-weight: bold; }}
        .positive {{ color: #28a745; }}
        .negative {{ color: #dc3545; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Weekly Trading Report</h1>
        <p>Week Ending: {stats['week_ending']}</p>
        
        <div class="metric-grid">
            <div class="metric-card">
                <div>Total P&L</div>
                <div class="metric-value {'positive' if stats['total_pnl'] >= 0 else 'negative'}">
                    ₹{stats['total_pnl']:,.2f}
                </div>
            </div>
            <div class="metric-card">
                <div>Total Trades</div>
                <div class="metric-value">{stats['total_trades']}</div>
            </div>
            <div class="metric-card">
                <div>Win Rate</div>
                <div class="metric-value">{stats['win_rate']:.1f}%</div>
            </div>
            <div class="metric-card">
                <div>Avg Daily P&L</div>
                <div class="metric-value {'positive' if stats['average_daily_pnl'] >= 0 else 'negative'}">
                    ₹{stats['average_daily_pnl']:,.2f}
                </div>
            </div>
        </div>
        
        <h2>Daily P&L Summary</h2>
        <img src="weekly_daily_pnl.png" alt="Daily P&L Chart">
        
        <h2>Cumulative P&L</h2>
        <img src="weekly_cumulative_pnl.png" alt="Cumulative P&L Chart">
        
        <h2>Best and Worst Days</h2>
        <p>Best Day: {stats['best_day'][0]} - ₹{stats['best_day'][1]:,.2f}</p>
        <p>Worst Day: {stats['worst_day'][0]} - ₹{stats['worst_day'][1]:,.2f}</p>
    </div>
</body>
</html>
    """
    
    with open(report_dir / 'weekly_report.html', 'w') as f:
        f.write(html_content)

if __name__ == "__main__":
    generate_weekly_report()
