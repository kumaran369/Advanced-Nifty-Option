"""
Report Generator for Trading Sessions
Creates HTML and PDF reports with charts
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import logging
from jinja2 import Template

class ReportGenerator:
    """Generate trading reports and analytics"""
    
    def __init__(self):
        self.report_dir = Path("reports")
        self.report_dir.mkdir(exist_ok=True)
        sns.set_style("whitegrid")
        
    def generate_daily_report(self, session_data):
        """Generate comprehensive daily report"""
        try:
            date = session_data['date']
            
            # Create HTML report
            html_content = self._create_html_report(session_data)
            html_file = self.report_dir / f"daily_report_{date}.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Create charts
            if session_data.get('trades'):
                self._create_pnl_chart(session_data)
                self._create_trade_distribution(session_data)
            
            logging.info(f"Daily report generated: {html_file}")
            
        except Exception as e:
            logging.error(f"Error generating daily report: {e}")
    
    def generate_trade_log(self, session_data):
        """Generate detailed trade log"""
        try:
            date = session_data['date']
            trades = session_data.get('trades', [])
            
            if not trades:
                logging.info("No trades to log")
                return
            
            # Convert to DataFrame
            df = pd.DataFrame(trades)
            
            # Add additional calculations
            df['duration_minutes'] = pd.to_timedelta(df['duration']).dt.total_seconds() / 60
            df['pnl_per_lot'] = df['pnl'] / (df['quantity'] / 75)
            
            # Save as CSV
            csv_file = self.report_dir / f"trade_log_{date}.csv"
            df.to_csv(csv_file, index=False)
            
            # Create detailed HTML trade log
            html = self._create_trade_log_html(df, session_data)
            html_file = self.report_dir / f"trade_log_{date}.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html)
            
            logging.info(f"Trade log generated: {csv_file}")
            
        except Exception as e:
            logging.error(f"Error generating trade log: {e}")
    
    def _create_html_report(self, session_data):
        """Create HTML report using template"""
        template = Template("""
<!DOCTYPE html>
<html>
<head>
    <title>Trading Report - {{ date }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1, h2 {
            color: #333;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #007bff;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
        .metric-label {
            color: #666;
            font-size: 14px;
        }
        .positive {
            color: #28a745;
        }
        .negative {
            color: #dc3545;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f8f9fa;
            font-weight: bold;
        }
        .chart-container {
            margin: 20px 0;
            text-align: center;
        }
        .error-section {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Trading Report - {{ date }}</h1>
        <p>Generated: {{ generated_at }}</p>
        
        <h2>Summary</h2>
        <div class="summary-grid">
            <div class="metric-card">
                <div class="metric-label">Total P&L</div>
                <div class="metric-value {% if stats.total_pnl >= 0 %}positive{% else %}negative{% endif %}">
                    ₹{{ "{:,.2f}".format(stats.total_pnl) }}
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Trades</div>
                <div class="metric-value">{{ stats.total_trades }}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value">{{ "{:.1f}".format(stats.win_rate) }}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Signals</div>
                <div class="metric-value">{{ total_signals }}</div>
            </div>
        </div>
        
        {% if trades %}
        <h2>Trade Summary</h2>
        <table>
            <tr>
                <th>Time</th>
                <th>Type</th>
                <th>Strike</th>
                <th>Entry</th>
                <th>Exit</th>
                <th>Qty</th>
                <th>P&L</th>
                <th>Duration</th>
            </tr>
            {% for trade in trades %}
            <tr>
                <td>{{ trade.entry_time }}</td>
                <td>{{ trade.type }}</td>
                <td>{{ trade.strike }}</td>
                <td>₹{{ "{:.2f}".format(trade.entry_price) }}</td>
                <td>₹{{ "{:.2f}".format(trade.exit_price) }}</td>
                <td>{{ trade.quantity }}</td>
                <td class="{% if trade.pnl >= 0 %}positive{% else %}negative{% endif %}">
                    ₹{{ "{:,.2f}".format(trade.pnl) }}
                </td>
                <td>{{ trade.duration }}</td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}
        
        {% if errors %}
        <div class="error-section">
            <h3>Errors ({{ errors|length }})</h3>
            <ul>
                {% for error in errors %}
                <li>{{ error.timestamp }} - {{ error.error }}</li>
                {% endfor %}
            </ul>
        </div>
        {% endif %}
        
        <h2>Session Details</h2>
        <p>Start Time: {{ start_time }}</p>
        <p>End Time: {{ end_time }}</p>
        <p>Duration: {{ duration }}</p>
        
        <div class="chart-container">
            <img src="pnl_chart_{{ date }}.png" alt="P&L Chart">
        </div>
    </div>
</body>
</html>
        """)
        
        return template.render(
            date=session_data['date'],
            generated_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            stats=session_data.get('statistics', {}),
            total_signals=len(session_data.get('signals', [])),
            trades=session_data.get('trades', []),
            errors=session_data.get('errors', []),
            start_time=session_data.get('start_time', 'N/A'),
            end_time=session_data.get('end_time', 'N/A'),
            duration=session_data.get('duration', 'N/A')
        )
    
    def _create_trade_log_html(self, df, session_data):
        """Create detailed trade log HTML"""
        template = Template("""
<!DOCTYPE html>
<html>
<head>
    <title>Trade Log - {{ date }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #4CAF50;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .positive {
            color: green;
            font-weight: bold;
        }
        .negative {
            color: red;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <h1>Detailed Trade Log - {{ date }}</h1>
    {{ table_html }}
</body>
</html>
        """)
        
        # Convert DataFrame to HTML with custom formatting
        html_table = df.to_html(classes='trade-table', index=False, escape=False)
        
        return template.render(
            date=session_data['date'],
            table_html=html_table
        )
    
    def _create_pnl_chart(self, session_data):
        """Create P&L progression chart"""
        trades = session_data['trades']
        date = session_data['date']
        
        # Calculate cumulative P&L
        cumulative_pnl = []
        current_pnl = 0
        timestamps = []
        
        for trade in trades:
            current_pnl += trade['pnl']
            cumulative_pnl.append(current_pnl)
            timestamps.append(pd.to_datetime(trade['timestamp']))
        
        # Create chart
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, cumulative_pnl, marker='o', linewidth=2)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.title(f'Cumulative P&L - {date}')
        plt.xlabel('Time')
        plt.ylabel('P&L (₹)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Add fill
        plt.fill_between(timestamps, 0, cumulative_pnl, 
                        where=[p >= 0 for p in cumulative_pnl], 
                        color='green', alpha=0.3)
        plt.fill_between(timestamps, 0, cumulative_pnl, 
                        where=[p < 0 for p in cumulative_pnl], 
                        color='red', alpha=0.3)
        
        # Save
        chart_file = self.report_dir / f"pnl_chart_{date}.png"
        plt.savefig(chart_file, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_trade_distribution(self, session_data):
        """Create trade distribution chart"""
        trades = session_data['trades']
        date = session_data['date']
        
        # Extract P&L values
        pnl_values = [trade['pnl'] for trade in trades]
        
        # Create histogram
        plt.figure(figsize=(10, 6))
        plt.hist(pnl_values, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        plt.title(f'P&L Distribution - {date}')
        plt.xlabel('P&L (₹)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        mean_pnl = sum(pnl_values) / len(pnl_values)
        plt.axvline(x=mean_pnl, color='g', linestyle='-', alpha=0.7, label=f'Mean: ₹{mean_pnl:.2f}')
        plt.legend()
        
        plt.tight_layout()
        
        # Save
        chart_file = self.report_dir / f"trade_distribution_{date}.png"
        plt.savefig(chart_file, dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_weekly_summary(self, week_data):
        """Generate weekly summary report"""
        # Implementation for weekly summary
        pass
