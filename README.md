# Nifty Options Trading System

Automated options trading system for NIFTY 50 index with GitHub Actions integration for daily trading sessions.

## Features

- **Automated Daily Trading**: Runs Monday-Friday during market hours
- **Black-Scholes Options Pricing**: Accurate premium calculations
- **Dynamic Risk Management**: Volatility-based stops and position sizing
- **Technical Indicators**: VWAP, RSI, SuperTrend
- **Real-time P&L Tracking**: With Greeks calculation
- **Comprehensive Reporting**: Daily reports and trade logs
- **GitHub Actions Integration**: Fully automated deployment

## Architecture

```
nifty-options-trader/
├── .github/
│   └── workflows/
│       └── daily-trading.yml    # GitHub Actions workflow
├── src/
│   ├── __init__.py
│   ├── main.py                  # Entry point for GitHub Actions
│   ├── options_trader.py        # Core trading logic
│   ├── indicators.py            # Technical indicators
│   ├── risk_manager.py          # Risk management
│   └── report_generator.py      # Report generation
├── scripts/
│   ├── generate_weekly_report.py
│   └── check_token_expiry.py
├── config/
│   └── settings.py              # Configuration
├── logs/                        # Trading logs (generated)
├── reports/                     # Trading reports (generated)
├── data/                        # Session data (generated)
├── requirements.txt
└── README.md
```

## Setup Instructions

### 1. Fork/Clone Repository

```bash
git clone https://github.com/yourusername/nifty-options-trader.git
cd nifty-options-trader
```

### 2. Configure GitHub Secrets

Go to Settings → Secrets and variables → Actions, and add:

- `UPSTOX_API_KEY`: Your Upstox API key
- `UPSTOX_ACCESS_TOKEN`: Your Upstox access token

### 3. Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export UPSTOX_API_KEY="your-api-key"
export UPSTOX_ACCESS_TOKEN="your-access-token"

# Run locally
python src/main.py
```

### 4. Enable GitHub Actions

The workflow will automatically run:
- Monday to Friday at 9:20 AM IST
- Can be manually triggered from Actions tab

## Configuration

Edit `config/settings.py` to customize:

```python
# Trading parameters
MAX_SIGNALS_PER_DAY = 5
MAX_RISK_PER_TRADE = 0.02  # 2% risk per trade
DELTA_THRESHOLD = 0.4      # Minimum delta for entry

# Technical indicators
RSI_PERIOD = 14
VWAP_WINDOW = 300
SUPER_ATR_PERIOD = 10
```

## Reports Generated

### Daily Reports
- Trading summary
- Signal analysis
- P&L breakdown
- Risk metrics

### Trade Logs
- Entry/exit details
- Greeks at entry/exit
- P&L calculation
- Duration

### Session Data
- Raw tick data
- All signals generated
- Error logs
- Market status

## Monitoring

### GitHub Actions Summary
Each run creates a summary with:
- Total signals and trades
- Final P&L
- Win rate
- Session duration

### Artifacts
All logs, reports, and data are saved as artifacts:
- Logs: 30-day retention
- Reports: 90-day retention
- Data: 90-day retention

## Token Management

### Automatic Expiry Check
- Daily check at 8 AM IST
- Creates issue if token expires within 24 hours

### Manual Token Update
1. Generate new token from Upstox
2. Update `UPSTOX_ACCESS_TOKEN` secret
3. Workflow will use new token automatically

## Safety Features

- **Market Hours Check**: Only runs during market hours
- **Weekend Skip**: Automatically skips weekends
- **Error Handling**: Comprehensive error catching
- **Emergency Shutdown**: Saves data on critical errors
- **Position Limits**: Maximum concurrent positions
- **Daily Loss Limit**: 6% of capital

## Customization

### Add Holiday Calendar
Edit `.github/workflows/daily-trading.yml`:

```yaml
- name: Check holidays
  run: |
    # Add holiday checking logic
    python scripts/check_holiday.py
```

### Notifications
Add notification step in workflow:

```yaml
- name: Send notification
  env:
    WEBHOOK_URL: ${{ secrets.DISCORD_WEBHOOK }}
  run: |
    python scripts/send_notification.py
```

### Strategy Modifications
Edit `src/indicators.py` to modify:
- Signal generation logic
- Technical indicators
- Entry/exit conditions

## Troubleshooting

### Common Issues

1. **Token Expired**
   - Generate new token
   - Update GitHub secret

2. **No Data Received**
   - Check market hours
   - Verify credentials
   - Check Upstox API status

3. **Workflow Timeout**
   - Normal for 6+ hour sessions
   - Check artifacts for partial data

### Logs Location
- GitHub: Actions → Select run → Artifacts
- Local: `logs/trading_YYYYMMDD.log`

## Disclaimer

This is for educational purposes only. Trading involves risk of loss. Always test thoroughly with paper trading before using real money.

## License

MIT License - See LICENSE file for details

## Contributing

1. Fork the repository
2. Create feature branch
3. Test thoroughly
4. Submit pull request

## Support

- Issues: GitHub Issues
- Documentation: Wiki
- Discussion: Discussions tab

---

**Note**: Remember to update your access token regularly as Upstox tokens expire.
