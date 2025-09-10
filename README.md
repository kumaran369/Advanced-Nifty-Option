# Nifty Options Trading Bot - Simple Version

Automated options trading system for NIFTY 50 index.

## Setup

1. Add your access token to GitHub Secrets:
   - Name: `UPSTOX_ACCESS_TOKEN`
   - Value: Your Upstox access token

2. For local testing:
   - Create `token.txt` file
   - Paste your access token in it
   - Run: `python main.py`

## How it works

- Runs automatically Monday-Friday at 9:20 AM IST
- Monitors NIFTY 50 for trading signals
- Uses VWAP, RSI, and SuperTrend indicators
- Calculates option prices using Black-Scholes
- Logs all signals and trades

## Manual Run

Go to Actions tab → Daily Trading → Run workflow

## Logs

Check Actions tab → Select run → Download logs artifact

## Token Management

When your token expires:
1. Get new token from Upstox
2. Update GitHub Secret: `UPSTOX_ACCESS_TOKEN`
3. For local: Update `token.txt`

That's it! Simple and automated.
