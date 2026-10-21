import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# 1. Harvest: Download 2 years of Apple data
# interval='1d' means one row per day
df = yf.download("AAPL", period="2y", interval="1d")

# 2. Flatten: Fix the column headers for Spyder's Variable Explorer
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# 3. Clean: Remove timezones so the spreadsheet viewer doesn't crash
df.index = df.index.tz_localize(None)

import numpy as np

# --- STEP 1: MATH (Add this before plotting) ---
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss.replace(0, np.nan)
df['RSI'] = 100 - (100 / (1 + rs))

# Calculate Moving Averages
df['SMA50'] = df['Close'].rolling(window=50).mean()
df['SMA200'] = df['Close'].rolling(window=200).mean()

# This creates 2 rows of charts that share the same dates (x-axis)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, 
                               gridspec_kw={'height_ratios': [3, 1]})

# Top Chart: Trends
ax1.plot(df['Close'], label='Price', color='black', alpha=0.3)
ax1.plot(df['SMA50'], label='50-Day (Fast)', color='green')
ax1.plot(df['SMA200'], label='200-Day (Slow)', color='red')
ax1.set_title('AAPL Technical Dashboard')
ax1.legend(loc='upper left')

# Bottom Chart: Momentum (RSI)
ax2.plot(df['RSI'], color='purple', label='RSI')
ax2.axhline(70, color='red', linestyle='--')   # Overbought line
ax2.axhline(30, color='green', linestyle='--') # Oversold line
ax2.set_ylim(0, 100)
ax2.set_ylabel('RSI')

plt.tight_layout()
plt.show()