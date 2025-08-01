import pandas as pd
import numpy as np
import ta
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv('config/config.env')

class TradingStrategy:
    def __init__(self, rsi_period=14, ma_short=10, ma_long=20, 
                 rsi_oversold=30, rsi_overbought=70):
        self.rsi_period = int(os.getenv('RSI_PERIOD', rsi_period))
        self.ma_short = int(os.getenv('MA_SHORT_PERIOD', ma_short))
        self.ma_long = int(os.getenv('MA_LONG_PERIOD', ma_long))
        self.rsi_oversold = int(os.getenv('RSI_OVERSOLD', rsi_oversold))
        self.rsi_overbought = int(os.getenv('RSI_OVERBOUGHT', rsi_overbought))
        
    def calculate_indicators(self, df):
        """Calculate RSI and Moving Averages"""
        df = df.copy()
        
        # Calculate RSI
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=self.rsi_period).rsi()
        
        # Calculate Moving Averages
        df['MA_Short'] = df['Close'].rolling(window=self.ma_short).mean()
        df['MA_Long'] = df['Close'].rolling(window=self.ma_long).mean()
        
        # Calculate additional indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Price_Change'] = df['Close'].pct_change()
        
        return df
    
    def generate_signals(self, df):
        """Generate buy/sell signals based on RSI + MA crossover strategy"""
        df = self.calculate_indicators(df)
        
        # Initialize signal columns
        df['Signal'] = 0
        df['Position'] = 0
        df['Trade_Type'] = ''
        
        for i in range(1, len(df)):
            # Buy Signal: RSI oversold + MA short crosses above MA long
            if (df['RSI'].iloc[i] < self.rsi_oversold and 
                df['MA_Short'].iloc[i] > df['MA_Long'].iloc[i] and
                df['MA_Short'].iloc[i-1] <= df['MA_Long'].iloc[i-1]):
                df.loc[df.index[i], 'Signal'] = 1
                df.loc[df.index[i], 'Trade_Type'] = 'BUY'
                
            # Sell Signal: RSI overbought + MA short crosses below MA long
            elif (df['RSI'].iloc[i] > self.rsi_overbought and 
                  df['MA_Short'].iloc[i] < df['MA_Long'].iloc[i] and
                  df['MA_Short'].iloc[i-1] >= df['MA_Long'].iloc[i-1]):
                df.loc[df.index[i], 'Signal'] = -1
                df.loc[df.index[i], 'Trade_Type'] = 'SELL'
        
        # Calculate positions
        df['Position'] = df['Signal'].replace(0, pd.NA).ffill().fillna(0)
        
        return df
    
    def backtest_strategy(self, df, initial_capital=100000):
        """Backtest the trading strategy"""
        df = self.generate_signals(df)
        
        # Initialize portfolio
        portfolio = pd.DataFrame(index=df.index)
        portfolio['Price'] = df['Close']
        portfolio['Signal'] = df['Signal']
        portfolio['Position'] = df['Position']
        
        # Calculate returns
        portfolio['Market_Return'] = df['Close'].pct_change()
        portfolio['Strategy_Return'] = portfolio['Position'].shift(1) * portfolio['Market_Return']
        
        # Calculate cumulative returns
        portfolio['Cumulative_Market_Return'] = (1 + portfolio['Market_Return']).cumprod()
        portfolio['Cumulative_Strategy_Return'] = (1 + portfolio['Strategy_Return']).cumprod()
        
        # Calculate portfolio value
        portfolio['Portfolio_Value'] = initial_capital * portfolio['Cumulative_Strategy_Return']
        
        # Calculate trade statistics
        trades = df[df['Signal'] != 0].copy()
        
        # Performance metrics
        total_return = portfolio['Cumulative_Strategy_Return'].iloc[-1] - 1
        market_return = portfolio['Cumulative_Market_Return'].iloc[-1] - 1
        
        # Calculate Sharpe ratio (assuming 252 trading days)
        strategy_std = portfolio['Strategy_Return'].std() * np.sqrt(252)
        sharpe_ratio = (total_return / strategy_std) if strategy_std != 0 else 0
        
        # Win rate calculation
        winning_trades = len(trades[trades['Price_Change'] > 0])
        total_trades = len(trades)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        results = {
            'total_return': total_return,
            'market_return': market_return,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'final_portfolio_value': portfolio['Portfolio_Value'].iloc[-1],
            'portfolio': portfolio,
            'trades': trades
        }
        
        return results