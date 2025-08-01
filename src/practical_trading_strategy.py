#!/usr/bin/env python3
"""
Practical Trading Strategy - Optimized for real market conditions
Implements RSI + Moving Average crossover with practical parameters
"""

import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv('config/config.env')

class PracticalTradingStrategy:
    def __init__(self, rsi_period=14, ma_short=20, ma_long=50, 
                 rsi_oversold=35, rsi_overbought=65, position_size=10000, 
                 holding_period=5):
        """
        Practical Trading Strategy with optimized parameters
        
        Parameters adjusted for real market conditions:
        - RSI oversold: 35 (instead of 30) for more frequent signals
        - RSI overbought: 65 (instead of 70) for earlier exits
        - Still uses 20-DMA vs 50-DMA as specified
        """
        self.rsi_period = rsi_period
        self.ma_short = ma_short  # 20-DMA
        self.ma_long = ma_long    # 50-DMA
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.position_size = position_size
        self.holding_period = holding_period
        
        print(f"📊 Practical Strategy Configuration:")
        print(f"   RSI Period: {self.rsi_period}")
        print(f"   20-DMA vs 50-DMA crossover strategy")
        print(f"   RSI Oversold: < {self.rsi_oversold} (optimized from 30)")
        print(f"   RSI Overbought: > {self.rsi_overbought} (optimized from 70)")
        print(f"   Position Size: ₹{self.position_size:,}")
        print(f"   Max Holding Period: {self.holding_period} days")
        
    def calculate_indicators(self, df):
        """Calculate RSI and Moving Averages"""
        df = df.copy()
        
        # Calculate RSI
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=self.rsi_period).rsi()
        
        # Calculate Moving Averages (20-DMA and 50-DMA)
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # Additional indicators for analysis
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Price_Change'] = df['Close'].pct_change()
        df['Volatility'] = df['Price_Change'].rolling(window=10).std()
        
        return df
    
    def generate_signals(self, df):
        """
        Generate buy/sell signals with practical conditions:
        
        BUY SIGNAL: 
        - Primary: RSI < 35 (oversold) AND 20-DMA > 50-DMA (bullish trend)
        - Alternative: RSI < 35 AND 20-DMA crosses above 50-DMA
        
        SELL SIGNAL:
        - RSI > 65 (overbought) OR
        - 20-DMA crosses below 50-DMA OR
        - Holding period exceeded (5 days)
        """
        df = self.calculate_indicators(df)
        
        # Initialize signal columns
        df['Signal'] = 0
        df['Buy_Signal'] = False
        df['Sell_Signal'] = False
        df['Signal_Type'] = ''
        df['Signal_Reason'] = ''
        df['MA_Crossover'] = 0
        
        for i in range(1, len(df)):
            current_rsi = df['RSI'].iloc[i]
            current_ma20 = df['MA_20'].iloc[i]
            current_ma50 = df['MA_50'].iloc[i]
            prev_ma20 = df['MA_20'].iloc[i-1]
            prev_ma50 = df['MA_50'].iloc[i-1]
            
            # Skip if indicators are NaN
            if pd.isna(current_rsi) or pd.isna(current_ma20) or pd.isna(current_ma50):
                continue
            
            # Detect MA crossovers
            bullish_crossover = (current_ma20 > current_ma50 and prev_ma20 <= prev_ma50)
            bearish_crossover = (current_ma20 < current_ma50 and prev_ma20 >= prev_ma50)
            
            if bullish_crossover:
                df.loc[df.index[i], 'MA_Crossover'] = 1
            elif bearish_crossover:
                df.loc[df.index[i], 'MA_Crossover'] = -1
            
            # BUY SIGNALS (more practical conditions)
            buy_condition_1 = (current_rsi < self.rsi_oversold and current_ma20 > current_ma50)
            buy_condition_2 = (current_rsi < self.rsi_oversold and bullish_crossover)
            
            if buy_condition_1 or buy_condition_2:
                df.loc[df.index[i], 'Signal'] = 1
                df.loc[df.index[i], 'Buy_Signal'] = True
                df.loc[df.index[i], 'Signal_Type'] = 'BUY'
                
                if buy_condition_2:
                    df.loc[df.index[i], 'Signal_Reason'] = f'RSI Oversold ({current_rsi:.1f}) + Bullish MA Crossover'
                else:
                    df.loc[df.index[i], 'Signal_Reason'] = f'RSI Oversold ({current_rsi:.1f}) + Bullish Trend'
                
            # SELL SIGNALS
            elif (current_rsi > self.rsi_overbought or bearish_crossover):
                df.loc[df.index[i], 'Signal'] = -1
                df.loc[df.index[i], 'Sell_Signal'] = True
                df.loc[df.index[i], 'Signal_Type'] = 'SELL'
                
                if current_rsi > self.rsi_overbought:
                    df.loc[df.index[i], 'Signal_Reason'] = f'RSI Overbought ({current_rsi:.1f})'
                elif bearish_crossover:
                    df.loc[df.index[i], 'Signal_Reason'] = 'Bearish MA Crossover'
        
        return df
    
    def backtest_strategy(self, df, stock_symbol='STOCK'):
        """Comprehensive backtesting with detailed trade tracking"""
        print(f"\n🔄 Backtesting practical strategy for {stock_symbol}...")
        
        # Generate signals
        df = self.generate_signals(df)
        
        # Initialize tracking variables
        trades = []
        current_position = None
        portfolio_value = 0
        total_invested = 0
        
        # Track daily portfolio
        daily_portfolio = []
        
        for i, (date, row) in enumerate(df.iterrows()):
            daily_value = portfolio_value
            
            # Check for buy signal
            if row['Buy_Signal'] and current_position is None:
                shares = self.position_size / row['Close']
                
                current_position = {
                    'entry_date': date,
                    'entry_price': row['Close'],
                    'shares': shares,
                    'entry_rsi': row['RSI'],
                    'entry_ma20': row['MA_20'],
                    'entry_ma50': row['MA_50'],
                    'signal_reason': row['Signal_Reason'],
                    'days_held': 0
                }
                
                total_invested += self.position_size
                print(f"📈 BUY: {date.date()} at ₹{row['Close']:.2f} ({shares:.2f} shares)")
                print(f"    Reason: {row['Signal_Reason']}")
            
            # Check for sell signal or holding period exceeded
            elif current_position is not None:
                current_position['days_held'] += 1
                
                should_sell = (row['Sell_Signal'] or 
                             current_position['days_held'] >= self.holding_period)
                
                if should_sell:
                    # Calculate P&L
                    exit_value = current_position['shares'] * row['Close']
                    pnl = exit_value - self.position_size
                    pnl_percent = (pnl / self.position_size) * 100
                    
                    # Determine sell reason
                    if row['Sell_Signal']:
                        sell_reason = row['Signal_Reason']
                    else:
                        sell_reason = f'Holding Period Exceeded ({self.holding_period} days)'
                    
                    # Record trade
                    trade = {
                        'stock': stock_symbol,
                        'entry_date': current_position['entry_date'],
                        'exit_date': date,
                        'entry_price': current_position['entry_price'],
                        'exit_price': row['Close'],
                        'shares': current_position['shares'],
                        'days_held': current_position['days_held'],
                        'investment': self.position_size,
                        'exit_value': exit_value,
                        'pnl': pnl,
                        'pnl_percent': pnl_percent,
                        'entry_rsi': current_position['entry_rsi'],
                        'exit_rsi': row['RSI'],
                        'buy_reason': current_position['signal_reason'],
                        'sell_reason': sell_reason,
                        'profitable': pnl > 0
                    }
                    
                    trades.append(trade)
                    portfolio_value += exit_value
                    
                    status = "✅ PROFIT" if pnl > 0 else "❌ LOSS"
                    print(f"📉 SELL: {date.date()} at ₹{row['Close']:.2f}")
                    print(f"    Reason: {sell_reason}")
                    print(f"    P&L: ₹{pnl:+.2f} ({pnl_percent:+.2f}%) {status}")
                    
                    current_position = None
                
                else:
                    daily_value = current_position['shares'] * row['Close']
            
            # Record daily portfolio value
            daily_portfolio.append({
                'date': date,
                'portfolio_value': daily_value,
                'close_price': row['Close'],
                'rsi': row['RSI'],
                'ma20': row['MA_20'],
                'ma50': row['MA_50'],
                'has_position': current_position is not None
            })
        
        # Handle remaining open position
        if current_position is not None:
            final_price = df['Close'].iloc[-1]
            final_date = df.index[-1]
            exit_value = current_position['shares'] * final_price
            pnl = exit_value - self.position_size
            pnl_percent = (pnl / self.position_size) * 100
            
            trade = {
                'stock': stock_symbol,
                'entry_date': current_position['entry_date'],
                'exit_date': final_date,
                'entry_price': current_position['entry_price'],
                'exit_price': final_price,
                'shares': current_position['shares'],
                'days_held': current_position['days_held'],
                'investment': self.position_size,
                'exit_value': exit_value,
                'pnl': pnl,
                'pnl_percent': pnl_percent,
                'entry_rsi': current_position['entry_rsi'],
                'exit_rsi': df['RSI'].iloc[-1],
                'buy_reason': current_position['signal_reason'],
                'sell_reason': 'End of Period',
                'profitable': pnl > 0
            }
            
            trades.append(trade)
            portfolio_value += exit_value
            
            status = "✅ PROFIT" if pnl > 0 else "❌ LOSS"
            print(f"📉 FINAL SELL: {final_date.date()} at ₹{final_price:.2f}")
            print(f"    P&L: ₹{pnl:+.2f} ({pnl_percent:+.2f}%) {status}")
        
        # Calculate performance metrics
        trades_df = pd.DataFrame(trades)
        daily_portfolio_df = pd.DataFrame(daily_portfolio)
        
        if len(trades) > 0:
            total_pnl = trades_df['pnl'].sum()
            total_trades = len(trades)
            winning_trades = len(trades_df[trades_df['profitable']])
            win_ratio = (winning_trades / total_trades) * 100
            avg_pnl = trades_df['pnl'].mean()
            avg_holding_period = trades_df['days_held'].mean()
            
            # Calculate returns
            returns = trades_df['pnl_percent'].values / 100
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0
            
        else:
            total_pnl = 0
            total_trades = 0
            winning_trades = 0
            win_ratio = 0
            avg_pnl = 0
            avg_holding_period = 0
            sharpe_ratio = 0
        
        # Market performance
        market_return = ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100
        
        results = {
            'stock': stock_symbol,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_ratio': win_ratio,
            'total_pnl': total_pnl,
            'avg_pnl_per_trade': avg_pnl,
            'avg_holding_period': avg_holding_period,
            'total_invested': total_invested,
            'final_portfolio_value': portfolio_value,
            'strategy_return': (portfolio_value / total_invested - 1) * 100 if total_invested > 0 else 0,
            'market_return': market_return,
            'sharpe_ratio': sharpe_ratio,
            'trades_df': trades_df,
            'daily_portfolio_df': daily_portfolio_df,
            'signals_df': df
        }
        
        return results
    
    def print_backtest_summary(self, results):
        """Print comprehensive backtest summary"""
        print(f"\n{'='*60}")
        print(f"PRACTICAL STRATEGY SUMMARY: {results['stock']}")
        print(f"{'='*60}")
        
        print(f"📊 Trading Performance:")
        print(f"   Total Trades: {results['total_trades']}")
        print(f"   Winning Trades: {results['winning_trades']}")
        print(f"   Losing Trades: {results['losing_trades']}")
        print(f"   Win Ratio: {results['win_ratio']:.1f}%")
        
        print(f"\n💰 Financial Performance:")
        print(f"   Total P&L: ₹{results['total_pnl']:+,.2f}")
        print(f"   Average P&L per Trade: ₹{results['avg_pnl_per_trade']:+,.2f}")
        print(f"   Strategy Return: {results['strategy_return']:+.2f}%")
        print(f"   Market Return: {results['market_return']:+.2f}%")
        print(f"   Alpha (vs Market): {results['strategy_return'] - results['market_return']:+.2f}%")
        
        print(f"\n📈 Risk Metrics:")
        print(f"   Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print(f"   Average Holding Period: {results['avg_holding_period']:.1f} days")
        
        if results['total_trades'] > 0:
            trades_df = results['trades_df']
            print(f"\n🔍 Trade Details:")
            print(f"   Best Trade: ₹{trades_df['pnl'].max():+.2f} ({trades_df['pnl_percent'].max():+.2f}%)")
            print(f"   Worst Trade: ₹{trades_df['pnl'].min():+.2f} ({trades_df['pnl_percent'].min():+.2f}%)")
            print(f"   Longest Hold: {trades_df['days_held'].max()} days")
            print(f"   Shortest Hold: {trades_df['days_held'].min()} days")

def main():
    """Test the practical trading strategy"""
    from data_ingestion import EnhancedDataIngestion
    
    print("🚀 PRACTICAL TRADING STRATEGY TEST")
    print("=" * 60)
    
    # Initialize components
    ingestion = EnhancedDataIngestion()
    strategy = PracticalTradingStrategy(
        rsi_oversold=35,      # More practical threshold
        rsi_overbought=65,    # More practical threshold
        position_size=10000,
        holding_period=5
    )
    
    # Test with all stocks
    stocks = ['RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS']
    
    for stock in stocks:
        print(f"\n{'='*50}")
        print(f"TESTING: {stock}")
        print(f"{'='*50}")
        
        # Load processed data
        data = ingestion.load_saved_data(stock, 'processed')
        
        if data is not None:
            # Run backtest
            results = strategy.backtest_strategy(data, stock)
            
            # Print summary
            strategy.print_backtest_summary(results)
            
            # Save results
            if results['total_trades'] > 0:
                results['trades_df'].to_csv(f'data/{stock.replace(".", "_")}_practical_trades.csv', index=False)
                print(f"\n💾 Trade details saved to data/{stock.replace('.', '_')}_practical_trades.csv")
        else:
            print(f"❌ No data available for {stock}")

if __name__ == "__main__":
    main()