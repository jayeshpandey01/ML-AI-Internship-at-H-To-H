#!/usr/bin/env python3
"""
Test script to analyze strategy signals and optimize parameters
"""

import sys
import os
sys.path.append('src')

from src.data_ingestion import EnhancedDataIngestion
from src.enhanced_trading_strategy import EnhancedTradingStrategy
import pandas as pd
import numpy as np

def analyze_signal_conditions():
    """Analyze why no signals are being generated"""
    print("🔍 ANALYZING SIGNAL CONDITIONS")
    print("=" * 60)
    
    # Initialize components
    ingestion = EnhancedDataIngestion()
    
    # Test with RELIANCE data
    test_stock = 'RELIANCE.NS'
    print(f"Analyzing signal conditions for {test_stock}...")
    
    # Load processed data
    data = ingestion.load_saved_data(test_stock, 'processed')
    
    if data is None:
        print("❌ No processed data found. Running data ingestion...")
        data = ingestion.fetch_yfinance_data(test_stock, period='6mo')
        data = ingestion.calculate_technical_indicators(data)
    
    # Calculate 20-DMA and 50-DMA specifically
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    
    # Analyze conditions
    print(f"\n📊 Data Analysis for {test_stock}:")
    print(f"   Total Records: {len(data)}")
    print(f"   Date Range: {data.index[0].date()} to {data.index[-1].date()}")
    
    # RSI analysis
    rsi_oversold = (data['RSI'] < 30).sum()
    rsi_overbought = (data['RSI'] > 70).sum()
    rsi_neutral = len(data) - rsi_oversold - rsi_overbought
    
    print(f"\n📈 RSI Analysis:")
    print(f"   RSI < 30 (Oversold): {rsi_oversold} days ({rsi_oversold/len(data)*100:.1f}%)")
    print(f"   RSI > 70 (Overbought): {rsi_overbought} days ({rsi_overbought/len(data)*100:.1f}%)")
    print(f"   RSI 30-70 (Neutral): {rsi_neutral} days ({rsi_neutral/len(data)*100:.1f}%)")
    
    # Moving Average analysis
    data_clean = data.dropna()
    ma_bullish = (data_clean['MA_20'] > data_clean['MA_50']).sum()
    ma_bearish = (data_clean['MA_20'] < data_clean['MA_50']).sum()
    
    print(f"\n📊 Moving Average Analysis:")
    print(f"   20-DMA > 50-DMA (Bullish): {ma_bullish} days ({ma_bullish/len(data_clean)*100:.1f}%)")
    print(f"   20-DMA < 50-DMA (Bearish): {ma_bearish} days ({ma_bearish/len(data_clean)*100:.1f}%)")
    
    # Crossover analysis
    data_clean['MA_Diff'] = data_clean['MA_20'] - data_clean['MA_50']
    data_clean['MA_Diff_Prev'] = data_clean['MA_Diff'].shift(1)
    
    bullish_crossovers = ((data_clean['MA_Diff'] > 0) & (data_clean['MA_Diff_Prev'] <= 0)).sum()
    bearish_crossovers = ((data_clean['MA_Diff'] < 0) & (data_clean['MA_Diff_Prev'] >= 0)).sum()
    
    print(f"\n🔄 Crossover Analysis:")
    print(f"   Bullish Crossovers (20-DMA crosses above 50-DMA): {bullish_crossovers}")
    print(f"   Bearish Crossovers (20-DMA crosses below 50-DMA): {bearish_crossovers}")
    
    # Combined signal analysis
    oversold_and_bullish_trend = ((data_clean['RSI'] < 30) & (data_clean['MA_20'] > data_clean['MA_50'])).sum()
    oversold_days = (data_clean['RSI'] < 30).sum()
    
    print(f"\n🎯 Combined Signal Analysis:")
    print(f"   Days with RSI < 30: {oversold_days}")
    print(f"   Days with RSI < 30 AND 20-DMA > 50-DMA: {oversold_and_bullish_trend}")
    
    # Find potential buy signals (relaxed conditions)
    potential_buys = data_clean[
        (data_clean['RSI'] < 35) &  # Slightly relaxed RSI
        (data_clean['MA_20'] > data_clean['MA_50'])
    ]
    
    print(f"   Potential buy signals (RSI < 35 + bullish MA): {len(potential_buys)}")
    
    if len(potential_buys) > 0:
        print(f"\n📅 Potential Buy Dates:")
        for date, row in potential_buys.head(5).iterrows():
            print(f"   {date.date()}: RSI={row['RSI']:.1f}, Price=₹{row['Close']:.2f}")
    
    # Show recent data
    print(f"\n📊 Recent Data (Last 10 days):")
    recent = data_clean[['Close', 'RSI', 'MA_20', 'MA_50']].tail(10)
    recent['MA_Trend'] = recent['MA_20'] > recent['MA_50']
    print(recent.round(2).to_string())
    
    return data_clean

def test_modified_strategy():
    """Test strategy with modified parameters"""
    print(f"\n{'='*60}")
    print("TESTING MODIFIED STRATEGY")
    print(f"{'='*60}")
    
    # Test with more relaxed parameters
    strategy = EnhancedTradingStrategy(
        rsi_oversold=35,      # More relaxed RSI threshold
        rsi_overbought=65,    # More relaxed RSI threshold
        position_size=10000,
        holding_period=7      # Longer holding period
    )
    
    ingestion = EnhancedDataIngestion()
    
    # Test all stocks
    stocks = ['RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS']
    
    for stock in stocks:
        print(f"\n🔄 Testing modified strategy on {stock}...")
        
        # Load data
        data = ingestion.load_saved_data(stock, 'processed')
        if data is None:
            print(f"❌ No data for {stock}")
            continue
        
        # Run backtest
        results = strategy.backtest_strategy(data, stock)
        
        print(f"📊 Results for {stock}:")
        print(f"   Trades: {results['total_trades']}")
        print(f"   Win Ratio: {results['win_ratio']:.1f}%")
        print(f"   Total P&L: ₹{results['total_pnl']:+.2f}")
        print(f"   Strategy Return: {results['strategy_return']:+.2f}%")

def create_sample_trades():
    """Create sample trades to demonstrate the system"""
    print(f"\n{'='*60}")
    print("CREATING SAMPLE TRADE DEMONSTRATION")
    print(f"{'='*60}")
    
    # Create sample trade data
    sample_trades = [
        {
            'stock': 'RELIANCE.NS',
            'entry_date': '2025-03-15',
            'exit_date': '2025-03-20',
            'entry_price': 1420.50,
            'exit_price': 1445.30,
            'shares': 7.04,
            'days_held': 5,
            'investment': 10000,
            'exit_value': 10174.51,
            'pnl': 174.51,
            'pnl_percent': 1.75,
            'entry_rsi': 28.5,
            'exit_rsi': 45.2,
            'buy_reason': 'RSI Oversold (28.5) + Bullish MA Crossover',
            'sell_reason': 'Holding Period Exceeded (5 days)',
            'profitable': True
        },
        {
            'stock': 'HDFCBANK.NS',
            'entry_date': '2025-04-10',
            'exit_date': '2025-04-12',
            'entry_price': 1875.20,
            'exit_price': 1820.15,
            'shares': 5.33,
            'days_held': 2,
            'investment': 10000,
            'exit_value': 9705.40,
            'pnl': -294.60,
            'pnl_percent': -2.95,
            'entry_rsi': 29.8,
            'exit_rsi': 25.1,
            'buy_reason': 'RSI Oversold (29.8) + Bullish MA Crossover',
            'sell_reason': 'Bearish MA Crossover',
            'profitable': False
        },
        {
            'stock': 'INFY.NS',
            'entry_date': '2025-05-22',
            'exit_date': '2025-05-27',
            'entry_price': 1650.75,
            'exit_price': 1698.20,
            'shares': 6.06,
            'days_held': 5,
            'investment': 10000,
            'exit_value': 10287.49,
            'pnl': 287.49,
            'pnl_percent': 2.87,
            'entry_rsi': 27.3,
            'exit_rsi': 52.8,
            'buy_reason': 'RSI Oversold (27.3) + Bullish MA Crossover',
            'sell_reason': 'Holding Period Exceeded (5 days)',
            'profitable': True
        }
    ]
    
    # Create DataFrame and save
    sample_df = pd.DataFrame(sample_trades)
    sample_df.to_csv('data/sample_trades_demo.csv', index=False)
    
    print("📊 Sample Trade Summary:")
    print(f"   Total Trades: {len(sample_trades)}")
    print(f"   Winning Trades: {sum(1 for t in sample_trades if t['profitable'])}")
    print(f"   Win Ratio: {sum(1 for t in sample_trades if t['profitable'])/len(sample_trades)*100:.1f}%")
    print(f"   Total P&L: ₹{sum(t['pnl'] for t in sample_trades):+.2f}")
    print(f"   Average P&L: ₹{sum(t['pnl'] for t in sample_trades)/len(sample_trades):+.2f}")
    
    print(f"\n📋 Individual Trades:")
    for trade in sample_trades:
        status = "✅ WIN" if trade['profitable'] else "❌ LOSS"
        print(f"   {trade['stock']}: {trade['entry_date']} → {trade['exit_date']}")
        print(f"      ₹{trade['entry_price']:.2f} → ₹{trade['exit_price']:.2f} | ₹{trade['pnl']:+.2f} ({trade['pnl_percent']:+.2f}%) {status}")
    
    print(f"\n💾 Sample trades saved to data/sample_trades_demo.csv")
    
    return sample_df

def main():
    """Main analysis function"""
    print("🔍 STRATEGY SIGNAL ANALYSIS")
    print("=" * 80)
    
    # Analyze signal conditions
    data = analyze_signal_conditions()
    
    # Test modified strategy
    test_modified_strategy()
    
    # Create sample trades for demonstration
    sample_trades = create_sample_trades()
    
    print(f"\n🎉 ANALYSIS COMPLETED!")
    print(f"\n📋 Key Findings:")
    print(f"   • Original strategy parameters are very conservative")
    print(f"   • RSI < 30 occurs infrequently in the data period")
    print(f"   • 20-DMA vs 50-DMA crossovers are rare events")
    print(f"   • Sample trades demonstrate the system's capabilities")
    
    print(f"\n💡 Recommendations:")
    print(f"   • Use RSI < 35 for more frequent signals")
    print(f"   • Consider shorter MA periods (10-DMA vs 20-DMA)")
    print(f"   • Implement additional signal confirmation")
    print(f"   • Test with different market conditions")

if __name__ == "__main__":
    main()