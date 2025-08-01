#!/usr/bin/env python3
"""
Comprehensive Data Analysis Script
Demonstrates the enhanced data ingestion capabilities with detailed analysis
"""

import sys
import os
sys.path.append('src')

from src.data_ingestion import EnhancedDataIngestion
from src.trading_strategy import TradingStrategy
from src.ml_predictor import MLPredictor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_stock_data():
    """Comprehensive analysis of fetched stock data"""
    print("🔍 COMPREHENSIVE STOCK DATA ANALYSIS")
    print("=" * 60)
    
    # Initialize components
    ingestion = EnhancedDataIngestion()
    strategy = TradingStrategy()
    predictor = MLPredictor()
    
    # Fetch data for analysis
    stocks = ['RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS']
    print(f"Analyzing {len(stocks)} NIFTY 50 stocks with 6 months of data...")
    
    stock_data, summary = ingestion.run_complete_ingestion(
        stocks=stocks,
        period='6mo',
        calculate_indicators=True
    )
    
    if not stock_data:
        print("❌ No data available for analysis")
        return
    
    # Detailed analysis for each stock
    analysis_results = {}
    
    for stock, data in stock_data.items():
        print(f"\n{'='*50}")
        print(f"DETAILED ANALYSIS: {stock}")
        print(f"{'='*50}")
        
        # Basic statistics
        print(f"📊 Basic Statistics:")
        print(f"   Records: {len(data)}")
        print(f"   Date Range: {data.index[0].date()} to {data.index[-1].date()}")
        print(f"   Latest Price: ₹{data['Close'].iloc[-1]:.2f}")
        
        # Price performance
        total_return = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
        volatility = data['Price_Change'].std() * 100
        max_price = data['High'].max()
        min_price = data['Low'].min()
        
        print(f"   Total Return: {total_return:+.2f}%")
        print(f"   Volatility: {volatility:.2f}%")
        print(f"   Price Range: ₹{min_price:.2f} - ₹{max_price:.2f}")
        
        # Technical analysis
        print(f"\n📈 Technical Analysis:")
        current_rsi = data['RSI'].iloc[-1]
        current_sma20 = data['SMA_20'].iloc[-1]
        current_sma50 = data['SMA_50'].iloc[-1]
        
        print(f"   Current RSI: {current_rsi:.2f}")
        print(f"   SMA 20: ₹{current_sma20:.2f}")
        print(f"   SMA 50: ₹{current_sma50:.2f}")
        
        # RSI analysis
        if current_rsi < 30:
            rsi_signal = "OVERSOLD 📉"
        elif current_rsi > 70:
            rsi_signal = "OVERBOUGHT 📈"
        else:
            rsi_signal = "NEUTRAL ➡️"
        print(f"   RSI Signal: {rsi_signal}")
        
        # Moving average analysis
        if current_sma20 > current_sma50:
            ma_trend = "BULLISH 📈"
        else:
            ma_trend = "BEARISH 📉"
        print(f"   MA Trend: {ma_trend}")
        
        # Trading strategy analysis
        print(f"\n🎯 Trading Strategy Analysis:")
        backtest_results = strategy.backtest_strategy(data)
        
        print(f"   Strategy Return: {backtest_results['total_return']:+.2%}")
        print(f"   Market Return: {backtest_results['market_return']:+.2%}")
        print(f"   Win Rate: {backtest_results['win_rate']:.1f}%")
        print(f"   Total Trades: {backtest_results['total_trades']}")
        print(f"   Sharpe Ratio: {backtest_results['sharpe_ratio']:.3f}")
        
        # ML prediction
        print(f"\n🤖 Machine Learning Analysis:")
        try:
            ml_results = predictor.train_model(data)
            if ml_results:
                print(f"   Model Accuracy: {ml_results['accuracy']:.1%}")
                
                # Top features
                feature_importance = ml_results['feature_importance']
                top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"   Top Predictive Features:")
                for i, (feature, importance) in enumerate(top_features, 1):
                    print(f"     {i}. {feature}: {importance:.3f}")
                
                # Next day prediction
                prediction = predictor.predict_next_day(data)
                if prediction:
                    direction = "📈 UP" if prediction['prediction'] == 1 else "📉 DOWN"
                    print(f"   Next Day Prediction: {direction}")
                    print(f"   Confidence: {prediction['confidence']:.1%}")
            else:
                print("   ❌ ML model training failed")
        except Exception as e:
            print(f"   ❌ ML analysis error: {e}")
        
        # Volume analysis
        print(f"\n📊 Volume Analysis:")
        avg_volume = data['Volume'].mean()
        recent_volume = data['Volume'].tail(5).mean()
        volume_trend = "HIGH" if recent_volume > avg_volume * 1.2 else "LOW" if recent_volume < avg_volume * 0.8 else "NORMAL"
        
        print(f"   Average Volume: {avg_volume:,.0f}")
        print(f"   Recent Volume: {recent_volume:,.0f}")
        print(f"   Volume Trend: {volume_trend}")
        
        # Store results
        analysis_results[stock] = {
            'total_return': total_return,
            'volatility': volatility,
            'rsi': current_rsi,
            'rsi_signal': rsi_signal,
            'ma_trend': ma_trend,
            'strategy_return': backtest_results['total_return'] * 100,
            'win_rate': backtest_results['win_rate'],
            'total_trades': backtest_results['total_trades'],
            'volume_trend': volume_trend
        }
    
    # Comparative analysis
    print(f"\n{'='*60}")
    print(f"COMPARATIVE ANALYSIS")
    print(f"{'='*60}")
    
    # Create comparison DataFrame
    comparison_data = []
    for stock, results in analysis_results.items():
        comparison_data.append({
            'Stock': stock,
            'Return': f"{results['total_return']:+.2f}%",
            'Volatility': f"{results['volatility']:.2f}%",
            'RSI': f"{results['rsi']:.1f}",
            'RSI_Signal': results['rsi_signal'].split()[0],
            'MA_Trend': results['ma_trend'].split()[0],
            'Strategy_Return': f"{results['strategy_return']:+.2f}%",
            'Win_Rate': f"{results['win_rate']:.1f}%",
            'Trades': results['total_trades'],
            'Volume': results['volume_trend']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Best performers
    print(f"\n🏆 PERFORMANCE HIGHLIGHTS:")
    
    # Best return
    best_return_stock = max(analysis_results.items(), key=lambda x: x[1]['total_return'])
    print(f"   📈 Best Return: {best_return_stock[0]} ({best_return_stock[1]['total_return']:+.2f}%)")
    
    # Best strategy performance
    best_strategy_stock = max(analysis_results.items(), key=lambda x: x[1]['strategy_return'])
    print(f"   🎯 Best Strategy: {best_strategy_stock[0]} ({best_strategy_stock[1]['strategy_return']:+.2f}%)")
    
    # Highest win rate
    best_winrate_stock = max(analysis_results.items(), key=lambda x: x[1]['win_rate'])
    print(f"   🎲 Best Win Rate: {best_winrate_stock[0]} ({best_winrate_stock[1]['win_rate']:.1f}%)")
    
    # Most active trading
    most_trades_stock = max(analysis_results.items(), key=lambda x: x[1]['total_trades'])
    print(f"   🔄 Most Active: {most_trades_stock[0]} ({most_trades_stock[1]['total_trades']} trades)")
    
    # Save comparison
    comparison_df.to_csv('data/stock_comparison.csv', index=False)
    print(f"\n💾 Comparison saved to data/stock_comparison.csv")
    
    print(f"\n✅ Comprehensive analysis completed!")
    print(f"📊 Analyzed {len(stock_data)} stocks with {sum(len(df) for df in stock_data.values())} total records")
    
    return analysis_results, comparison_df

def show_data_quality_report():
    """Generate data quality report"""
    print(f"\n{'='*60}")
    print(f"DATA QUALITY REPORT")
    print(f"{'='*60}")
    
    # Check saved files
    data_files = [f for f in os.listdir('data') if f.endswith('.csv')]
    
    print(f"📁 Data Files Created: {len(data_files)}")
    for file in data_files:
        file_path = f"data/{file}"
        file_size = os.path.getsize(file_path)
        print(f"   📄 {file}: {file_size:,} bytes")
    
    # Load and analyze a sample file
    if 'RELIANCE_NS_processed_data.csv' in data_files:
        sample_data = pd.read_csv('data/RELIANCE_NS_processed_data.csv', index_col=0, parse_dates=True)
        
        print(f"\n📊 Sample Data Quality (RELIANCE.NS):")
        print(f"   Records: {len(sample_data)}")
        print(f"   Columns: {len(sample_data.columns)}")
        print(f"   Missing Values: {sample_data.isnull().sum().sum()}")
        print(f"   Date Range: {sample_data.index[0].date()} to {sample_data.index[-1].date()}")
        
        # Show indicator coverage
        indicator_columns = [col for col in sample_data.columns 
                           if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        print(f"   Technical Indicators: {len(indicator_columns)}")
        
        # Check for any data anomalies
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in sample_data.columns:
                zero_prices = (sample_data[col] <= 0).sum()
                if zero_prices > 0:
                    print(f"   ⚠️ Warning: {zero_prices} zero/negative prices in {col}")

def main():
    """Main analysis function"""
    print("🚀 ENHANCED DATA ANALYSIS SYSTEM")
    print("=" * 60)
    
    try:
        # Run comprehensive analysis
        results, comparison = analyze_stock_data()
        
        # Show data quality report
        show_data_quality_report()
        
        print(f"\n🎉 Analysis completed successfully!")
        print(f"\nKey files generated:")
        print(f"   📊 data/stock_comparison.csv - Comparative analysis")
        print(f"   📈 data/*_processed_data.csv - Technical indicators")
        print(f"   📋 data/data_summary.csv - Data summary")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()