#!/usr/bin/env python3
"""
Demo script showing the algo-trading system with extended data
"""

import sys
import os
sys.path.append('src')

from src.api_call import DataFetcher
from src.trading_strategy import TradingStrategy
from src.ml_predictor import MLPredictor

def run_demo():
    """Run a comprehensive demo"""
    print("=" * 60)
    print("ALGO-TRADING SYSTEM DEMO")
    print("=" * 60)
    
    # Initialize components
    fetcher = DataFetcher()
    strategy = TradingStrategy()
    predictor = MLPredictor()
    
    # Test with RELIANCE stock with more data
    stock = 'RELIANCE.NS'
    print(f"Analyzing {stock} with 6 months of data...")
    
    # Fetch more data for better analysis
    data = fetcher.fetch_yfinance_data(stock, period='6mo', interval='1d')
    
    if data is None or data.empty:
        print(f"❌ Could not fetch data for {stock}")
        return
    
    print(f"✅ Fetched {len(data)} records")
    print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
    
    # Run strategy analysis
    print("\n📊 Running trading strategy analysis...")
    results = strategy.backtest_strategy(data)
    
    print(f"Strategy Results:")
    print(f"  📈 Total Return: {results['total_return']:.2%}")
    print(f"  📊 Market Return: {results['market_return']:.2%}")
    print(f"  🎯 Win Rate: {results['win_rate']:.1f}%")
    print(f"  🔄 Total Trades: {results['total_trades']}")
    print(f"  📉 Sharpe Ratio: {results['sharpe_ratio']:.3f}")
    print(f"  💰 Final Portfolio Value: ₹{results['final_portfolio_value']:,.2f}")
    
    # Show recent trades
    if not results['trades'].empty:
        print(f"\n🔄 Recent Trades:")
        recent_trades = results['trades'].tail(5)
        for idx, trade in recent_trades.iterrows():
            print(f"  {idx.date()}: {trade['Trade_Type']} at ₹{trade['Close']:.2f} (RSI: {trade.get('RSI', 'N/A'):.1f})")
    
    # ML Analysis
    print(f"\n🤖 Training ML model...")
    try:
        ml_results = predictor.train_model(data)
        
        if ml_results:
            print(f"ML Model Results:")
            print(f"  🎯 Accuracy: {ml_results['accuracy']:.1%}")
            
            # Show top features
            feature_importance = ml_results['feature_importance']
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"  📊 Top Features:")
            for feature, importance in top_features:
                print(f"    - {feature}: {importance:.3f}")
            
            # Make prediction
            prediction = predictor.predict_next_day(data)
            if prediction:
                direction = "📈 UP" if prediction['prediction'] == 1 else "📉 DOWN"
                print(f"  🔮 Next Day Prediction: {direction}")
                print(f"  🎯 Confidence: {prediction['confidence']:.1%}")
        
    except Exception as e:
        print(f"❌ ML analysis failed: {e}")
    
    print(f"\n✅ Demo completed successfully!")
    print(f"\nTo run the full system with Google Sheets integration:")
    print(f"1. Set up Google Sheets credentials in config/")
    print(f"2. Update config/config.env with your sheet ID")
    print(f"3. Run: python src/main.py")

if __name__ == "__main__":
    run_demo()