#!/usr/bin/env python3
"""
Test script to verify the algo-trading system setup
"""

import sys
import os
sys.path.append('src')

from src.api_call import DataFetcher
from src.trading_strategy import TradingStrategy
from src.ml_predictor import MLPredictor

def test_data_fetching():
    """Test data fetching functionality"""
    print("Testing data fetching...")
    
    fetcher = DataFetcher()
    
    # Test with a single stock
    test_stock = 'RELIANCE.NS'
    data = fetcher.fetch_yfinance_data(test_stock, period='1mo')
    
    if data is not None and not data.empty:
        print(f"✅ Successfully fetched {len(data)} records for {test_stock}")
        print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
        print(f"   Columns: {list(data.columns)}")
        return data
    else:
        print(f"❌ Failed to fetch data for {test_stock}")
        return None

def test_trading_strategy(data):
    """Test trading strategy"""
    print("\nTesting trading strategy...")
    
    if data is None:
        print("❌ No data available for strategy testing")
        return None
    
    strategy = TradingStrategy()
    results = strategy.backtest_strategy(data)
    
    print(f"✅ Strategy backtest completed:")
    print(f"   Total Return: {results['total_return']:.2%}")
    print(f"   Win Rate: {results['win_rate']:.1f}%")
    print(f"   Total Trades: {results['total_trades']}")
    
    return results

def test_ml_predictor(data):
    """Test ML predictor"""
    print("\nTesting ML predictor...")
    
    if data is None:
        print("❌ No data available for ML testing")
        return None
    
    predictor = MLPredictor()
    
    try:
        ml_results = predictor.train_model(data)
        
        if ml_results:
            print(f"✅ ML model trained successfully:")
            print(f"   Accuracy: {ml_results['accuracy']:.3f}")
            
            # Test prediction
            prediction = predictor.predict_next_day(data)
            if prediction:
                direction = "UP" if prediction['prediction'] == 1 else "DOWN"
                print(f"   Next day prediction: {direction}")
                print(f"   Confidence: {prediction['confidence']:.3f}")
            
            return ml_results
        else:
            print("❌ ML model training failed")
            return None
            
    except Exception as e:
        print(f"❌ ML predictor error: {e}")
        return None

def main():
    """Run all tests"""
    print("=" * 50)
    print("ALGO-TRADING SYSTEM TEST")
    print("=" * 50)
    
    # Test data fetching
    data = test_data_fetching()
    
    # Test trading strategy
    strategy_results = test_trading_strategy(data)
    
    # Test ML predictor
    ml_results = test_ml_predictor(data)
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    if data is not None:
        print("✅ Data fetching: PASSED")
    else:
        print("❌ Data fetching: FAILED")
    
    if strategy_results is not None:
        print("✅ Trading strategy: PASSED")
    else:
        print("❌ Trading strategy: FAILED")
    
    if ml_results is not None:
        print("✅ ML predictor: PASSED")
    else:
        print("❌ ML predictor: FAILED")
    
    print("\nNote: Google Sheets integration requires proper credentials setup.")
    print("Run 'python src/main.py' for full system test with Google Sheets.")

if __name__ == "__main__":
    main()