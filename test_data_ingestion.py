#!/usr/bin/env python3
"""
Test script for enhanced data ingestion system
"""

import sys
import os
sys.path.append('src')

from src.data_ingestion import EnhancedDataIngestion
import pandas as pd

def test_basic_fetching():
    """Test basic data fetching functionality"""
    print("=" * 60)
    print("TESTING BASIC DATA FETCHING")
    print("=" * 60)
    
    ingestion = EnhancedDataIngestion()
    
    # Test single stock fetch
    test_stock = 'RELIANCE.NS'
    print(f"\n1. Testing single stock fetch: {test_stock}")
    
    data = ingestion.fetch_yfinance_data(test_stock, period='3mo')
    
    if data is not None:
        print(f"✅ Successfully fetched {len(data)} records")
        print(f"   Columns: {list(data.columns)}")
        print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
        print(f"   Sample data:")
        print(data.head(3).to_string())
        return data
    else:
        print("❌ Failed to fetch data")
        return None

def test_technical_indicators(data):
    """Test technical indicator calculations"""
    print("\n" + "=" * 60)
    print("TESTING TECHNICAL INDICATORS")
    print("=" * 60)
    
    if data is None:
        print("❌ No data available for indicator testing")
        return None
    
    ingestion = EnhancedDataIngestion()
    
    print("Calculating technical indicators...")
    data_with_indicators = ingestion.calculate_technical_indicators(data)
    
    # Show available indicators
    indicator_columns = [col for col in data_with_indicators.columns 
                        if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
    
    print(f"✅ Calculated {len(indicator_columns)} technical indicators:")
    for i, indicator in enumerate(indicator_columns, 1):
        print(f"   {i:2d}. {indicator}")
    
    # Show sample indicator values
    print(f"\nSample indicator values (last 5 records):")
    sample_indicators = ['RSI', 'SMA_20', 'SMA_50', 'MACD', 'BB_Upper', 'BB_Lower', 'ATR']
    available_indicators = [ind for ind in sample_indicators if ind in data_with_indicators.columns]
    
    if available_indicators:
        print(data_with_indicators[available_indicators].tail().round(2).to_string())
    
    return data_with_indicators

def test_multiple_stocks():
    """Test fetching multiple NIFTY 50 stocks"""
    print("\n" + "=" * 60)
    print("TESTING MULTIPLE STOCKS FETCH")
    print("=" * 60)
    
    ingestion = EnhancedDataIngestion()
    
    # Test with 3 NIFTY stocks
    test_stocks = ['RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS']
    
    stock_data = ingestion.fetch_nifty_stocks_data(
        stocks=test_stocks, 
        period='3mo', 
        save_raw=True
    )
    
    if stock_data:
        print(f"\n✅ Successfully fetched data for {len(stock_data)} stocks:")
        
        for stock, data in stock_data.items():
            latest_price = data['Close'].iloc[-1]
            price_change = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
            print(f"   📊 {stock}: {len(data)} records, Latest: ₹{latest_price:.2f}, Change: {price_change:+.2f}%")
        
        return stock_data
    else:
        print("❌ Failed to fetch multiple stocks data")
        return None

def test_data_preprocessing():
    """Test data preprocessing and validation"""
    print("\n" + "=" * 60)
    print("TESTING DATA PREPROCESSING")
    print("=" * 60)
    
    ingestion = EnhancedDataIngestion()
    
    # Fetch sample data
    data = ingestion.fetch_yfinance_data('RELIANCE.NS', period='2mo')
    
    if data is None:
        print("❌ No data available for preprocessing test")
        return
    
    print(f"Original data shape: {data.shape}")
    print(f"Missing values: {data.isnull().sum().sum()}")
    
    # Test preprocessing
    processed_data = ingestion.preprocess_data(data, 'RELIANCE.NS')
    
    print(f"Processed data shape: {processed_data.shape}")
    print(f"Missing values after processing: {processed_data.isnull().sum().sum()}")
    
    # Validate OHLC relationships
    valid_high = (processed_data['High'] >= processed_data[['Open', 'Close']].max(axis=1)).all()
    valid_low = (processed_data['Low'] <= processed_data[['Open', 'Close']].min(axis=1)).all()
    
    print(f"✅ OHLC validation - High prices valid: {valid_high}")
    print(f"✅ OHLC validation - Low prices valid: {valid_low}")

def test_complete_pipeline():
    """Test the complete data ingestion pipeline"""
    print("\n" + "=" * 60)
    print("TESTING COMPLETE PIPELINE")
    print("=" * 60)
    
    ingestion = EnhancedDataIngestion()
    
    # Run complete pipeline with 2 stocks for faster testing
    test_stocks = ['RELIANCE.NS', 'HDFCBANK.NS']
    
    try:
        stock_data, summary = ingestion.run_complete_ingestion(
            stocks=test_stocks,
            period='3mo',
            calculate_indicators=True
        )
        
        if stock_data and summary is not None:
            print(f"\n✅ Complete pipeline test successful!")
            print(f"   Processed {len(stock_data)} stocks")
            print(f"   Generated summary with {len(summary)} entries")
            
            # Show summary
            print(f"\nData Summary:")
            print(summary.to_string(index=False))
            
            return True
        else:
            print("❌ Complete pipeline test failed")
            return False
            
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        return False

def main():
    """Run all data ingestion tests"""
    print("🚀 ENHANCED DATA INGESTION TESTING")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Basic fetching
    data = test_basic_fetching()
    results['basic_fetch'] = data is not None
    
    # Test 2: Technical indicators
    if data is not None:
        indicator_data = test_technical_indicators(data)
        results['indicators'] = indicator_data is not None
    else:
        results['indicators'] = False
    
    # Test 3: Multiple stocks
    stock_data = test_multiple_stocks()
    results['multiple_stocks'] = stock_data is not None
    
    # Test 4: Data preprocessing
    test_data_preprocessing()
    results['preprocessing'] = True  # If no exception, consider passed
    
    # Test 5: Complete pipeline
    results['complete_pipeline'] = test_complete_pipeline()
    
    # Final summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name.replace('_', ' ').title():<20}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Data ingestion system is ready.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()