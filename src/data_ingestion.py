#!/usr/bin/env python3
"""
Enhanced Data Ingestion Module for Algo-Trading System
Handles data fetching, preprocessing, storage, and technical indicator calculations
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import ta
import os
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv('config/config.env')

class EnhancedDataIngestion:
    def __init__(self):
        self.api_key = os.environ.get('ALPHAVANTAGE_API_KEY')
        self.data_dir = 'data'
        self.ensure_data_directory()
        
    def ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"Created data directory: {self.data_dir}")
    
    def fetch_alpha_vantage_daily(self, symbol, outputsize='full'):
        """
        Fetch daily data from Alpha Vantage API
        Handles rate limits (5 calls/minute for free tier)
        """
        url = 'https://www.alphavantage.co/query'
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'outputsize': outputsize,  # 'compact' (100 days) or 'full' (20+ years)
            'apikey': self.api_key
        }
        
        print(f"Fetching Alpha Vantage data for {symbol}...")
        
        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            # Check for API errors
            if "Error Message" in data:
                print(f"❌ Alpha Vantage Error: {data['Error Message']}")
                return None
            
            if "Note" in data:
                print(f"⚠️ Alpha Vantage Rate Limit: {data['Note']}")
                print("Waiting 60 seconds before retry...")
                time.sleep(60)
                return self.fetch_alpha_vantage_daily(symbol, outputsize)
            
            # Extract time series data
            time_series_key = 'Time Series (Daily)'
            if time_series_key not in data:
                print(f"❌ Unexpected response structure for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df.astype(float)
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            
            print(f"✅ Fetched {len(df)} records from Alpha Vantage for {symbol}")
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error fetching {symbol}: {e}")
            return None
        except Exception as e:
            print(f"❌ Error processing Alpha Vantage data for {symbol}: {e}")
            return None
    
    def fetch_yfinance_data(self, symbol, period='6mo', interval='1d'):
        """
        Fetch data from Yahoo Finance using yfinance
        More reliable for Indian stocks
        """
        print(f"Fetching yfinance data for {symbol}...")
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                print(f"❌ No data found for {symbol}")
                return None
            
            # Ensure consistent column names
            df = df.rename(columns={
                'Open': 'Open',
                'High': 'High', 
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })
            
            # Keep only OHLCV columns
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            print(f"✅ Fetched {len(df)} records from yfinance for {symbol}")
            return df
            
        except Exception as e:
            print(f"❌ Error fetching yfinance data for {symbol}: {e}")
            return None
    
    def fetch_nifty_stocks_data(self, stocks=None, period='6mo', save_raw=True):
        """
        Fetch data for multiple NIFTY 50 stocks with rate limiting
        """
        if stocks is None:
            stocks = os.getenv('NIFTY_STOCKS', 'RELIANCE.NS,HDFCBANK.NS,INFY.NS').split(',')
        
        print(f"\n{'='*60}")
        print(f"FETCHING DATA FOR {len(stocks)} NIFTY 50 STOCKS")
        print(f"{'='*60}")
        print(f"Stocks: {', '.join(stocks)}")
        print(f"Period: {period}")
        print(f"Save raw data: {save_raw}")
        
        stock_data = {}
        failed_stocks = []
        
        for i, stock in enumerate(stocks, 1):
            print(f"\n[{i}/{len(stocks)}] Processing {stock}...")
            
            # Try yfinance first (better for Indian stocks)
            data = self.fetch_yfinance_data(stock, period=period)
            
            # Fallback to Alpha Vantage if yfinance fails
            if data is None and self.api_key:
                print(f"Trying Alpha Vantage for {stock}...")
                data = self.fetch_alpha_vantage_daily(stock.replace('.NS', ''))
                
                # Rate limiting for Alpha Vantage
                if i < len(stocks):
                    print("Waiting 12 seconds (Alpha Vantage rate limit)...")
                    time.sleep(12)
            
            if data is not None:
                # Data preprocessing
                data = self.preprocess_data(data, stock)
                stock_data[stock] = data
                
                # Save raw data
                if save_raw:
                    self.save_raw_data(data, stock)
                
                print(f"✅ Successfully processed {stock}: {len(data)} records")
                print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
            else:
                failed_stocks.append(stock)
                print(f"❌ Failed to fetch data for {stock}")
        
        # Summary
        print(f"\n{'='*60}")
        print(f"DATA FETCHING SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Successfully fetched: {len(stock_data)} stocks")
        print(f"❌ Failed to fetch: {len(failed_stocks)} stocks")
        
        if failed_stocks:
            print(f"Failed stocks: {', '.join(failed_stocks)}")
        
        return stock_data
    
    def preprocess_data(self, df, symbol):
        """
        Clean and preprocess stock data
        """
        df = df.copy()
        
        # Handle missing values
        if df.isnull().any().any():
            print(f"⚠️ Found missing values in {symbol}, forward filling...")
            df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Ensure correct data types
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any remaining NaN rows
        df = df.dropna()
        
        # Ensure chronological order
        df = df.sort_index()
        
        # Basic data validation
        if len(df) < 30:
            print(f"⚠️ Warning: Only {len(df)} records for {symbol} (minimum 30 recommended)")
        
        # Validate OHLC relationships
        invalid_ohlc = (df['High'] < df['Low']) | (df['Close'] > df['High']) | (df['Close'] < df['Low'])
        if invalid_ohlc.any():
            print(f"⚠️ Warning: Found {invalid_ohlc.sum()} invalid OHLC relationships in {symbol}")
            df = df[~invalid_ohlc]
        
        return df
    
    def calculate_technical_indicators(self, df):
        """
        Calculate comprehensive technical indicators
        """
        df = df.copy()
        
        print("Calculating technical indicators...")
        
        # RSI (14-day period)
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        
        # Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()  # 20-day Simple MA
        df['SMA_50'] = df['Close'].rolling(window=50).mean()  # 50-day Simple MA
        df['EMA_12'] = df['Close'].ewm(span=12).mean()        # 12-day Exponential MA
        df['EMA_26'] = df['Close'].ewm(span=26).mean()        # 26-day Exponential MA
        
        # MACD (Moving Average Convergence Divergence)
        macd_indicator = ta.trend.MACD(df['Close'])
        df['MACD'] = macd_indicator.macd()
        df['MACD_Signal'] = macd_indicator.macd_signal()
        df['MACD_Histogram'] = macd_indicator.macd_diff()
        
        # Bollinger Bands
        bb_indicator = ta.volatility.BollingerBands(df['Close'])
        df['BB_Upper'] = bb_indicator.bollinger_hband()
        df['BB_Middle'] = bb_indicator.bollinger_mavg()
        df['BB_Lower'] = bb_indicator.bollinger_lband()
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Average True Range (ATR)
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        
        # Stochastic Oscillator
        stoch_indicator = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = stoch_indicator.stoch()
        df['Stoch_D'] = stoch_indicator.stoch_signal()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price-based indicators
        df['Price_Change'] = df['Close'].pct_change()
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Open_Ratio'] = df['Close'] / df['Open']
        
        # Volatility
        df['Volatility'] = df['Price_Change'].rolling(window=20).std()
        
        print(f"✅ Calculated {len([col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']])} technical indicators")
        
        return df
    
    def save_raw_data(self, df, symbol):
        """Save raw data to CSV for debugging and reuse"""
        filename = f"{self.data_dir}/{symbol.replace('.', '_')}_raw_data.csv"
        df.to_csv(filename)
        print(f"💾 Saved raw data to {filename}")
    
    def save_processed_data(self, df, symbol):
        """Save processed data with indicators to CSV"""
        filename = f"{self.data_dir}/{symbol.replace('.', '_')}_processed_data.csv"
        df.to_csv(filename)
        print(f"💾 Saved processed data to {filename}")
    
    def load_saved_data(self, symbol, data_type='processed'):
        """Load previously saved data"""
        suffix = 'processed_data.csv' if data_type == 'processed' else 'raw_data.csv'
        filename = f"{self.data_dir}/{symbol.replace('.', '_')}_{suffix}"
        
        if os.path.exists(filename):
            df = pd.read_csv(filename, index_col=0, parse_dates=True)
            print(f"📂 Loaded saved data from {filename}")
            return df
        else:
            print(f"❌ No saved data found: {filename}")
            return None
    
    def get_data_summary(self, stock_data):
        """Generate comprehensive data summary"""
        print(f"\n{'='*60}")
        print(f"DATA SUMMARY")
        print(f"{'='*60}")
        
        summary_data = []
        
        for stock, df in stock_data.items():
            summary = {
                'Stock': stock,
                'Records': len(df),
                'Start_Date': df.index[0].strftime('%Y-%m-%d'),
                'End_Date': df.index[-1].strftime('%Y-%m-%d'),
                'Avg_Volume': f"{df['Volume'].mean():,.0f}",
                'Price_Range': f"₹{df['Low'].min():.2f} - ₹{df['High'].max():.2f}",
                'Latest_Close': f"₹{df['Close'].iloc[-1]:.2f}",
                'Total_Return': f"{((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100:.2f}%"
            }
            summary_data.append(summary)
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Save summary
        summary_df.to_csv(f"{self.data_dir}/data_summary.csv", index=False)
        print(f"\n💾 Data summary saved to {self.data_dir}/data_summary.csv")
        
        return summary_df
    
    def run_complete_ingestion(self, stocks=None, period='6mo', calculate_indicators=True):
        """
        Run complete data ingestion pipeline
        """
        print(f"\n{'='*60}")
        print(f"COMPLETE DATA INGESTION PIPELINE")
        print(f"{'='*60}")
        
        # Step 1: Fetch raw data
        stock_data = self.fetch_nifty_stocks_data(stocks, period)
        
        if not stock_data:
            print("❌ No data fetched. Exiting...")
            return None
        
        # Step 2: Calculate technical indicators
        if calculate_indicators:
            print(f"\n{'='*40}")
            print(f"CALCULATING TECHNICAL INDICATORS")
            print(f"{'='*40}")
            
            for stock in stock_data:
                print(f"\nProcessing indicators for {stock}...")
                stock_data[stock] = self.calculate_technical_indicators(stock_data[stock])
                self.save_processed_data(stock_data[stock], stock)
        
        # Step 3: Generate summary
        summary = self.get_data_summary(stock_data)
        
        print(f"\n✅ Data ingestion pipeline completed successfully!")
        print(f"📊 Processed {len(stock_data)} stocks with {sum(len(df) for df in stock_data.values())} total records")
        
        return stock_data, summary

def main():
    """Main function for testing data ingestion"""
    ingestion = EnhancedDataIngestion()
    
    # Run complete ingestion
    stock_data, summary = ingestion.run_complete_ingestion()
    
    if stock_data:
        print(f"\n🎉 Data ingestion completed!")
        print(f"Available stocks: {list(stock_data.keys())}")
        
        # Show sample data
        sample_stock = list(stock_data.keys())[0]
        sample_data = stock_data[sample_stock]
        print(f"\nSample data for {sample_stock} (last 5 records):")
        print(sample_data[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'SMA_20', 'MACD']].tail())

if __name__ == "__main__":
    main()