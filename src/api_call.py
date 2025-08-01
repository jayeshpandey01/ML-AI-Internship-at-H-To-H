import requests
import os
import pandas as pd
import numpy as np
import yfinance as yf
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load environment variables
load_dotenv('config/config.env')

# Get API key from environment
API_KEY = os.environ.get('ALPHAVANTAGE_API_KEY')

class DataFetcher:
    def __init__(self):
        self.api_key = API_KEY
        self.data_dir = 'data'
        self.ensure_data_directory()
        
    def ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
    def fetch_alpha_vantage_data(self, symbol='IBM', interval='5min', outputsize='compact'):
        """Fetch data from Alpha Vantage API"""
        url = 'https://www.alphavantage.co/query'
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': interval,
            'outputsize': outputsize,
            'apikey': self.api_key
        }

        response = requests.get(url, params=params)
        data = response.json()

        if "Error Message" in data or "Note" in data:
            print(f"Error fetching data for {symbol}:", data)
            return None

        time_series_key = f'Time Series ({interval})'
        if time_series_key in data:
            df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df.astype(float)
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            return df
        else:
            print(f"Unexpected response structure for {symbol}:", data)
            return None

    def fetch_yfinance_data(self, symbol, period='1mo', interval='1d'):
        """Fetch data from Yahoo Finance using yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            if df.empty:
                print(f"No data found for symbol {symbol}")
                return None
            return df
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None

    def fetch_nifty_stocks_data(self, stocks=['RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS'], 
                               period='6mo', interval='1d', save_to_csv=True):
        """Fetch data for multiple NIFTY 50 stocks with enhanced preprocessing"""
        stock_data = {}
        
        for stock in stocks:
            print(f"Fetching data for {stock}...")
            data = self.fetch_yfinance_data(stock, period=period, interval=interval)
            
            if data is not None:
                # Basic preprocessing
                data = self._preprocess_basic(data, stock)
                stock_data[stock] = data
                
                # Save to CSV if requested
                if save_to_csv:
                    filename = f"{self.data_dir}/{stock.replace('.', '_')}_data.csv"
                    data.to_csv(filename)
                    print(f"💾 Saved data to {filename}")
                
                print(f"Successfully fetched {len(data)} records for {stock}")
                print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
            else:
                print(f"Failed to fetch data for {stock}")
                
        return stock_data
    
    def _preprocess_basic(self, df, symbol):
        """Basic data preprocessing"""
        df = df.copy()
        
        # Handle missing values
        if df.isnull().any().any():
            print(f"⚠️ Handling missing values for {symbol}")
            df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Ensure correct data types
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any remaining NaN rows
        df = df.dropna()
        
        # Sort by date
        df = df.sort_index()
        
        return df

# Legacy function for backward compatibility
def fetch_intraday_data(symbol='IBM', interval='5min'):
    fetcher = DataFetcher()
    return fetcher.fetch_alpha_vantage_data(symbol, interval)

