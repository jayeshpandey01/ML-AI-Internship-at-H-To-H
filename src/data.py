"""
Data Management Module

This module provides comprehensive data fetching, validation, caching, and preprocessing
capabilities for the algo trading system. It consolidates functionality from multiple
data sources and provides a unified interface for all data operations.

Classes:
    DataFetcher: Main class for data fetching operations
    DataValidator: Validates and cleans market data
    DataCache: Manages data caching to reduce API calls
    DataPreprocessor: Handles data preprocessing and technical indicators

Example:
    >>> from src.data import DataFetcher
    >>> fetcher = DataFetcher()
    >>> data = fetcher.fetch_stock_data('RELIANCE.NS', period='6mo')
    >>> print(f"Fetched {len(data)} records")
"""

import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import hashlib

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import ta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import ConfigManager
from .utils import (
    validate_data_integrity,
    retry_on_exception,
    ensure_directory_exists,
    format_currency,
    format_percentage
)


class DataFetchError(Exception):
    """Raised when data fetching fails."""
    pass


class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass


class DataCache:
    """
    Manages data caching to reduce API calls and improve performance.
    
    This class provides intelligent caching of market data with configurable
    expiry times and automatic cache invalidation.
    
    Attributes:
        cache_dir: Directory to store cached data
        default_expiry: Default cache expiry time in minutes
    """
    
    def __init__(self, cache_dir: str = "data/cache", default_expiry: int = 60) -> None:
        """
        Initialize the data cache.
        
        Args:
            cache_dir: Directory to store cached data
            default_expiry: Default cache expiry time in minutes
        """
        self.cache_dir = Path(cache_dir)
        self.default_expiry = default_expiry
        self.logger = logging.getLogger(__name__)
        
        # Ensure cache directory exists
        ensure_directory_exists(str(self.cache_dir))
    
    def _get_cache_key(self, symbol: str, period: str, interval: str, source: str) -> str:
        """
        Generate a unique cache key for the data request.
        
        Args:
            symbol: Stock symbol
            period: Data period
            interval: Data interval
            source: Data source
            
        Returns:
            Unique cache key
        """
        key_string = f"{symbol}_{period}_{interval}_{source}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """
        Get the file path for cached data.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Path to cache file
        """
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _get_metadata_path(self, cache_key: str) -> Path:
        """
        Get the file path for cache metadata.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Path to metadata file
        """
        return self.cache_dir / f"{cache_key}_meta.json"
    
    def is_cache_valid(self, cache_key: str, expiry_minutes: Optional[int] = None) -> bool:
        """
        Check if cached data is still valid.
        
        Args:
            cache_key: Cache key to check
            expiry_minutes: Custom expiry time in minutes
            
        Returns:
            True if cache is valid, False otherwise
        """
        metadata_path = self._get_metadata_path(cache_key)
        
        if not metadata_path.exists():
            return False
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            cached_time = datetime.fromisoformat(metadata['timestamp'])
            expiry_time = expiry_minutes or self.default_expiry
            
            return datetime.now() - cached_time < timedelta(minutes=expiry_time)
            
        except Exception as e:
            self.logger.warning(f"Error checking cache validity: {e}")
            return False
    
    def get_cached_data(self, cache_key: str) -> Optional[pd.DataFrame]:
        """
        Retrieve cached data if valid.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached DataFrame or None if not available/invalid
        """
        if not self.is_cache_valid(cache_key):
            return None
        
        cache_path = self._get_cache_path(cache_key)
        
        try:
            data = pd.read_pickle(cache_path)
            self.logger.info(f"Retrieved cached data: {cache_key}")
            return data
            
        except Exception as e:
            self.logger.warning(f"Error loading cached data: {e}")
            return None
    
    def cache_data(
        self, 
        data: pd.DataFrame, 
        cache_key: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Cache data with metadata.
        
        Args:
            data: DataFrame to cache
            cache_key: Cache key
            metadata: Additional metadata to store
            
        Returns:
            True if caching successful, False otherwise
        """
        try:
            # Save data
            cache_path = self._get_cache_path(cache_key)
            data.to_pickle(cache_path)
            
            # Save metadata
            cache_metadata = {
                'timestamp': datetime.now().isoformat(),
                'rows': len(data),
                'columns': list(data.columns),
                'date_range': {
                    'start': data.index[0].isoformat() if not data.empty else None,
                    'end': data.index[-1].isoformat() if not data.empty else None
                }
            }
            
            if metadata:
                cache_metadata.update(metadata)
            
            metadata_path = self._get_metadata_path(cache_key)
            with open(metadata_path, 'w') as f:
                json.dump(cache_metadata, f, indent=2)
            
            self.logger.info(f"Cached data: {cache_key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error caching data: {e}")
            return False
    
    def clear_cache(self, older_than_hours: Optional[int] = None) -> int:
        """
        Clear cached data.
        
        Args:
            older_than_hours: Only clear cache older than specified hours
            
        Returns:
            Number of cache files cleared
        """
        cleared_count = 0
        
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                should_clear = True
                
                if older_than_hours:
                    file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                    if datetime.now() - file_time < timedelta(hours=older_than_hours):
                        should_clear = False
                
                if should_clear:
                    # Remove data file
                    cache_file.unlink()
                    
                    # Remove metadata file
                    meta_file = cache_file.with_suffix('_meta.json')
                    if meta_file.exists():
                        meta_file.unlink()
                    
                    cleared_count += 1
            
            self.logger.info(f"Cleared {cleared_count} cache files")
            return cleared_count
            
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
            return 0


class DataValidator:
    """
    Validates and cleans market data.
    
    This class provides comprehensive validation and cleaning capabilities
    for market data to ensure data quality and reliability.
    """
    
    def __init__(self) -> None:
        """Initialize the data validator."""
        self.logger = logging.getLogger(__name__)
    
    def validate_ohlcv_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate OHLCV (Open, High, Low, Close, Volume) data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing validation results
        """
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        validation_result = validate_data_integrity(data, required_columns)
        
        if not validation_result['is_valid']:
            return validation_result
        
        # Additional OHLCV-specific validations
        ohlc_errors = []
        
        # Check OHLC relationships
        invalid_high = data['High'] < data[['Open', 'Close']].max(axis=1)
        invalid_low = data['Low'] > data[['Open', 'Close']].min(axis=1)
        invalid_ohlc = data['High'] < data['Low']
        
        if invalid_high.any():
            count = invalid_high.sum()
            ohlc_errors.append(f"High price lower than Open/Close in {count} records")
        
        if invalid_low.any():
            count = invalid_low.sum()
            ohlc_errors.append(f"Low price higher than Open/Close in {count} records")
        
        if invalid_ohlc.any():
            count = invalid_ohlc.sum()
            ohlc_errors.append(f"High price lower than Low price in {count} records")
        
        # Check for negative prices
        negative_prices = (data[['Open', 'High', 'Low', 'Close']] <= 0).any(axis=1)
        if negative_prices.any():
            count = negative_prices.sum()
            ohlc_errors.append(f"Non-positive prices found in {count} records")
        
        # Check for negative volume
        negative_volume = data['Volume'] < 0
        if negative_volume.any():
            count = negative_volume.sum()
            ohlc_errors.append(f"Negative volume found in {count} records")
        
        # Check for extreme price movements (>50% in one day)
        if len(data) > 1:
            price_changes = data['Close'].pct_change().abs()
            extreme_moves = price_changes > 0.5
            if extreme_moves.any():
                count = extreme_moves.sum()
                validation_result['warnings'].append(f"Extreme price movements (>50%) in {count} records")
        
        if ohlc_errors:
            validation_result['is_valid'] = False
            validation_result['errors'].extend(ohlc_errors)
        
        return validation_result
    
    def clean_ohlcv_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Clean OHLCV data by fixing common issues.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Stock symbol for logging
            
        Returns:
            Cleaned DataFrame
        """
        cleaned_data = data.copy()
        original_length = len(cleaned_data)
        
        # Remove rows with invalid OHLC relationships
        invalid_ohlc = cleaned_data['High'] < cleaned_data['Low']
        if invalid_ohlc.any():
            cleaned_data = cleaned_data[~invalid_ohlc]
            self.logger.warning(f"Removed {invalid_ohlc.sum()} rows with invalid OHLC for {symbol}")
        
        # Remove rows with non-positive prices
        negative_prices = (cleaned_data[['Open', 'High', 'Low', 'Close']] <= 0).any(axis=1)
        if negative_prices.any():
            cleaned_data = cleaned_data[~negative_prices]
            self.logger.warning(f"Removed {negative_prices.sum()} rows with non-positive prices for {symbol}")
        
        # Handle missing values
        if cleaned_data.isnull().any().any():
            cleaned_data = cleaned_data.fillna(method='ffill').fillna(method='bfill')
            self.logger.info(f"Forward/backward filled missing values for {symbol}")
        
        # Remove any remaining NaN rows
        cleaned_data = cleaned_data.dropna()
        
        # Sort by date
        cleaned_data = cleaned_data.sort_index()
        
        # Log cleaning summary
        removed_rows = original_length - len(cleaned_data)
        if removed_rows > 0:
            self.logger.info(f"Data cleaning for {symbol}: removed {removed_rows} invalid rows")
        
        return cleaned_data
    
    def detect_outliers(
        self, 
        data: pd.DataFrame, 
        column: str = 'Close',
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> pd.Series:
        """
        Detect outliers in price data.
        
        Args:
            data: DataFrame with price data
            column: Column to analyze for outliers
            method: Outlier detection method ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Boolean Series indicating outliers
        """
        if method == 'iqr':
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            return (data[column] < lower_bound) | (data[column] > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
            return z_scores > threshold
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")


class DataPreprocessor:
    """
    Handles data preprocessing and technical indicator calculations.
    
    This class provides comprehensive preprocessing capabilities including
    technical indicator calculations, feature engineering, and data transformations.
    """
    
    def __init__(self) -> None:
        """Initialize the data preprocessor."""
        self.logger = logging.getLogger(__name__)
    
    def calculate_basic_indicators(
        self, 
        data: pd.DataFrame,
        rsi_period: int = 14,
        ma_short: int = 10,
        ma_long: int = 20
    ) -> pd.DataFrame:
        """
        Calculate basic technical indicators required for trading strategy.
        
        Args:
            data: DataFrame with OHLCV data
            rsi_period: Period for RSI calculation
            ma_short: Period for short moving average
            ma_long: Period for long moving average
            
        Returns:
            DataFrame with basic indicators added
        """
        df = data.copy()
        
        # RSI (Relative Strength Index)
        # RSI = 100 - (100 / (1 + RS))
        # RS = Average Gain / Average Loss over n periods
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Moving Averages
        df['MA_Short'] = df['Close'].rolling(window=ma_short).mean()
        df['MA_Long'] = df['Close'].rolling(window=ma_long).mean()
        
        # Price change percentage
        df['Price_Change_Pct'] = df['Close'].pct_change() * 100
        
        self.logger.info(f"Calculated basic indicators: RSI({rsi_period}), MA({ma_short}, {ma_long})")
        return df
    
    def calculate_advanced_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate advanced technical indicators for comprehensive analysis.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with advanced indicators added
        """
        df = data.copy()
        
        try:
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
            df['ATR'] = ta.volatility.AverageTrueRange(
                df['High'], df['Low'], df['Close']
            ).average_true_range()
            
            # Stochastic Oscillator
            stoch_indicator = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
            df['Stoch_K'] = stoch_indicator.stoch()
            df['Stoch_D'] = stoch_indicator.stoch_signal()
            
            # Volume indicators
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            
            # Volatility
            df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()
            
            self.logger.info("Calculated advanced technical indicators")
            
        except Exception as e:
            self.logger.error(f"Error calculating advanced indicators: {e}")
        
        return df
    
    def prepare_ml_features(self, data: pd.DataFrame, lookback_window: int = 30) -> pd.DataFrame:
        """
        Prepare features for machine learning models.
        
        Args:
            data: DataFrame with price and indicator data
            lookback_window: Number of periods to look back for features
            
        Returns:
            DataFrame with ML features
        """
        df = data.copy()
        
        # Price-based features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'Returns_Mean_{window}'] = df['Returns'].rolling(window).mean()
            df[f'Returns_Std_{window}'] = df['Returns'].rolling(window).std()
            df[f'Price_Min_{window}'] = df['Close'].rolling(window).min()
            df[f'Price_Max_{window}'] = df['Close'].rolling(window).max()
        
        # Lagged features
        for lag in [1, 2, 3, 5]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
            if 'RSI' in df.columns:
                df[f'RSI_Lag_{lag}'] = df['RSI'].shift(lag)
        
        # Target variable (next day return)
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        self.logger.info(f"Prepared ML features with {lookback_window} period lookback")
        return df


class DataFetcher:
    """
    Main class for data fetching operations.
    
    This class provides a unified interface for fetching market data from multiple
    sources including Alpha Vantage and Yahoo Finance, with intelligent caching,
    validation, and preprocessing capabilities.
    
    Attributes:
        config: Configuration manager instance
        cache: Data cache instance
        validator: Data validator instance
        preprocessor: Data preprocessor instance
    
    Example:
        >>> fetcher = DataFetcher()
        >>> data = fetcher.fetch_stock_data('RELIANCE.NS', period='6mo')
        >>> print(f"Fetched {len(data)} records")
    """
    
    def __init__(self, config: Optional[ConfigManager] = None) -> None:
        """
        Initialize the data fetcher.
        
        Args:
            config: Optional configuration manager instance
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or ConfigManager()
        
        # Initialize components
        self.cache = DataCache()
        self.validator = DataValidator()
        self.preprocessor = DataPreprocessor()
        
        # Load configuration
        try:
            if not self.config.settings:
                self.config.load_config()
        except Exception as e:
            self.logger.warning(f"Could not load configuration: {e}")
        
        # Set up HTTP session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    @retry_on_exception(max_retries=3, delay=1.0, exceptions=(requests.RequestException,))
    def _fetch_alpha_vantage_data(
        self, 
        symbol: str,
        function: str = 'TIME_SERIES_DAILY',
        outputsize: str = 'compact'
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data from Alpha Vantage API with retry logic.
        
        Args:
            symbol: Stock symbol
            function: Alpha Vantage function to call
            outputsize: Output size ('compact' or 'full')
            
        Returns:
            DataFrame with stock data or None if failed
        """
        api_config = self.config.get_api_config()
        api_key = api_config.get('alphavantage_api_key')
        
        if not api_key:
            self.logger.error("Alpha Vantage API key not configured")
            return None
        
        url = 'https://www.alphavantage.co/query'
        params = {
            'function': function,
            'symbol': symbol,
            'outputsize': outputsize,
            'apikey': api_key
        }
        
        self.logger.info(f"Fetching Alpha Vantage data for {symbol}")
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Check for API errors
            if "Error Message" in data:
                raise DataFetchError(f"Alpha Vantage Error: {data['Error Message']}")
            
            if "Note" in data:
                self.logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                time.sleep(60)  # Wait for rate limit reset
                return self._fetch_alpha_vantage_data(symbol, function, outputsize)
            
            # Extract time series data
            time_series_key = 'Time Series (Daily)'
            if time_series_key not in data:
                raise DataFetchError(f"Unexpected response structure for {symbol}")
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df.astype(float)
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            
            self.logger.info(f"Successfully fetched {len(df)} records from Alpha Vantage for {symbol}")
            return df
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error fetching {symbol} from Alpha Vantage: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error processing Alpha Vantage data for {symbol}: {e}")
            raise DataFetchError(f"Failed to fetch Alpha Vantage data: {e}")
    
    @retry_on_exception(max_retries=3, delay=1.0, exceptions=(Exception,))
    def _fetch_yfinance_data(
        self, 
        symbol: str,
        period: str = '6mo',
        interval: str = '1d'
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data from Yahoo Finance using yfinance with retry logic.
        
        Args:
            symbol: Stock symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            DataFrame with stock data or None if failed
        """
        self.logger.info(f"Fetching yfinance data for {symbol} (period={period}, interval={interval})")
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                raise DataFetchError(f"No data found for {symbol}")
            
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
            
            self.logger.info(f"Successfully fetched {len(df)} records from yfinance for {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching yfinance data for {symbol}: {e}")
            raise DataFetchError(f"Failed to fetch yfinance data: {e}")
    
    def fetch_stock_data(
        self,
        symbol: str,
        period: str = '6mo',
        interval: str = '1d',
        source: str = 'auto',
        use_cache: bool = True,
        validate: bool = True,
        add_indicators: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Fetch stock data with intelligent source selection and caching.
        
        Args:
            symbol: Stock symbol
            period: Data period
            interval: Data interval
            source: Data source ('yfinance', 'alphavantage', or 'auto')
            use_cache: Whether to use cached data
            validate: Whether to validate data
            add_indicators: Whether to add basic technical indicators
            
        Returns:
            DataFrame with stock data or None if failed
        """
        # Check cache first
        if use_cache:
            cache_key = self.cache._get_cache_key(symbol, period, interval, source)
            cached_data = self.cache.get_cached_data(cache_key)
            if cached_data is not None:
                self.logger.info(f"Using cached data for {symbol}")
                return cached_data
        
        # Fetch fresh data
        data = None
        
        if source == 'auto':
            # Try yfinance first (better for most stocks)
            try:
                data = self._fetch_yfinance_data(symbol, period, interval)
            except DataFetchError:
                self.logger.info(f"yfinance failed for {symbol}, trying Alpha Vantage")
                try:
                    # Convert symbol for Alpha Vantage (remove .NS suffix for Indian stocks)
                    av_symbol = symbol.replace('.NS', '')
                    data = self._fetch_alpha_vantage_data(av_symbol)
                except DataFetchError:
                    self.logger.error(f"Both data sources failed for {symbol}")
        
        elif source == 'yfinance':
            data = self._fetch_yfinance_data(symbol, period, interval)
        
        elif source == 'alphavantage':
            av_symbol = symbol.replace('.NS', '')
            data = self._fetch_alpha_vantage_data(av_symbol)
        
        else:
            raise ValueError(f"Unknown data source: {source}")
        
        if data is None:
            return None
        
        # Validate data
        if validate:
            validation_result = self.validator.validate_ohlcv_data(data)
            if not validation_result['is_valid']:
                self.logger.warning(f"Data validation failed for {symbol}: {validation_result['errors']}")
                data = self.validator.clean_ohlcv_data(data, symbol)
        
        # Add basic indicators
        if add_indicators:
            trading_config = self.config.get_trading_config()
            data = self.preprocessor.calculate_basic_indicators(
                data,
                rsi_period=trading_config.get('rsi_period', 14),
                ma_short=trading_config.get('ma_short_period', 10),
                ma_long=trading_config.get('ma_long_period', 20)
            )
        
        # Cache the data
        if use_cache and data is not None:
            cache_key = self.cache._get_cache_key(symbol, period, interval, source)
            self.cache.cache_data(data, cache_key, {'symbol': symbol, 'source': source})
        
        return data
    
    def fetch_multiple_stocks(
        self,
        symbols: List[str],
        period: str = '6mo',
        interval: str = '1d',
        source: str = 'auto',
        use_cache: bool = True,
        validate: bool = True,
        add_indicators: bool = True,
        delay_between_requests: float = 1.0
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks with rate limiting.
        
        Args:
            symbols: List of stock symbols
            period: Data period
            interval: Data interval
            source: Data source
            use_cache: Whether to use cached data
            validate: Whether to validate data
            add_indicators: Whether to add basic technical indicators
            delay_between_requests: Delay between requests in seconds
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        self.logger.info(f"Fetching data for {len(symbols)} stocks")
        
        stock_data = {}
        failed_stocks = []
        
        for i, symbol in enumerate(symbols, 1):
            self.logger.info(f"[{i}/{len(symbols)}] Processing {symbol}")
            
            try:
                data = self.fetch_stock_data(
                    symbol=symbol,
                    period=period,
                    interval=interval,
                    source=source,
                    use_cache=use_cache,
                    validate=validate,
                    add_indicators=add_indicators
                )
                
                if data is not None:
                    stock_data[symbol] = data
                    self.logger.info(f"Successfully processed {symbol}: {len(data)} records")
                else:
                    failed_stocks.append(symbol)
                    self.logger.error(f"Failed to fetch data for {symbol}")
                
                # Rate limiting
                if i < len(symbols) and delay_between_requests > 0:
                    time.sleep(delay_between_requests)
                    
            except Exception as e:
                failed_stocks.append(symbol)
                self.logger.error(f"Error processing {symbol}: {e}")
        
        # Summary
        self.logger.info(f"Data fetching completed: {len(stock_data)} successful, {len(failed_stocks)} failed")
        if failed_stocks:
            self.logger.warning(f"Failed stocks: {', '.join(failed_stocks)}")
        
        return stock_data
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get the latest price for a stock.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Latest price or None if failed
        """
        try:
            data = self.fetch_stock_data(
                symbol=symbol,
                period='1d',
                interval='1m',
                use_cache=False,
                add_indicators=False
            )
            
            if data is not None and not data.empty:
                return float(data['Close'].iloc[-1])
            
        except Exception as e:
            self.logger.error(f"Error getting latest price for {symbol}: {e}")
        
        return None
    
    def get_data_summary(self, stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate a comprehensive summary of fetched data.
        
        Args:
            stock_data: Dictionary mapping symbols to DataFrames
            
        Returns:
            DataFrame containing data summary
        """
        summary_data = []
        
        for symbol, df in stock_data.items():
            if df.empty:
                continue
            
            # Calculate summary statistics
            total_return = ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100
            volatility = df['Close'].pct_change().std() * np.sqrt(252) * 100  # Annualized
            
            summary = {
                'Symbol': symbol,
                'Records': len(df),
                'Start_Date': df.index[0].strftime('%Y-%m-%d'),
                'End_Date': df.index[-1].strftime('%Y-%m-%d'),
                'Latest_Price': format_currency(df['Close'].iloc[-1]),
                'Price_Range': f"{format_currency(df['Low'].min())} - {format_currency(df['High'].max())}",
                'Total_Return': format_percentage(total_return / 100),
                'Volatility': format_percentage(volatility / 100),
                'Avg_Volume': f"{df['Volume'].mean():,.0f}",
                'Data_Quality': 'Good' if len(df) > 100 else 'Limited'
            }
            summary_data.append(summary)
        
        return pd.DataFrame(summary_data)
    
    def save_data(self, data: pd.DataFrame, symbol: str, data_type: str = 'processed') -> bool:
        """
        Save data to CSV file.
        
        Args:
            data: DataFrame to save
            symbol: Stock symbol
            data_type: Type of data ('raw' or 'processed')
            
        Returns:
            True if save successful, False otherwise
        """
        try:
            ensure_directory_exists('data')
            filename = f"data/{symbol.replace('.', '_')}_{data_type}_data.csv"
            data.to_csv(filename)
            self.logger.info(f"Saved {data_type} data to {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving data for {symbol}: {e}")
            return False
    
    def load_saved_data(self, symbol: str, data_type: str = 'processed') -> Optional[pd.DataFrame]:
        """
        Load previously saved data.
        
        Args:
            symbol: Stock symbol
            data_type: Type of data ('raw' or 'processed')
            
        Returns:
            DataFrame with loaded data or None if not found
        """
        try:
            filename = f"data/{symbol.replace('.', '_')}_{data_type}_data.csv"
            if Path(filename).exists():
                df = pd.read_csv(filename, index_col=0, parse_dates=True)
                self.logger.info(f"Loaded saved data from {filename}")
                return df
            else:
                self.logger.warning(f"No saved data found: {filename}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error loading saved data for {symbol}: {e}")
            return None