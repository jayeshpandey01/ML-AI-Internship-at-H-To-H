"""
Utility Functions Module

This module provides common utility functions used throughout the algo trading system.
It includes logging setup, data validation, mathematical utilities, and formatting helpers.

Functions:
    setup_logging: Configure system logging
    validate_data_integrity: Validate market data integrity
    calculate_returns: Calculate financial returns
    format_currency: Format currency values
    format_percentage: Format percentage values
    ensure_directory_exists: Create directory if it doesn't exist

Example:
    >>> from src.utils import setup_logging, format_currency
    >>> setup_logging()
    >>> formatted_price = format_currency(1234.56)
    >>> print(formatted_price)  # ₹1,234.56
"""

import os
import logging
import logging.config
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def setup_logging(
    config_path: str = "config/logging.conf",
    log_level: Optional[str] = None,
    log_dir: str = "logs"
) -> None:
    """
    Set up logging configuration for the application.
    
    Args:
        config_path: Path to logging configuration file
        log_level: Override log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
        
    Example:
        >>> setup_logging(log_level="DEBUG")
    """
    # Ensure log directory exists
    ensure_directory_exists(log_dir)
    
    # Use configuration file if it exists
    if Path(config_path).exists():
        try:
            logging.config.fileConfig(config_path)
        except Exception as e:
            # Fallback to basic configuration
            _setup_basic_logging(log_level, log_dir)
            logging.error(f"Error loading logging config from {config_path}: {e}")
    else:
        # Use basic configuration
        _setup_basic_logging(log_level, log_dir)
    
    # Override log level if specified
    if log_level:
        numeric_level = getattr(logging, log_level.upper(), None)
        if isinstance(numeric_level, int):
            logging.getLogger().setLevel(numeric_level)
        else:
            logging.warning(f"Invalid log level: {log_level}")


def _setup_basic_logging(log_level: Optional[str] = None, log_dir: str = "logs") -> None:
    """
    Set up basic logging configuration as fallback.
    
    Args:
        log_level: Log level to use
        log_dir: Directory to store log files
    """
    level = getattr(logging, (log_level or "INFO").upper(), logging.INFO)
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            f"{log_dir}/algo_trading.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        logging.error(f"Could not create file handler: {e}")


def ensure_directory_exists(directory: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        directory: Directory path to create
        
    Example:
        >>> ensure_directory_exists("logs")
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def validate_data_integrity(
    data: pd.DataFrame,
    required_columns: List[str],
    min_rows: int = 1
) -> Dict[str, Any]:
    """
    Validate market data integrity.
    
    Args:
        data: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        
    Returns:
        Dictionary containing validation results
        
    Example:
        >>> data = pd.DataFrame({'Open': [100, 101], 'Close': [101, 102]})
        >>> result = validate_data_integrity(data, ['Open', 'Close'])
        >>> print(result['is_valid'])  # True
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'row_count': len(data),
        'column_count': len(data.columns),
        'missing_values': {},
        'data_types': {}
    }
    
    # Check if DataFrame is empty
    if data.empty:
        validation_result['is_valid'] = False
        validation_result['errors'].append("DataFrame is empty")
        return validation_result
    
    # Check minimum rows
    if len(data) < min_rows:
        validation_result['is_valid'] = False
        validation_result['errors'].append(f"Insufficient data: {len(data)} rows < {min_rows} required")
    
    # Check required columns
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        validation_result['is_valid'] = False
        validation_result['errors'].append(f"Missing required columns: {missing_columns}")
    
    # Check for missing values
    for column in data.columns:
        missing_count = data[column].isnull().sum()
        if missing_count > 0:
            validation_result['missing_values'][column] = missing_count
            if missing_count / len(data) > 0.1:  # More than 10% missing
                validation_result['warnings'].append(f"High missing values in {column}: {missing_count}")
    
    # Check data types
    for column in data.columns:
        validation_result['data_types'][column] = str(data[column].dtype)
    
    # Check for duplicate indices
    if data.index.duplicated().any():
        validation_result['warnings'].append("Duplicate indices found")
    
    # Check for negative prices (if price columns exist)
    price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
    for col in price_columns:
        if col in data.columns and (data[col] <= 0).any():
            validation_result['warnings'].append(f"Non-positive values found in {col}")
    
    return validation_result


def calculate_returns(
    prices: Union[pd.Series, np.ndarray],
    method: str = 'simple'
) -> Union[pd.Series, np.ndarray]:
    """
    Calculate financial returns from price series.
    
    Args:
        prices: Price series
        method: Return calculation method ('simple' or 'log')
        
    Returns:
        Returns series
        
    Example:
        >>> prices = pd.Series([100, 105, 102, 108])
        >>> returns = calculate_returns(prices)
        >>> print(returns.iloc[1])  # 0.05 (5% return)
    """
    if method == 'simple':
        if isinstance(prices, pd.Series):
            return prices.pct_change()
        else:
            return np.diff(prices) / prices[:-1]
    elif method == 'log':
        if isinstance(prices, pd.Series):
            return np.log(prices / prices.shift(1))
        else:
            return np.log(prices[1:] / prices[:-1])
    else:
        raise ValueError(f"Unknown return calculation method: {method}")


def calculate_volatility(
    returns: Union[pd.Series, np.ndarray],
    window: Optional[int] = None,
    annualize: bool = True
) -> Union[float, pd.Series]:
    """
    Calculate volatility from returns.
    
    Args:
        returns: Returns series
        window: Rolling window size (None for full period)
        annualize: Whether to annualize volatility
        
    Returns:
        Volatility value or series
        
    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.015, -0.01])
        >>> vol = calculate_volatility(returns)
        >>> print(f"Annualized volatility: {vol:.2%}")
    """
    if isinstance(returns, pd.Series):
        if window:
            vol = returns.rolling(window=window).std()
        else:
            vol = returns.std()
    else:
        vol = np.std(returns)
    
    if annualize:
        # Assume daily returns, annualize with sqrt(252)
        vol = vol * np.sqrt(252)
    
    return vol


def format_currency(
    amount: float,
    currency_symbol: str = "₹",
    decimal_places: int = 2
) -> str:
    """
    Format currency values with proper formatting.
    
    Args:
        amount: Amount to format
        currency_symbol: Currency symbol to use
        decimal_places: Number of decimal places
        
    Returns:
        Formatted currency string
        
    Example:
        >>> formatted = format_currency(1234.56)
        >>> print(formatted)  # ₹1,234.56
    """
    return f"{currency_symbol}{amount:,.{decimal_places}f}"


def format_percentage(
    value: float,
    decimal_places: int = 2,
    include_sign: bool = True
) -> str:
    """
    Format percentage values.
    
    Args:
        value: Percentage value (as decimal, e.g., 0.05 for 5%)
        decimal_places: Number of decimal places
        include_sign: Whether to include + sign for positive values
        
    Returns:
        Formatted percentage string
        
    Example:
        >>> formatted = format_percentage(0.0523)
        >>> print(formatted)  # +5.23%
    """
    percentage = value * 100
    sign = "+" if include_sign and percentage > 0 else ""
    return f"{sign}{percentage:.{decimal_places}f}%"


def format_number(
    number: float,
    decimal_places: int = 2,
    use_thousands_separator: bool = True
) -> str:
    """
    Format numbers with proper formatting.
    
    Args:
        number: Number to format
        decimal_places: Number of decimal places
        use_thousands_separator: Whether to use thousands separator
        
    Returns:
        Formatted number string
        
    Example:
        >>> formatted = format_number(1234567.89)
        >>> print(formatted)  # 1,234,567.89
    """
    if use_thousands_separator:
        return f"{number:,.{decimal_places}f}"
    else:
        return f"{number:.{decimal_places}f}"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, handling division by zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value to return if division by zero
        
    Returns:
        Division result or default value
        
    Example:
        >>> result = safe_divide(10, 0, default=float('inf'))
        >>> print(result)  # inf
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (ZeroDivisionError, TypeError):
        return default


def calculate_sharpe_ratio(
    returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio from returns.
    
    Args:
        returns: Returns series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year (252 for daily)
        
    Returns:
        Sharpe ratio
        
    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.02])
        >>> sharpe = calculate_sharpe_ratio(returns)
        >>> print(f"Sharpe ratio: {sharpe:.3f}")
    """
    if isinstance(returns, pd.Series):
        excess_returns = returns - (risk_free_rate / periods_per_year)
        return safe_divide(
            excess_returns.mean() * periods_per_year,
            excess_returns.std() * np.sqrt(periods_per_year)
        )
    else:
        excess_returns = returns - (risk_free_rate / periods_per_year)
        return safe_divide(
            np.mean(excess_returns) * periods_per_year,
            np.std(excess_returns) * np.sqrt(periods_per_year)
        )


def calculate_max_drawdown(prices: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
    """
    Calculate maximum drawdown from price series.
    
    Args:
        prices: Price series
        
    Returns:
        Dictionary containing max drawdown information
        
    Example:
        >>> prices = pd.Series([100, 110, 95, 105, 90, 120])
        >>> dd_info = calculate_max_drawdown(prices)
        >>> print(f"Max drawdown: {dd_info['max_drawdown']:.2%}")
    """
    if isinstance(prices, pd.Series):
        cumulative_max = prices.expanding().max()
        drawdown = (prices - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()
        max_drawdown_date = drawdown.idxmin() if hasattr(drawdown, 'idxmin') else None
    else:
        cumulative_max = np.maximum.accumulate(prices)
        drawdown = (prices - cumulative_max) / cumulative_max
        max_drawdown = np.min(drawdown)
        max_drawdown_date = np.argmin(drawdown)
    
    return {
        'max_drawdown': max_drawdown,
        'max_drawdown_date': max_drawdown_date,
        'drawdown_series': drawdown
    }


def get_business_days_between(start_date: datetime, end_date: datetime) -> int:
    """
    Calculate number of business days between two dates.
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        Number of business days
        
    Example:
        >>> from datetime import datetime
        >>> start = datetime(2024, 1, 1)
        >>> end = datetime(2024, 1, 10)
        >>> days = get_business_days_between(start, end)
        >>> print(f"Business days: {days}")
    """
    return pd.bdate_range(start=start_date, end=end_date).size


def create_date_range(
    start_date: str,
    end_date: str,
    freq: str = 'D'
) -> pd.DatetimeIndex:
    """
    Create a date range with specified frequency.
    
    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        freq: Frequency string ('D' for daily, 'B' for business days)
        
    Returns:
        DatetimeIndex with specified range
        
    Example:
        >>> date_range = create_date_range('2024-01-01', '2024-01-10', 'B')
        >>> print(len(date_range))  # Number of business days
    """
    return pd.date_range(start=start_date, end=end_date, freq=freq)


def load_json_config(file_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        file_path: Path to JSON configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON is invalid
        
    Example:
        >>> config = load_json_config('config/settings.json')
        >>> print(config.get('api_key'))
    """
    import json
    
    config_path = Path(file_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")


def save_json_config(config: Dict[str, Any], file_path: str) -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        file_path: Path to save JSON file
        
    Example:
        >>> config = {'api_key': 'your_key', 'timeout': 30}
        >>> save_json_config(config, 'config/settings.json')
    """
    import json
    
    config_path = Path(file_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)


def retry_on_exception(
    func,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator to retry function on exception.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        delay: Initial delay between retries
        backoff_factor: Multiplier for delay on each retry
        exceptions: Tuple of exceptions to catch
        
    Returns:
        Decorated function
        
    Example:
        >>> @retry_on_exception(max_retries=3, delay=1.0)
        ... def unreliable_function():
        ...     # Function that might fail
        ...     pass
    """
    import time
    import functools
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logging.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {current_delay} seconds..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logging.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
                        raise last_exception
            
            return None
        return wrapper
    return decorator