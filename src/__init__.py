"""
Algo Trading System

A comprehensive algorithmic trading system with machine learning predictions,
automated backtesting, Google Sheets integration, and Telegram alerts.

Author: Your Name
Version: 2.0.0
License: MIT
"""

__version__ = "2.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main classes for easy access
from .main import AlgoTradingSystem
from .data import DataFetcher
from .strategy import RSIMACrossoverStrategy
from .ml_model import RandomForestPredictor
from .sheets import SheetsManager
from .telegram import TelegramNotifier

__all__ = [
    "AlgoTradingSystem",
    "DataFetcher", 
    "RSIMACrossoverStrategy",
    "RandomForestPredictor",
    "SheetsManager",
    "TelegramNotifier"
]