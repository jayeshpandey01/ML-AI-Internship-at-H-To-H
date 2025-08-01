# 🚀 Algo-Trading System

A comprehensive, production-ready algorithmic trading system with machine learning predictions, automated backtesting, Google Sheets integration, and real-time Telegram alerts.

## ✨ Features

### 📊 **Data Management**
- **Multi-Source Data Fetching**: Alpha Vantage & Yahoo Finance with intelligent fallback
- **Smart Caching System**: Reduces API calls and improves performance
- **Data Validation & Cleaning**: Ensures data quality and integrity
- **Technical Indicators**: 20+ built-in indicators (RSI, MACD, Bollinger Bands, etc.)

### 🎯 **Trading Strategy**
- **RSI + Moving Average Crossover**: Proven technical analysis strategy
- **Comprehensive Backtesting**: Historical performance analysis with detailed metrics
- **Risk Management**: Configurable parameters and position sizing
- **Performance Analytics**: Sharpe ratio, win rate, drawdown analysis

### 🤖 **Machine Learning**
- **Random Forest Predictor**: Next-day price movement predictions
- **Feature Engineering**: 30+ technical and statistical features
- **Model Validation**: Cross-validation and performance metrics
- **Confidence Scoring**: Prediction confidence levels

### 📈 **Automation & Integration**
- **Google Sheets Logging**: Automated trade logs and performance reports
- **Telegram Alerts**: Real-time notifications for trades, errors, and system status
- **System Monitoring**: Health checks and performance tracking
- **Configuration Management**: Centralized, validated configuration system

### 🏗️ **Professional Architecture**
- **Modular Design**: Clean separation of concerns
- **Type Hints**: Full type annotation for better code quality
- **Comprehensive Logging**: Detailed logging with configurable levels
- **Error Handling**: Robust error recovery and retry mechanisms
- **Testing Suite**: Unit and integration tests
- **Code Quality**: PEP 8 compliant with automated formatting

## 📋 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: 1GB free space for data and logs

### API Requirements
- **Alpha Vantage API Key**: [Get free key](https://www.alphavantage.co/support/#api-key) (optional but recommended)
- **Google Sheets API**: Service account credentials for automated logging
- **Telegram Bot Token**: For real-time alerts (optional)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd algo-trading-system

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt

# Set up pre-commit hooks (optional)
pre-commit install
```

### 2. Configuration Setup

#### 2.1 Basic Configuration
```bash
# Copy example configuration
cp config/config.example.env config/config.env

# Edit configuration with your settings
# Update API keys, stock symbols, and trading parameters
```

#### 2.2 Alpha Vantage API (Recommended)
1. Visit [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
2. Get your free API key
3. Update `ALPHAVANTAGE_API_KEY` in `config/config.env`

#### 2.3 Google Sheets Integration
1. **Create Google Sheet**:
   - Go to [Google Sheets](https://sheets.google.com)
   - Create a new spreadsheet
   - Copy the Sheet ID from the URL

2. **Enable Google Sheets API**:
   - Visit [Google Cloud Console](https://console.cloud.google.com)
   - Create a new project or select existing
   - Enable Google Sheets API
   - Create service account credentials
   - Download JSON credentials file

3. **Configure**:
   ```bash
   # Place credentials file
   mv /path/to/credentials.json config/google_credentials.json
   
   # Update config
   GOOGLE_SHEET_ID=your_sheet_id_here
   ```

#### 2.4 Telegram Alerts (Optional)
1. **Create Telegram Bot**:
   - Message @BotFather on Telegram
   - Create new bot with `/newbot`
   - Save the bot token

2. **Get Chat ID**:
   ```bash
   # Update bot token in config first
   python get_chat_id.py
   ```

3. **Test Alerts**:
   ```bash
   python src/telegram.py
   ```

### 3. Run the System

#### Basic Analysis
```bash
# Run complete analysis
python src/main.py

# Run with Telegram alerts
python src/main_with_telegram.py
```

#### Advanced Usage
```bash
# Run specific components
python -m src.data          # Data fetching only
python -m src.strategy      # Strategy backtesting only
python -m src.ml_model      # ML model training only

# Run tests
pytest tests/

# Check code quality
flake8 src/
black src/
mypy src/
```

## 📊 Strategy Details

### RSI + Moving Average Crossover

**Buy Signal**:
- RSI < 30 (oversold)
- Short MA (10) crosses above Long MA (20)

**Sell Signal**:
- RSI > 70 (overbought)
- Short MA (10) crosses below Long MA (20)

### Machine Learning Features

The ML model uses the following features:
- Technical indicators (RSI, MACD, Bollinger Bands, ATR)
- Price ratios and changes
- Moving averages (5, 10, 20 periods)
- Lag features (1, 2, 3, 5 periods)
- Volume indicators

## 📈 Output

The system generates:

1. **Console Output**: Real-time analysis progress and summary
2. **Google Sheets Logs**:
   - Trade Log: Individual trade records
   - P&L Summary: Performance metrics per stock
   - Analytics: Technical indicators and predictions

## 🏗️ Project Architecture

### Directory Structure
```
algo-trading-system/
├── src/                           # Source code
│   ├── __init__.py               # Package initialization
│   ├── main.py                   # Main orchestration script
│   ├── data.py                   # Unified data management
│   ├── strategy.py               # Trading strategy implementation
│   ├── ml_model.py               # Machine learning models
│   ├── sheets.py                 # Google Sheets integration
│   ├── telegram.py               # Telegram alerts
│   ├── config.py                 # Configuration management
│   └── utils.py                  # Utility functions
├── tests/                         # Test suite
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_strategy.py
│   ├── test_ml_model.py
│   ├── test_sheets.py
│   ├── test_telegram.py
│   └── test_integration.py
├── config/                        # Configuration files
│   ├── config.env                # Main configuration
│   ├── config.example.env        # Configuration template
│   ├── logging.conf              # Logging configuration
│   └── google_credentials.json   # Google API credentials
├── docs/                          # Documentation
│   ├── README.md                 # This file
│   ├── API_SETUP.md              # API setup guide
│   ├── ARCHITECTURE.md           # System architecture
│   └── TROUBLESHOOTING.md        # Common issues
├── data/                          # Data storage
│   ├── cache/                    # Cached API responses
│   └── exports/                  # Exported data files
├── logs/                          # Log files
├── requirements.txt               # Production dependencies
├── requirements-dev.txt           # Development dependencies
├── setup.py                       # Package setup
├── pyproject.toml                # Tool configuration
├── .flake8                       # Linting configuration
├── .pre-commit-config.yaml       # Pre-commit hooks
└── .gitignore                    # Git ignore rules
```

### System Components

#### 🔄 **Data Layer** (`src/data.py`)
- **DataFetcher**: Multi-source data acquisition
- **DataCache**: Intelligent caching system
- **DataValidator**: Data quality assurance
- **DataPreprocessor**: Technical indicator calculations

#### 📈 **Strategy Layer** (`src/strategy.py`)
- **TradingStrategy**: Base strategy interface
- **RSIMACrossoverStrategy**: Main trading strategy
- **Backtester**: Historical performance testing
- **PerformanceAnalyzer**: Metrics calculation

#### 🤖 **ML Layer** (`src/ml_model.py`)
- **MLPredictor**: Base ML interface
- **RandomForestPredictor**: Price prediction model
- **FeatureEngineer**: Feature creation and selection
- **ModelValidator**: Model performance evaluation

#### 🔗 **Integration Layer**
- **SheetsManager** (`src/sheets.py`): Google Sheets automation
- **TelegramNotifier** (`src/telegram.py`): Real-time alerts
- **ConfigManager** (`src/config.py`): Configuration management

#### 🎛️ **Control Layer** (`src/main.py`)
- **AlgoTradingSystem**: Main system orchestrator
- **SystemMonitor**: Health and performance monitoring
- **TaskScheduler**: Automated task management

## ⚙️ Configuration Guide

### Core Configuration (`config/config.env`)

#### Trading Strategy Parameters
```env
# RSI Settings
RSI_PERIOD=14                    # RSI calculation period
RSI_OVERSOLD=30                  # Oversold threshold
RSI_OVERBOUGHT=70                # Overbought threshold

# Moving Average Settings
MA_SHORT_PERIOD=10               # Short MA period
MA_LONG_PERIOD=20                # Long MA period

# Stock Selection
NIFTY_STOCKS=RELIANCE.NS,HDFCBANK.NS,INFY.NS,TCS.NS,ICICIBANK.NS
```

#### API Configuration
```env
# Data Sources
ALPHAVANTAGE_API_KEY=your_api_key_here

# Google Sheets
GOOGLE_SHEETS_CREDENTIALS_PATH=config/google_credentials.json
GOOGLE_SHEET_ID=your_sheet_id_here

# Telegram Alerts
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

#### Alert Settings
```env
# Enable/Disable Alert Types
TELEGRAM_ENABLE_TRADE_ALERTS=true
TELEGRAM_ENABLE_ERROR_ALERTS=true
TELEGRAM_ENABLE_STATUS_ALERTS=true
TELEGRAM_ENABLE_ML_ALERTS=true
```

#### Advanced Settings
```env
# System Configuration
LOG_LEVEL=INFO
ENABLE_DATA_CACHE=true
CACHE_EXPIRY_MINUTES=60
MAX_CONCURRENT_REQUESTS=5

# ML Model Settings
ML_MODEL_TYPE=RandomForest
ML_FEATURE_WINDOW=30
ML_TRAIN_TEST_SPLIT=0.8

# Risk Management
MAX_POSITION_SIZE=0.1
STOP_LOSS_PERCENTAGE=0.05
TAKE_PROFIT_PERCENTAGE=0.15
```

## 📊 Performance Metrics

The system calculates:
- **Total Return**: Strategy vs market performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Win Rate**: Percentage of profitable trades
- **ML Accuracy**: Prediction model performance

## 🔧 Customization

### Adding New Stocks
Update `NIFTY_STOCKS` in `config/config.env`:
```env
NIFTY_STOCKS=RELIANCE.NS,TCS.NS,HDFCBANK.NS,INFY.NS,HINDUNILVR.NS
```

### Modifying Strategy Parameters
Adjust RSI and MA periods in the configuration file or directly in the `TradingStrategy` class.

### Extending ML Features
Add new technical indicators in the `MLPredictor.create_features()` method.

## 📊 System Output Examples

### Console Output
```
🚀 ENHANCED ALGO-TRADING SYSTEM WITH TELEGRAM ALERTS
============================================================
Analyzing stocks: RELIANCE.NS, HDFCBANK.NS, INFY.NS
Analysis started at: 2024-01-15 14:00:00

📱 Telegram Alerts: ✅ Enabled
   Trade Alerts: ✅  Error Alerts: ✅  Status Alerts: ✅  ML Alerts: ✅

========================================
ANALYZING RELIANCE.NS
========================================
1. Fetching data for RELIANCE.NS...
✅ Fetched 126 records for RELIANCE.NS
   Date range: 2023-07-15 to 2024-01-15

2. Running trading strategy for RELIANCE.NS...
✅ Strategy analysis complete:
   - Total Return: 12.45%
   - Market Return: 8.32%
   - Win Rate: 65.2%
   - Total Trades: 23
   - Sharpe Ratio: 1.234
   - Max Drawdown: -5.67%

3. Training ML model for RELIANCE.NS...
✅ ML model trained with accuracy: 0.678
   - Features used: 28
   - Cross-validation score: 0.645
   - Next day prediction: UP (confidence: 72.3%)

4. Logging results to Google Sheets...
✅ Results logged to Google Sheets

========================================
ENHANCED ANALYSIS SUMMARY
========================================
Stock        Return     Win Rate   Trades   ML Acc   Next Day  
------------------------------------------------------------
RELIANCE.NS  +12.45%    65.2%      23       0.678    UP        
HDFCBANK.NS  +8.73%     58.8%      17       0.692    DOWN      
INFY.NS      +15.21%    71.4%      21       0.634    UP        

========================================
OVERALL PERFORMANCE
========================================
Total Trades: 61
Total P&L: ₹+7,152.20
Overall Win Rate: 65.0%
Average ML Accuracy: 66.8%
Analysis Duration: 45.2 seconds
Best Trade: ₹+840.30
Worst Trade: ₹-127.45

✅ Analysis complete! 📱 Telegram alerts sent!
📊 Check Google Sheets for detailed logs.
```

### Telegram Alert Examples

#### 🟢 Buy Signal Alert
```
🟢 TRADING SIGNAL 🟢

📊 Stock: RELIANCE.NS
🎯 Action: BUY
💰 Price: ₹1,450.75
📈 Reason: RSI Oversold (32.1) + Bullish MA Crossover
🤖 ML Confidence: 85.0%
⏰ Time: 2024-01-15 14:30:25
```

#### 📈 Performance Summary
```
📈 PERFORMANCE SUMMARY 📈

🎯 Total Trades: 7
💰 Total P&L: ₹+2,930.47
📊 Win Rate: 85.7%
🤖 ML Accuracy: 66.8%
⏱️ Duration: 45.2s
⏰ Time: 2024-01-15 14:40:15

🏆 Best Trade: ₹+840.30
📉 Worst Trade: ₹-127.45
```

### Google Sheets Output

The system automatically creates and updates sheets with:

1. **Trade Log**: Individual trade records with entry/exit prices, P&L, and signals
2. **P&L Summary**: Performance metrics by stock and overall portfolio
3. **ML Predictions**: Model predictions with confidence scores and accuracy
4. **Technical Analysis**: RSI, MA, and other indicator values
5. **System Logs**: Execution logs and error tracking

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test modules
pytest tests/test_data.py
pytest tests/test_strategy.py
pytest tests/test_ml_model.py

# Run integration tests
pytest tests/test_integration.py -v
```

### Code Quality Checks
```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/

# Security scan
bandit -r src/
```

## 🔧 Customization Guide

### Adding New Stocks
```env
# Update in config/config.env
NIFTY_STOCKS=RELIANCE.NS,TCS.NS,HDFCBANK.NS,INFY.NS,HINDUNILVR.NS,ITC.NS
```

### Creating Custom Strategies
```python
# src/custom_strategy.py
from src.strategy import TradingStrategy

class MyCustomStrategy(TradingStrategy):
    def generate_signals(self, data):
        # Implement your custom logic
        pass
```

### Adding New Technical Indicators
```python
# In src/data.py DataPreprocessor class
def calculate_custom_indicators(self, data):
    # Add your custom indicators
    data['Custom_Indicator'] = your_calculation(data)
    return data
```

### Extending ML Features
```python
# In src/ml_model.py
def create_custom_features(self, data):
    # Add new features for ML model
    data['New_Feature'] = feature_calculation(data)
    return data
```

## 🚨 Important Notes & Disclaimers

### ⚠️ **Risk Warnings**
- **Educational Purpose**: This system is designed for learning and research
- **No Financial Advice**: Not intended as investment advice
- **Paper Trading**: Test thoroughly before any real trading
- **Market Risk**: Past performance doesn't guarantee future results

### 🔒 **Security Considerations**
- **API Keys**: Keep your API keys secure and never commit them to version control
- **Credentials**: Store Google credentials safely and restrict access
- **Environment**: Use environment variables for sensitive data
- **Network**: Ensure secure network connections for API calls

### 📊 **Data & Performance**
- **API Limits**: Respect rate limits (Alpha Vantage: 5 calls/minute free tier)
- **Data Quality**: Validate data before making decisions
- **Internet Connection**: Stable connection required for real-time data
- **System Resources**: Monitor memory usage with large datasets

### 🔧 **Technical Limitations**
- **Backtesting Bias**: Historical performance may not reflect future results
- **Market Conditions**: Strategy performance varies with market conditions
- **Slippage**: Real trading involves costs not reflected in backtests
- **Latency**: Real-time execution may differ from backtested results

## 🆘 Troubleshooting

### Common Issues

#### Data Fetching Problems
```bash
# Check API key configuration
python -c "from src.config import ConfigManager; c=ConfigManager(); c.load_config(); print(c.get_api_config())"

# Test data fetching
python -c "from src.data import DataFetcher; f=DataFetcher(); data=f.fetch_stock_data('RELIANCE.NS'); print(len(data) if data is not None else 'Failed')"
```

#### Google Sheets Issues
```bash
# Verify credentials
python -c "from src.sheets import SheetsManager; s=SheetsManager(); print(s.setup_connection())"
```

#### Telegram Alerts Issues
```bash
# Test Telegram connection
python src/telegram.py
```

### Getting Help
1. Check the [Troubleshooting Guide](docs/TROUBLESHOOTING.md)
2. Review log files in the `logs/` directory
3. Run diagnostic scripts in the `tests/` directory
4. Check configuration with validation tools

## 🔮 Roadmap & Future Enhancements

### 🎯 **Planned Features**
- [ ] **Multi-timeframe Analysis**: Support for different time intervals
- [ ] **Portfolio Optimization**: Modern portfolio theory implementation
- [ ] **Sentiment Analysis**: News and social media sentiment integration
- [ ] **Real-time Execution**: Live trading capabilities with broker APIs
- [ ] **Advanced Risk Management**: Position sizing, stop-loss, take-profit
- [ ] **Strategy Optimization**: Genetic algorithms for parameter tuning

### 🚀 **Advanced Integrations**
- [ ] **Database Support**: PostgreSQL/MongoDB for data storage
- [ ] **Web Dashboard**: Real-time monitoring interface
- [ ] **Mobile App**: iOS/Android companion app
- [ ] **Cloud Deployment**: AWS/GCP deployment options
- [ ] **API Service**: RESTful API for external integrations
- [ ] **Webhook Support**: Real-time event notifications

### 📈 **Enhanced Analytics**
- [ ] **Performance Attribution**: Factor-based performance analysis
- [ ] **Risk Metrics**: VaR, CVaR, and other risk measures
- [ ] **Correlation Analysis**: Cross-asset correlation studies
- [ ] **Regime Detection**: Market regime identification
- [ ] **Stress Testing**: Portfolio stress testing capabilities

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Educational Use**: This project is primarily intended for educational and research purposes.

## 🤝 Contributing

We welcome contributions! Please follow these steps:

### Development Setup
```bash
# Fork and clone the repository
git clone https://github.com/yourusername/algo-trading-system.git
cd algo-trading-system

# Create development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install
```

### Contribution Guidelines
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Write** tests for your changes
4. **Ensure** all tests pass (`pytest`)
5. **Check** code quality (`flake8`, `black`, `mypy`)
6. **Commit** your changes (`git commit -m 'Add amazing feature'`)
7. **Push** to the branch (`git push origin feature/amazing-feature`)
8. **Open** a Pull Request

### Code Standards
- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed

## 📚 Documentation

- **[API Setup Guide](docs/API_SETUP.md)**: Detailed API configuration instructions
- **[Architecture Guide](docs/ARCHITECTURE.md)**: System design and component details
- **[Troubleshooting Guide](docs/TROUBLESHOOTING.md)**: Common issues and solutions
- **[Development Guide](docs/DEVELOPMENT.md)**: Contributing and development setup

## 🙏 Acknowledgments

- **Alpha Vantage**: For providing free stock market data API
- **Yahoo Finance**: For reliable financial data through yfinance
- **Google Sheets API**: For seamless data logging and reporting
- **Telegram Bot API**: For real-time alert capabilities
- **Open Source Community**: For the amazing Python libraries that make this possible

## 📞 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/algo-trading-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/algo-trading-system/discussions)
- **Email**: your.email@example.com

## ⭐ Star History

If you find this project helpful, please consider giving it a star! ⭐

---

## 🚨 **IMPORTANT DISCLAIMER**

**This software is for educational and research purposes only. It is not intended to provide financial advice or recommendations for actual trading or investment decisions.**

### Risk Warnings:
- **No Guarantee**: Past performance does not guarantee future results
- **Market Risk**: All investments carry risk of loss
- **Educational Tool**: This is a learning and research tool, not a trading system
- **Professional Advice**: Consult qualified financial advisors before making investment decisions
- **Testing Required**: Thoroughly test any strategy before considering real implementation
- **Regulatory Compliance**: Ensure compliance with local financial regulations

### Technical Disclaimers:
- **Data Accuracy**: We cannot guarantee the accuracy of third-party data sources
- **System Reliability**: Software may contain bugs or experience downtime
- **API Dependencies**: Functionality depends on external API availability
- **Performance Variation**: Actual results may vary from backtested performance

**USE AT YOUR OWN RISK. THE AUTHORS AND CONTRIBUTORS ARE NOT RESPONSIBLE FOR ANY FINANCIAL LOSSES OR DAMAGES.**

---

*Built with ❤️ for the algorithmic trading community*#   M L - A I - I n t e r n s h i p - a t - H - T o - H  
 