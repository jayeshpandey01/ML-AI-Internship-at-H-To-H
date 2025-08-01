# 🚀 Algo-Trading System

A comprehensive, production-ready algorithmic trading system with machine learning predictions, automated backtesting, Google Sheets integration, and real-time Telegram alerts.

---

## ✨ Features

### 📊 Data Management

* **Multi-Source Data Fetching**: Alpha Vantage & Yahoo Finance with intelligent fallback
* **Smart Caching System**: Reduces API calls and improves performance
* **Data Validation & Cleaning**: Ensures data quality and integrity
* **Technical Indicators**: 20+ built-in indicators (RSI, MACD, Bollinger Bands, etc.)

### 🎯 Trading Strategy

* **RSI + Moving Average Crossover**: Proven technical analysis strategy
* **Comprehensive Backtesting**: Historical performance analysis with detailed metrics
* **Risk Management**: Configurable parameters and position sizing
* **Performance Analytics**: Sharpe ratio, win rate, drawdown analysis

### 🤖 Machine Learning

* **Random Forest Predictor**: Next-day price movement predictions
* **Feature Engineering**: 30+ technical and statistical features
* **Model Validation**: Cross-validation and performance metrics
* **Confidence Scoring**: Prediction confidence levels

### 📈 Automation & Integration

* **Google Sheets Logging**: Automated trade logs and performance reports
* **Telegram Alerts**: Real-time notifications for trades, errors, and system status
* **System Monitoring**: Health checks and performance tracking
* **Configuration Management**: Centralized, validated configuration system

### 🏗️ Professional Architecture

* **Modular Design**: Clean separation of concerns
* **Type Hints**: Full type annotation for better code quality
* **Comprehensive Logging**: Detailed logging with configurable levels
* **Error Handling**: Robust error recovery and retry mechanisms
* **Testing Suite**: Unit and integration tests
* **Code Quality**: PEP 8 compliant with automated formatting

---

## 📋 Requirements

### System Requirements

* **Python**: 3.8 or higher
* **Operating System**: Windows, macOS, or Linux
* **Memory**: Minimum 4GB RAM (8GB recommended)
* **Storage**: 1GB free space for data and logs

### API Requirements

* **Alpha Vantage API Key**: [Get free key](https://www.alphavantage.co/support/#api-key)
* **Google Sheets API**: Service account credentials
* **Telegram Bot Token**: For real-time alerts (optional)

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone <your-repo-url>
cd algo-trading-system
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configuration Setup

#### 2.1 Copy and Modify Configuration

```bash
cp config/config.example.env config/config.env
```

Update `config.env` with:

* API Keys
* Stock Symbols
* Trading Parameters

#### 2.2 Google Sheets

1. Create a spreadsheet
2. Enable Google Sheets API via [Cloud Console](https://console.cloud.google.com)
3. Download credentials as `google_credentials.json`
4. Set `GOOGLE_SHEETS_CREDENTIALS_PATH=config/google_credentials.json`

#### 2.3 Telegram Alerts (Optional)

1. Create bot via [@BotFather](https://t.me/BotFather)
2. Get bot token and chat ID
3. Configure in `config.env`

### 3. Run the System

```bash
python src/main.py               # Basic analysis
python src/main_with_telegram.py # With Telegram alerts
```

---

## 📊 Strategy: RSI + MA Crossover

**Buy:** RSI < 30 and MA(10) > MA(20)

**Sell:** RSI > 70 and MA(10) < MA(20)

---

## 🤖 ML Features

* RSI, MACD, Bollinger Bands
* Price ratios, volume, moving averages
* Lag features, volatility indicators

---

## 📈 Output

### Console

```
Stock: RELIANCE.NS
Total Return: 12.45%
ML Accuracy: 67.8%
Buy Signal: ✅
```

### Telegram

```
🟢 BUY SIGNAL
Stock: RELIANCE.NS
Reason: RSI + MA
ML Confidence: 85.0%
```

### Google Sheets

* **Trade Log**
* **P\&L Summary**
* **ML Predictions**
* **Indicator Values**

---

## 📇 Architecture

```
algo-trading-system/
├── src/
│   ├── data.py          # Data pipeline
│   ├── strategy.py      # Trading logic
│   ├── ml_model.py      # ML model
│   ├── sheets.py        # Google Sheets
│   ├── telegram.py      # Telegram alerts
│   ├── main.py          # Main runner
├── config/
├── tests/
├── logs/
├── data/
```

---

## 🔧 Customization

### Add Stocks

```env
NIFTY_STOCKS=RELIANCE.NS,TCS.NS,ITC.NS
```

### New Strategy

Create `custom_strategy.py`:

```python
class MyStrategy(TradingStrategy):
    def generate_signals(self, data):
        # Your logic
        return signals
```

### Add Indicator

In `DataPreprocessor`:

```python
data['custom'] = some_function(data)
```

---

## ⚠️ Disclaimers

* For **educational purposes** only
* **Not financial advice**
* Use **paper trading** to test
* **Thoroughly test** before deploying
* Respect API **rate limits**

---

## 🚨 Troubleshooting

### Data Not Fetching

```bash
python -c "from src.data import DataFetcher; f=DataFetcher(); print(f.fetch_stock_data('RELIANCE.NS'))"
```

### Sheets Issue

```bash
python -c "from src.sheets import SheetsManager; print(SheetsManager().setup_connection())"
```

---

## 📅 Roadmap

* [ ] Multi-timeframe support
* [ ] Real-time trading
* [ ] Sentiment analysis
* [ ] Portfolio optimization
* [ ] Cloud deployment

---

## 📓 License

MIT License. Use at your own risk.

---

*Built with ❤️ for the trading & learning community*
