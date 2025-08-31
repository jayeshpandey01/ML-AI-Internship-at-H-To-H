# 📱 Telegram Alerts Setup Guide

## Overview
This guide will help you set up real-time Telegram notifications for your algo-trading system. You'll receive alerts for:

- 🟢 **Buy/Sell Signals** - Real-time trading signals with prices and reasons
- 🚨 **Error Alerts** - System errors and API issues
- 📊 **System Status** - Startup, shutdown, and health updates
- 🤖 **ML Predictions** - Machine learning predictions with confidence scores
- 📈 **Performance Summaries** - Daily trading performance reports

## Step 1: Create Your Telegram Bot

### 1.1 Start a Chat with BotFather
1. Open Telegram
2. Search for `@BotFather`
3. Start a conversation and send `/start`

### 1.2 Create a New Bot
1. Send `/newbot` to BotFather
2. Choose a name for your bot (e.g., "My Trading Bot")
3. Choose a username (must end with 'bot', e.g., "my_trading_alerts_bot")
4. **Save the bot token** - you'll need this!

Example response:
```
Congratulations! Here is your token:

Keep your token secure and store it safely, it can be used by anyone to control your bot.
```

## Step 2: Get Your Chat ID

### 2.1 Message Your Bot
1. Search for your bot username in Telegram
2. Start a conversation
3. Send any message (like "hello" or "test")

### 2.2 Run the Chat ID Finder
```bash
python get_chat_id.py
```

This script will:
- Find your chat ID from recent messages
- Automatically update your `config/config.env` file
- Confirm the setup is working

## Step 3: Configure Your System

Your `config/config.env` should now contain:
```env
# Telegram Configuration
TELEGRAM_BOT_TOKEN=6757118610:AAFmIzSIwfdh9z8JYCKLOfzDWvZ0b_AErA0
TELEGRAM_CHAT_ID=123456789

# Telegram Alert Settings
TELEGRAM_ENABLE_TRADE_ALERTS=true
TELEGRAM_ENABLE_ERROR_ALERTS=true
TELEGRAM_ENABLE_STATUS_ALERTS=true
TELEGRAM_ENABLE_ML_ALERTS=true
```

## Step 4: Test Your Setup

### 4.1 Basic Test
```bash
python src/telegram_alerts.py
```

This will send various test messages to verify everything works.

### 4.2 Integration Test
```bash
python test_telegram_integration.py
```

This simulates a full trading session with alerts.

### 4.3 Production Test
```bash
python src/main_with_telegram.py
```

This runs your actual trading system with Telegram alerts enabled.

## Step 5: Customize Alert Settings

You can enable/disable specific alert types in `config/config.env`:

```env
# Enable/disable specific alert types
TELEGRAM_ENABLE_TRADE_ALERTS=true    # Buy/sell signals
TELEGRAM_ENABLE_ERROR_ALERTS=true    # System errors
TELEGRAM_ENABLE_STATUS_ALERTS=true   # System status updates
TELEGRAM_ENABLE_ML_ALERTS=true       # ML predictions
```

## Alert Types and Examples

### 🟢 Buy Signal Alert
```
🟢 TRADING SIGNAL 🟢

📊 Stock: RELIANCE.NS
🎯 Action: BUY
💰 Price: ₹1,450.75
📈 Reason: RSI Oversold (32.1) + Bullish MA Crossover
🤖 ML Confidence: 85.0%
⏰ Time: 2024-01-15 14:30:25
```

### 🔴 Sell Signal Alert
```
🔴 TRADING SIGNAL 🔴

📊 Stock: RELIANCE.NS
🎯 Action: SELL
💰 Price: ₹1,485.20
📈 Reason: Target reached
⏰ Time: 2024-01-15 15:45:10
```

### 🚨 Error Alert
```
🚨 SYSTEM ERROR 🚨

⚠️ Type: API Rate Limit
📍 Component: Data Fetcher
💬 Message: Alpha Vantage API rate limit exceeded
⏰ Time: 2024-01-15 14:25:30

🔧 Action: System will attempt automatic recovery
```

### 📊 System Status Alert
```
🚀 SYSTEM STATUS 🚀

📊 Status: STARTED
⏰ Time: 2024-01-15 14:00:00
📋 Details: Analysis started for 3 stocks: RELIANCE.NS, HDFCBANK.NS, INFY.NS
```

### 🤖 ML Prediction Alert
```
🤖 ML PREDICTION 🤖

📊 Stock: HDFCBANK.NS
📈 Prediction: UP
🎯 Confidence: 73.0%
💰 Current Price: ₹2,012.20
⏰ Time: 2024-01-15 14:35:45

📋 Note: This is a next-day price movement prediction
```

### 📈 Performance Summary
```
📈 PERFORMANCE SUMMARY 📈

🎯 Total Trades: 7
💰 Total P&L: ₹+2,930.47
📊 Win Rate: 85.7%
🤖 ML Accuracy: 58.5%
⏱️ Duration: 3.8s
⏰ Time: 2024-01-15 14:40:15

🏆 Best Trade: ₹+840.30
📉 Worst Trade: ₹-17.55
```

## Troubleshooting

### Common Issues

#### 1. "Chat not found" Error
- **Cause**: You haven't messaged your bot yet
- **Solution**: Send any message to your bot first, then run `get_chat_id.py`

#### 2. "Unauthorized" Error
- **Cause**: Invalid bot token
- **Solution**: Double-check your bot token from BotFather

#### 3. "Network Error" or "Timeout"
- **Cause**: Internet connectivity issues
- **Solution**: Check your internet connection and try again

#### 4. Rate Limiting
- **Cause**: Sending too many messages too quickly
- **Solution**: The system automatically handles rate limiting with delays

### Getting Help

If you encounter issues:

1. **Check Configuration**: Ensure your bot token and chat ID are correct
2. **Test Basic Functionality**: Run `python src/telegram_alerts.py`
3. **Check Logs**: Look for error messages in the console output
4. **Verify Bot Status**: Make sure your bot is active and not blocked

## Security Best Practices

1. **Keep Your Bot Token Secret**: Never share it publicly or commit it to version control
2. **Use Environment Variables**: Store sensitive data in `config/config.env`
3. **Limit Bot Permissions**: Your bot only needs to send messages
4. **Monitor Usage**: Keep track of who has access to your bot

## Advanced Features

### Rate Limiting
The system automatically handles Telegram's rate limits:
- Minimum 1 second between messages
- Automatic retry with exponential backoff
- Graceful handling of rate limit errors

### Error Recovery
- Automatic retry for network errors
- Fallback to console logging if Telegram fails
- Continues operation even if alerts fail

### Message Formatting
- Uses HTML formatting for rich text
- Includes emojis for visual appeal
- Structured layout for easy reading

## Integration with Existing System

The Telegram alerts are designed to integrate seamlessly with your existing trading system:

1. **Non-Intrusive**: Alerts don't affect trading logic
2. **Optional**: System works fine even if Telegram is disabled
3. **Configurable**: Enable/disable specific alert types
4. **Reliable**: Continues trading even if alerts fail

## Next Steps

Once your Telegram alerts are working:

1. **Customize Messages**: Modify alert templates in `src/telegram_alerts.py`
2. **Add New Alert Types**: Create custom alerts for specific events
3. **Set Up Monitoring**: Use status alerts to monitor system health
4. **Scale Up**: Add alerts for multiple trading strategies

---

🎉 **Congratulations!** Your algo-trading system now has real-time Telegram notifications!

For questions or issues, check the troubleshooting section above or review the code in `src/telegram_alerts.py`.