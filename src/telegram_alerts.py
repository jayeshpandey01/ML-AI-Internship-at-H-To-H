#!/usr/bin/env python3
"""
Telegram Alerts System - Step 7: Bonus Task
Sends real-time alerts for trading signals, errors, and system status updates
"""

import asyncio
import logging
from datetime import datetime
import os
from dotenv import load_dotenv
import json
import traceback
from telegram import Bot
from telegram.error import TelegramError, RetryAfter, NetworkError
import time

# Load environment variables
load_dotenv('config/config.env')

class TelegramAlertsSystem:
    def __init__(self):
        """
        Initialize Telegram Alerts System
        """
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.bot = None
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting
        self.last_message_time = 0
        self.min_message_interval = 1  # Minimum 1 second between messages
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        
        print(f"📱 Telegram Alerts System Initialized:")
        print(f"   Bot Token: {'✅ Configured' if self.bot_token else '❌ Missing'}")
        print(f"   Chat ID: {'✅ Configured' if self.chat_id else '❌ Missing'}")
        
        if self.bot_token and self.chat_id:
            self.bot = Bot(token=self.bot_token)
            self.logger.info("Telegram bot initialized successfully")
        else:
            self.logger.warning("Telegram bot not configured - alerts will be disabled")
            print("⚠️ Telegram configuration missing. Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to config.env")
    
    def is_configured(self):
        """Check if Telegram is properly configured"""
        return self.bot is not None and self.bot_token and self.chat_id
    
    async def send_message_async(self, message, parse_mode='HTML'):
        """
        Send message asynchronously with error handling and rate limiting
        """
        if not self.is_configured():
            self.logger.warning("Telegram not configured - message not sent")
            return False
        
        try:
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_message_time
            if time_since_last < self.min_message_interval:
                await asyncio.sleep(self.min_message_interval - time_since_last)
            
            # Send message with retries
            for attempt in range(self.max_retries):
                try:
                    await self.bot.send_message(
                        chat_id=self.chat_id,
                        text=message,
                        parse_mode=parse_mode
                    )
                    
                    self.last_message_time = time.time()
                    self.logger.info(f"Telegram message sent successfully (attempt {attempt + 1})")
                    return True
                    
                except RetryAfter as e:
                    # Telegram rate limit - wait and retry
                    wait_time = e.retry_after + 1
                    self.logger.warning(f"Telegram rate limit hit, waiting {wait_time} seconds")
                    await asyncio.sleep(wait_time)
                    
                except NetworkError as e:
                    # Network error - retry with exponential backoff
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** attempt)
                        self.logger.warning(f"Network error, retrying in {wait_time} seconds: {e}")
                        await asyncio.sleep(wait_time)
                    else:
                        raise
                        
                except TelegramError as e:
                    self.logger.error(f"Telegram API error: {e}")
                    return False
            
            return False
            
        except Exception as e:
            self.logger.error(f"Unexpected error sending Telegram message: {e}")
            return False
    
    def send_message(self, message, parse_mode='HTML'):
        """
        Synchronous wrapper for sending messages
        """
        try:
            # Run async function in event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.send_message_async(message, parse_mode))
            loop.close()
            return result
        except Exception as e:
            self.logger.error(f"Error in synchronous message send: {e}")
            return False
    
    def send_trade_signal_alert(self, stock, signal_type, price, reason, ml_confidence=None):
        """
        Send trading signal alert
        
        Parameters:
        - stock: Stock symbol
        - signal_type: 'BUY' or 'SELL'
        - price: Trade price
        - reason: Signal reason
        - ml_confidence: ML confidence score (optional)
        """
        try:
            # Determine emoji based on signal type
            emoji = "🟢" if signal_type.upper() == "BUY" else "🔴"
            
            # Format message
            message = f"""
{emoji} <b>TRADING SIGNAL</b> {emoji}

📊 <b>Stock:</b> {stock}
🎯 <b>Action:</b> {signal_type.upper()}
💰 <b>Price:</b> ₹{price:.2f}
📈 <b>Reason:</b> {reason}
"""
            
            if ml_confidence:
                message += f"🤖 <b>ML Confidence:</b> {ml_confidence:.1%}\n"
            
            message += f"⏰ <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            success = self.send_message(message)
            
            if success:
                self.logger.info(f"Trade signal alert sent: {signal_type} {stock} at ₹{price:.2f}")
            else:
                self.logger.error(f"Failed to send trade signal alert: {signal_type} {stock}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending trade signal alert: {e}")
            return False
    
    def send_error_alert(self, error_type, error_message, component=None):
        """
        Send error alert
        
        Parameters:
        - error_type: Type of error
        - error_message: Error description
        - component: System component where error occurred
        """
        try:
            message = f"""
🚨 <b>SYSTEM ERROR</b> 🚨

⚠️ <b>Type:</b> {error_type}
📍 <b>Component:</b> {component or 'Unknown'}
💬 <b>Message:</b> {error_message}
⏰ <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🔧 <b>Action:</b> System will attempt automatic recovery
"""
            
            success = self.send_message(message)
            
            if success:
                self.logger.info(f"Error alert sent: {error_type}")
            else:
                self.logger.error(f"Failed to send error alert: {error_type}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending error alert: {e}")
            return False
    
    def send_system_status_alert(self, status, details=None):
        """
        Send system status alert
        
        Parameters:
        - status: System status ('STARTED', 'STOPPED', 'HEALTHY', 'WARNING')
        - details: Additional status details
        """
        try:
            # Determine emoji based on status
            status_emojis = {
                'STARTED': '🚀',
                'STOPPED': '⏹️',
                'HEALTHY': '💚',
                'WARNING': '⚠️',
                'ERROR': '🚨'
            }
            
            emoji = status_emojis.get(status.upper(), '📊')
            
            message = f"""
{emoji} <b>SYSTEM STATUS</b> {emoji}

📊 <b>Status:</b> {status.upper()}
⏰ <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            if details:
                message += f"📋 <b>Details:</b> {details}\n"
            
            success = self.send_message(message)
            
            if success:
                self.logger.info(f"System status alert sent: {status}")
            else:
                self.logger.error(f"Failed to send system status alert: {status}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending system status alert: {e}")
            return False
    
    def send_performance_summary(self, summary_data):
        """
        Send performance summary alert
        
        Parameters:
        - summary_data: Dictionary containing performance metrics
        """
        try:
            message = f"""
📈 <b>PERFORMANCE SUMMARY</b> 📈

🎯 <b>Total Trades:</b> {summary_data.get('total_trades', 0)}
💰 <b>Total P&L:</b> ₹{summary_data.get('total_pnl', 0):+,.2f}
📊 <b>Win Rate:</b> {summary_data.get('win_rate', 0):.1f}%
🤖 <b>ML Accuracy:</b> {summary_data.get('ml_accuracy', 0):.1%}
⏱️ <b>Duration:</b> {summary_data.get('duration', 0):.1f}s
⏰ <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🏆 <b>Best Trade:</b> ₹{summary_data.get('best_trade', 0):+.2f}
📉 <b>Worst Trade:</b> ₹{summary_data.get('worst_trade', 0):+.2f}
"""
            
            success = self.send_message(message)
            
            if success:
                self.logger.info("Performance summary alert sent")
            else:
                self.logger.error("Failed to send performance summary alert")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending performance summary: {e}")
            return False
    
    def send_ml_prediction_alert(self, stock, prediction, confidence, current_price):
        """
        Send ML prediction alert
        
        Parameters:
        - stock: Stock symbol
        - prediction: 'UP' or 'DOWN'
        - confidence: Confidence score
        - current_price: Current stock price
        """
        try:
            emoji = "📈" if prediction.upper() == "UP" else "📉"
            
            message = f"""
🤖 <b>ML PREDICTION</b> 🤖

📊 <b>Stock:</b> {stock}
{emoji} <b>Prediction:</b> {prediction.upper()}
🎯 <b>Confidence:</b> {confidence:.1%}
💰 <b>Current Price:</b> ₹{current_price:.2f}
⏰ <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📋 <b>Note:</b> This is a next-day price movement prediction
"""
            
            success = self.send_message(message)
            
            if success:
                self.logger.info(f"ML prediction alert sent: {stock} {prediction}")
            else:
                self.logger.error(f"Failed to send ML prediction alert: {stock}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending ML prediction alert: {e}")
            return False
    
    def send_test_message(self):
        """
        Send test message to verify bot functionality
        """
        try:
            message = f"""
🧪 <b>TEST MESSAGE</b> 🧪

✅ Telegram bot is working correctly!
📱 Bot Token: Configured
💬 Chat ID: Configured
⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🚀 Algo-Trading System is ready to send alerts!
"""
            
            success = self.send_message(message)
            
            if success:
                print("✅ Test message sent successfully!")
                self.logger.info("Test message sent successfully")
            else:
                print("❌ Failed to send test message")
                self.logger.error("Failed to send test message")
            
            return success
            
        except Exception as e:
            print(f"❌ Error sending test message: {e}")
            self.logger.error(f"Error sending test message: {e}")
            return False
    
    def handle_rate_limit_gracefully(self, func, *args, **kwargs):
        """
        Handle rate limits gracefully for any function
        """
        try:
            return func(*args, **kwargs)
        except RetryAfter as e:
            self.logger.warning(f"Rate limit hit, waiting {e.retry_after} seconds")
            time.sleep(e.retry_after + 1)
            return func(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Error in rate limit handler: {e}")
            return False
    
    def get_bot_info(self):
        """
        Get bot information for verification
        """
        if not self.is_configured():
            return None
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def get_info():
                return await self.bot.get_me()
            
            bot_info = loop.run_until_complete(get_info())
            loop.close()
            
            return {
                'id': bot_info.id,
                'username': bot_info.username,
                'first_name': bot_info.first_name,
                'is_bot': bot_info.is_bot
            }
            
        except Exception as e:
            self.logger.error(f"Error getting bot info: {e}")
            return None

def main():
    """
    Test the Telegram Alerts System
    """
    print("🧪 TELEGRAM ALERTS SYSTEM TEST")
    print("=" * 60)
    
    # Initialize alerts system
    alerts = TelegramAlertsSystem()
    
    if not alerts.is_configured():
        print("❌ Telegram not configured. Please add the following to config/config.env:")
        print("   TELEGRAM_BOT_TOKEN=your_bot_token_here")
        print("   TELEGRAM_CHAT_ID=your_chat_id_here")
        print("\n📋 Setup Instructions:")
        print("   1. Create a bot with @BotFather on Telegram")
        print("   2. Get your bot token")
        print("   3. Get your chat ID by messaging @userinfobot")
        print("   4. Add both to config/config.env")
        return
    
    # Get bot info
    print("🤖 Getting bot information...")
    bot_info = alerts.get_bot_info()
    if bot_info:
        print(f"✅ Bot Info:")
        print(f"   Username: @{bot_info['username']}")
        print(f"   Name: {bot_info['first_name']}")
        print(f"   ID: {bot_info['id']}")
    
    # Test different types of alerts
    print("\n📱 Testing different alert types...")
    
    # 1. Test message
    print("1. Sending test message...")
    alerts.send_test_message()
    time.sleep(2)
    
    # 2. Trade signal alert
    print("2. Sending trade signal alert...")
    alerts.send_trade_signal_alert(
        stock='RELIANCE.NS',
        signal_type='BUY',
        price=1450.75,
        reason='RSI Oversold (32.1) + Bullish MA Crossover',
        ml_confidence=0.85
    )
    time.sleep(2)
    
    # 3. Error alert
    print("3. Sending error alert...")
    alerts.send_error_alert(
        error_type='API Rate Limit',
        error_message='Alpha Vantage API rate limit exceeded',
        component='Data Fetcher'
    )
    time.sleep(2)
    
    # 4. System status alert
    print("4. Sending system status alert...")
    alerts.send_system_status_alert(
        status='HEALTHY',
        details='All systems operational, 3 stocks monitored'
    )
    time.sleep(2)
    
    # 5. Performance summary
    print("5. Sending performance summary...")
    alerts.send_performance_summary({
        'total_trades': 7,
        'total_pnl': 2930.47,
        'win_rate': 85.7,
        'ml_accuracy': 0.585,
        'duration': 3.76,
        'best_trade': 840.30,
        'worst_trade': -17.55
    })
    time.sleep(2)
    
    # 6. ML prediction alert
    print("6. Sending ML prediction alert...")
    alerts.send_ml_prediction_alert(
        stock='HDFCBANK.NS',
        prediction='UP',
        confidence=0.73,
        current_price=2012.20
    )
    
    print("\n✅ All test alerts sent!")
    print("📱 Check your Telegram chat to verify alerts were received")

if __name__ == "__main__":
    main()