#!/usr/bin/env python3
"""
Test Telegram Integration with Trading System
This script demonstrates how to integrate Telegram alerts with your existing trading system
"""

import sys
import os
sys.path.append('src')

from telegram_alerts import TelegramAlertsSystem
import time
import random

def simulate_trading_session():
    """
    Simulate a trading session with Telegram alerts
    """
    print("🚀 SIMULATED TRADING SESSION WITH TELEGRAM ALERTS")
    print("=" * 60)
    
    # Initialize Telegram alerts
    alerts = TelegramAlertsSystem()
    
    if not alerts.is_configured():
        print("❌ Telegram not configured. Please:")
        print("   1. Message your bot @pbl_project_bot")
        print("   2. Run: python get_chat_id.py")
        print("   3. Run this script again")
        return
    
    # Send startup notification
    print("📱 Sending startup notification...")
    alerts.send_system_status_alert(
        status='STARTED',
        details='Algo-Trading System initialized, monitoring 3 stocks'
    )
    time.sleep(2)
    
    # Simulate some trading signals
    stocks = ['RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS']
    
    for i, stock in enumerate(stocks):
        print(f"📊 Processing {stock}...")
        
        # Simulate buy signal
        buy_price = random.uniform(1400, 2500)
        alerts.send_trade_signal_alert(
            stock=stock,
            signal_type='BUY',
            price=buy_price,
            reason='RSI Oversold + Bullish MA Crossover',
            ml_confidence=random.uniform(0.7, 0.9)
        )
        time.sleep(3)
        
        # Simulate sell signal
        sell_price = buy_price * random.uniform(1.01, 1.05)  # Simulate profit
        pnl = sell_price - buy_price
        alerts.send_trade_signal_alert(
            stock=stock,
            signal_type='SELL',
            price=sell_price,
            reason='Target reached',
            ml_confidence=random.uniform(0.6, 0.8)
        )
        time.sleep(3)
        
        # Send ML prediction
        prediction = random.choice(['UP', 'DOWN'])
        alerts.send_ml_prediction_alert(
            stock=stock,
            prediction=prediction,
            confidence=random.uniform(0.6, 0.85),
            current_price=sell_price
        )
        time.sleep(2)
    
    # Simulate an error
    print("⚠️ Simulating error alert...")
    alerts.send_error_alert(
        error_type='API Rate Limit',
        error_message='Alpha Vantage API rate limit exceeded, switching to backup data source',
        component='Data Fetcher'
    )
    time.sleep(3)
    
    # Send performance summary
    print("📈 Sending performance summary...")
    alerts.send_performance_summary({
        'total_trades': 6,
        'total_pnl': 1247.83,
        'win_rate': 100.0,
        'ml_accuracy': 0.75,
        'duration': 45.2,
        'best_trade': 523.45,
        'worst_trade': 89.12
    })
    time.sleep(2)
    
    # Send completion status
    alerts.send_system_status_alert(
        status='HEALTHY',
        details='Trading session completed successfully'
    )
    
    print("✅ Simulated trading session completed!")
    print("📱 Check your Telegram for all the alerts")

def test_error_scenarios():
    """
    Test error handling scenarios
    """
    print("\n🧪 TESTING ERROR SCENARIOS")
    print("=" * 40)
    
    alerts = TelegramAlertsSystem()
    
    if not alerts.is_configured():
        print("❌ Telegram not configured")
        return
    
    # Test different error types
    error_scenarios = [
        {
            'type': 'Data Fetch Error',
            'message': 'Failed to fetch data for RELIANCE.NS',
            'component': 'Data Ingestion'
        },
        {
            'type': 'ML Model Error',
            'message': 'Model prediction failed due to insufficient data',
            'component': 'ML Predictor'
        },
        {
            'type': 'Google Sheets Error',
            'message': 'Failed to update Google Sheets with trade data',
            'component': 'Sheets Integration'
        }
    ]
    
    for scenario in error_scenarios:
        print(f"📤 Sending error alert: {scenario['type']}")
        alerts.send_error_alert(
            error_type=scenario['type'],
            error_message=scenario['message'],
            component=scenario['component']
        )
        time.sleep(2)
    
    print("✅ Error scenario testing completed!")

def main():
    """
    Main function to run all tests
    """
    print("📱 TELEGRAM INTEGRATION TEST SUITE")
    print("=" * 60)
    
    # Test basic functionality
    alerts = TelegramAlertsSystem()
    
    print("🔍 Checking configuration...")
    if alerts.is_configured():
        print("✅ Telegram is properly configured")
        
        # Get bot info
        bot_info = alerts.get_bot_info()
        if bot_info:
            print(f"🤖 Bot: @{bot_info['username']} ({bot_info['first_name']})")
        
        # Send test message
        print("📤 Sending test message...")
        if alerts.send_test_message():
            print("✅ Test message sent successfully!")
            
            # Ask user if they want to run full simulation
            response = input("\n🤔 Do you want to run the full trading simulation? (y/n): ").lower()
            if response in ['y', 'yes']:
                simulate_trading_session()
                
                # Ask about error testing
                response = input("\n🤔 Do you want to test error scenarios? (y/n): ").lower()
                if response in ['y', 'yes']:
                    test_error_scenarios()
            else:
                print("✅ Basic test completed!")
        else:
            print("❌ Test message failed")
    else:
        print("❌ Telegram not configured")
        print("\n📋 Setup Instructions:")
        print("1. Message your bot: @pbl_project_bot")
        print("2. Run: python get_chat_id.py")
        print("3. Run this script again")

if __name__ == "__main__":
    main()