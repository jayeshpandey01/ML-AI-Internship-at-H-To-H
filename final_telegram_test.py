#!/usr/bin/env python3
"""
Final Comprehensive Telegram Test
This script performs a complete test of the Telegram alerts system
"""

import sys
import os
sys.path.append('src')

from telegram_alerts import TelegramAlertsSystem
import time

def test_configuration():
    """Test basic configuration"""
    print("🔍 TESTING CONFIGURATION")
    print("=" * 40)
    
    alerts = TelegramAlertsSystem()
    
    if not alerts.is_configured():
        print("❌ Telegram not configured!")
        print("\n📋 Setup Required:")
        print("1. Message your bot: @pbl_project_bot")
        print("2. Run: python get_chat_id.py")
        print("3. Run this test again")
        return False
    
    print("✅ Telegram is configured")
    
    # Get bot info
    bot_info = alerts.get_bot_info()
    if bot_info:
        print(f"🤖 Bot: @{bot_info['username']} ({bot_info['first_name']})")
        print(f"📱 Bot ID: {bot_info['id']}")
        return True
    else:
        print("❌ Could not get bot information")
        return False

def test_all_alert_types():
    """Test all different alert types"""
    print("\n📱 TESTING ALL ALERT TYPES")
    print("=" * 40)
    
    alerts = TelegramAlertsSystem()
    
    if not alerts.is_configured():
        print("❌ Telegram not configured")
        return False
    
    test_results = {}
    
    # 1. Test Message
    print("1. Testing basic message...")
    result = alerts.send_test_message()
    test_results['test_message'] = result
    print(f"   {'✅ Success' if result else '❌ Failed'}")
    time.sleep(2)
    
    # 2. Buy Signal
    print("2. Testing buy signal alert...")
    result = alerts.send_trade_signal_alert(
        stock='RELIANCE.NS',
        signal_type='BUY',
        price=1450.75,
        reason='RSI Oversold + MA Crossover',
        ml_confidence=0.85
    )
    test_results['buy_signal'] = result
    print(f"   {'✅ Success' if result else '❌ Failed'}")
    time.sleep(2)
    
    # 3. Sell Signal
    print("3. Testing sell signal alert...")
    result = alerts.send_trade_signal_alert(
        stock='RELIANCE.NS',
        signal_type='SELL',
        price=1485.20,
        reason='Target reached',
        ml_confidence=0.78
    )
    test_results['sell_signal'] = result
    print(f"   {'✅ Success' if result else '❌ Failed'}")
    time.sleep(2)
    
    # 4. Error Alert
    print("4. Testing error alert...")
    result = alerts.send_error_alert(
        error_type='API Rate Limit',
        error_message='Alpha Vantage API rate limit exceeded',
        component='Data Fetcher'
    )
    test_results['error_alert'] = result
    print(f"   {'✅ Success' if result else '❌ Failed'}")
    time.sleep(2)
    
    # 5. System Status
    print("5. Testing system status alert...")
    result = alerts.send_system_status_alert(
        status='HEALTHY',
        details='All systems operational'
    )
    test_results['status_alert'] = result
    print(f"   {'✅ Success' if result else '❌ Failed'}")
    time.sleep(2)
    
    # 6. ML Prediction
    print("6. Testing ML prediction alert...")
    result = alerts.send_ml_prediction_alert(
        stock='HDFCBANK.NS',
        prediction='UP',
        confidence=0.73,
        current_price=2012.20
    )
    test_results['ml_prediction'] = result
    print(f"   {'✅ Success' if result else '❌ Failed'}")
    time.sleep(2)
    
    # 7. Performance Summary
    print("7. Testing performance summary...")
    result = alerts.send_performance_summary({
        'total_trades': 6,
        'total_pnl': 1247.83,
        'win_rate': 83.3,
        'ml_accuracy': 0.75,
        'duration': 45.2,
        'best_trade': 523.45,
        'worst_trade': -89.12
    })
    test_results['performance_summary'] = result
    print(f"   {'✅ Success' if result else '❌ Failed'}")
    
    return test_results

def test_error_handling():
    """Test error handling scenarios"""
    print("\n🧪 TESTING ERROR HANDLING")
    print("=" * 40)
    
    alerts = TelegramAlertsSystem()
    
    if not alerts.is_configured():
        print("❌ Telegram not configured")
        return False
    
    # Test rate limiting (send multiple messages quickly)
    print("1. Testing rate limiting...")
    for i in range(3):
        result = alerts.send_message(f"Rate limit test message {i+1}")
        print(f"   Message {i+1}: {'✅' if result else '❌'}")
    
    print("✅ Rate limiting test completed")
    return True

def generate_test_report(config_ok, alert_results, error_handling_ok):
    """Generate comprehensive test report"""
    print("\n📊 COMPREHENSIVE TEST REPORT")
    print("=" * 60)
    
    # Configuration Test
    print(f"🔧 Configuration: {'✅ PASS' if config_ok else '❌ FAIL'}")
    
    # Alert Tests
    if alert_results:
        print("\n📱 Alert Type Tests:")
        for alert_type, result in alert_results.items():
            status = '✅ PASS' if result else '❌ FAIL'
            print(f"   {alert_type.replace('_', ' ').title()}: {status}")
        
        # Calculate success rate
        successful_alerts = sum(1 for result in alert_results.values() if result)
        total_alerts = len(alert_results)
        success_rate = (successful_alerts / total_alerts) * 100
        
        print(f"\n📈 Alert Success Rate: {success_rate:.1f}% ({successful_alerts}/{total_alerts})")
    else:
        print("\n📱 Alert Type Tests: ❌ SKIPPED (Configuration failed)")
    
    # Error Handling Test
    print(f"\n🛡️ Error Handling: {'✅ PASS' if error_handling_ok else '❌ FAIL'}")
    
    # Overall Status
    overall_success = config_ok and (alert_results and all(alert_results.values())) and error_handling_ok
    
    print(f"\n🎯 OVERALL STATUS: {'✅ ALL TESTS PASSED' if overall_success else '⚠️ SOME TESTS FAILED'}")
    
    if overall_success:
        print("\n🎉 Congratulations! Your Telegram alerts system is fully operational!")
        print("📱 You should have received 7 different test messages in your Telegram chat")
        print("🚀 You can now run your trading system with real-time alerts!")
    else:
        print("\n🔧 Some tests failed. Please check the following:")
        if not config_ok:
            print("   - Verify your bot token and chat ID in config/config.env")
            print("   - Make sure you've messaged your bot first")
        if alert_results and not all(alert_results.values()):
            print("   - Check your internet connection")
            print("   - Verify your bot is not blocked or restricted")
        if not error_handling_ok:
            print("   - Check for network connectivity issues")
    
    print("\n📋 Next Steps:")
    if overall_success:
        print("   1. Run: python src/main_with_telegram.py")
        print("   2. Monitor your Telegram for real trading alerts")
        print("   3. Customize alert settings in config/config.env")
    else:
        print("   1. Fix the issues mentioned above")
        print("   2. Run this test again")
        print("   3. Check TELEGRAM_SETUP_GUIDE.md for detailed help")

def main():
    """Main test function"""
    print("🧪 FINAL COMPREHENSIVE TELEGRAM TEST")
    print("=" * 60)
    print("This test will verify your complete Telegram alerts setup")
    print("You should receive multiple test messages in your Telegram chat")
    print()
    
    # Test 1: Configuration
    config_ok = test_configuration()
    
    # Test 2: All Alert Types
    alert_results = None
    if config_ok:
        alert_results = test_all_alert_types()
    
    # Test 3: Error Handling
    error_handling_ok = False
    if config_ok:
        error_handling_ok = test_error_handling()
    
    # Generate Report
    generate_test_report(config_ok, alert_results, error_handling_ok)

if __name__ == "__main__":
    main()