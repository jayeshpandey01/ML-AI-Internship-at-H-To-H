#!/usr/bin/env python3
"""
Enhanced Algo-Trading System with Telegram Alerts
Integrates real-time Telegram notifications with the existing trading system
"""

import os
import sys
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import time

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api_call import DataFetcher
from trading_strategy import TradingStrategy
from ml_predictor import MLPredictor
from sheets_integration import SheetsLogger
from telegram_alerts import TelegramAlertsSystem

# Load environment variables
load_dotenv('config/config.env')

class EnhancedAlgoTradingSystem:
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.strategy = TradingStrategy()
        self.ml_predictor = MLPredictor()
        self.sheets_logger = SheetsLogger()
        self.telegram_alerts = TelegramAlertsSystem()
        
        # Get stock symbols from config
        self.stocks = os.getenv('NIFTY_STOCKS', 'RELIANCE.NS,HDFCBANK.NS,INFY.NS').split(',')
        
        # Alert settings from config
        self.enable_trade_alerts = os.getenv('TELEGRAM_ENABLE_TRADE_ALERTS', 'true').lower() == 'true'
        self.enable_error_alerts = os.getenv('TELEGRAM_ENABLE_ERROR_ALERTS', 'true').lower() == 'true'
        self.enable_status_alerts = os.getenv('TELEGRAM_ENABLE_STATUS_ALERTS', 'true').lower() == 'true'
        self.enable_ml_alerts = os.getenv('TELEGRAM_ENABLE_ML_ALERTS', 'true').lower() == 'true'
        
        print(f"📱 Telegram Alerts: {'✅ Enabled' if self.telegram_alerts.is_configured() else '❌ Disabled'}")
        if self.telegram_alerts.is_configured():
            print(f"   Trade Alerts: {'✅' if self.enable_trade_alerts else '❌'}")
            print(f"   Error Alerts: {'✅' if self.enable_error_alerts else '❌'}")
            print(f"   Status Alerts: {'✅' if self.enable_status_alerts else '❌'}")
            print(f"   ML Alerts: {'✅' if self.enable_ml_alerts else '❌'}")
        
    def send_alert(self, alert_type, *args, **kwargs):
        """Send alert if enabled and configured"""
        if not self.telegram_alerts.is_configured():
            return False
        
        try:
            if alert_type == 'trade' and self.enable_trade_alerts:
                return self.telegram_alerts.send_trade_signal_alert(*args, **kwargs)
            elif alert_type == 'error' and self.enable_error_alerts:
                return self.telegram_alerts.send_error_alert(*args, **kwargs)
            elif alert_type == 'status' and self.enable_status_alerts:
                return self.telegram_alerts.send_system_status_alert(*args, **kwargs)
            elif alert_type == 'ml' and self.enable_ml_alerts:
                return self.telegram_alerts.send_ml_prediction_alert(*args, **kwargs)
            elif alert_type == 'performance':
                return self.telegram_alerts.send_performance_summary(*args, **kwargs)
            
            return False
        except Exception as e:
            print(f"⚠️ Alert sending failed: {e}")
            return False
    
    def run_analysis(self):
        """Run complete analysis for all stocks with Telegram alerts"""
        start_time = datetime.now()
        
        print("=" * 60)
        print("ENHANCED ALGO-TRADING SYSTEM WITH TELEGRAM ALERTS")
        print("=" * 60)
        print(f"Analyzing stocks: {', '.join(self.stocks)}")
        print(f"Analysis started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Send startup notification
        self.send_alert('status', 
                       status='STARTED',
                       details=f'Analysis started for {len(self.stocks)} stocks: {", ".join(self.stocks)}')
        
        # Set up Google Sheets
        if not self.sheets_logger.setup_sheets():
            print("Warning: Google Sheets setup failed. Continuing without logging.")
            self.send_alert('error',
                           error_type='Google Sheets Setup Error',
                           error_message='Failed to initialize Google Sheets connection',
                           component='Sheets Integration')
        
        results_summary = {}
        total_trades = 0
        total_pnl = 0
        successful_analyses = 0
        ml_predictions = []
        
        for stock in self.stocks:
            print(f"\n{'='*40}")
            print(f"ANALYZING {stock}")
            print(f"{'='*40}")
            
            try:
                # Fetch stock data
                print(f"1. Fetching data for {stock}...")
                data = self.data_fetcher.fetch_yfinance_data(stock, period='6mo', interval='1d')
                
                if data is None or data.empty:
                    error_msg = f"No data available for {stock}"
                    print(f"❌ {error_msg}")
                    self.send_alert('error',
                                   error_type='Data Fetch Error',
                                   error_message=error_msg,
                                   component='Data Fetcher')
                    continue
                
                print(f"✅ Fetched {len(data)} records for {stock}")
                
                # Run trading strategy
                print(f"2. Running trading strategy for {stock}...")
                backtest_results = self.strategy.backtest_strategy(data)
                
                print(f"✅ Strategy analysis complete:")
                print(f"   - Total Return: {backtest_results['total_return']:.2%}")
                print(f"   - Market Return: {backtest_results['market_return']:.2%}")
                print(f"   - Win Rate: {backtest_results['win_rate']:.1f}%")
                print(f"   - Total Trades: {backtest_results['total_trades']}")
                print(f"   - Sharpe Ratio: {backtest_results['sharpe_ratio']:.3f}")
                
                # Send trade alerts for recent signals
                trades = backtest_results['trades']
                if not trades.empty and len(trades) > 0:
                    # Get the most recent trade
                    latest_trade = trades.iloc[-1]
                    current_price = data['Close'].iloc[-1]
                    
                    # Determine if this is a buy or sell signal
                    if latest_trade['Trade_Type'] == 'Buy':
                        self.send_alert('trade',
                                       stock=stock,
                                       signal_type='BUY',
                                       price=current_price,
                                       reason=f"RSI: {latest_trade.get('RSI', 'N/A'):.1f}, MA Crossover",
                                       ml_confidence=None)
                    elif latest_trade['Trade_Type'] == 'Sell':
                        # Calculate P&L for sell signal
                        pnl = backtest_results.get('total_pnl', 0)
                        self.send_alert('trade',
                                       stock=stock,
                                       signal_type='SELL',
                                       price=current_price,
                                       reason=f"Strategy Exit Signal",
                                       ml_confidence=None)
                
                # Train ML model
                print(f"3. Training ML model for {stock}...")
                ml_results = self.ml_predictor.train_model(data)
                
                if ml_results:
                    print(f"✅ ML model trained with accuracy: {ml_results['accuracy']:.3f}")
                    
                    # Make next-day prediction
                    prediction = self.ml_predictor.predict_next_day(data)
                    if prediction:
                        direction = "UP" if prediction['prediction'] == 1 else "DOWN"
                        print(f"   - Next day prediction: {direction} (confidence: {prediction['confidence']:.3f})")
                        
                        # Send ML prediction alert
                        self.send_alert('ml',
                                       stock=stock,
                                       prediction=direction,
                                       confidence=prediction['confidence'],
                                       current_price=data['Close'].iloc[-1])
                        
                        ml_predictions.append({
                            'stock': stock,
                            'prediction': direction,
                            'confidence': prediction['confidence']
                        })
                else:
                    print("❌ ML model training failed")
                    self.send_alert('error',
                                   error_type='ML Training Error',
                                   error_message=f'Failed to train ML model for {stock}',
                                   component='ML Predictor')
                    prediction = None
                
                # Log to Google Sheets
                print(f"4. Logging results to Google Sheets...")
                try:
                    # Update P&L summary
                    self.sheets_logger.update_pnl_summary(stock, backtest_results)
                    
                    # Log recent trades
                    if not trades.empty:
                        latest_trade = trades.iloc[-1]
                        self.sheets_logger.log_trade(
                            stock=stock,
                            action=latest_trade['Trade_Type'],
                            price=latest_trade['Close'],
                            rsi=latest_trade.get('RSI'),
                            ma_short=latest_trade.get('MA_Short'),
                            ma_long=latest_trade.get('MA_Long'),
                            ml_prediction=prediction['prediction'] if prediction else None,
                            confidence=prediction['confidence'] if prediction else None
                        )
                    
                    print("✅ Results logged to Google Sheets")
                    
                except Exception as e:
                    error_msg = f"Error logging to Google Sheets: {e}"
                    print(f"❌ {error_msg}")
                    self.send_alert('error',
                                   error_type='Google Sheets Error',
                                   error_message=error_msg,
                                   component='Sheets Logger')
                
                # Store results
                results_summary[stock] = {
                    'backtest_results': backtest_results,
                    'ml_results': ml_results,
                    'prediction': prediction
                }
                
                # Update totals
                total_trades += backtest_results['total_trades']
                total_pnl += backtest_results.get('total_pnl', 0)
                successful_analyses += 1
                
                # Small delay between stocks to avoid overwhelming Telegram
                time.sleep(1)
                
            except Exception as e:
                error_msg = f"Error analyzing {stock}: {e}"
                print(f"❌ {error_msg}")
                self.send_alert('error',
                               error_type='Analysis Error',
                               error_message=error_msg,
                               component='Stock Analyzer')
                continue
        
        # Calculate performance metrics
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Calculate overall win rate
        total_winning_trades = sum(r['backtest_results'].get('winning_trades', 0) for r in results_summary.values())
        overall_win_rate = (total_winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate ML accuracy
        ml_accuracies = [r['ml_results']['accuracy'] for r in results_summary.values() if r['ml_results']]
        avg_ml_accuracy = sum(ml_accuracies) / len(ml_accuracies) if ml_accuracies else 0
        
        # Find best and worst trades
        all_trades = []
        for r in results_summary.values():
            trades = r['backtest_results']['trades']
            if not trades.empty:
                all_trades.extend(trades['PnL'].tolist())
        
        best_trade = max(all_trades) if all_trades else 0
        worst_trade = min(all_trades) if all_trades else 0
        
        # Send performance summary
        performance_data = {
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'win_rate': overall_win_rate,
            'ml_accuracy': avg_ml_accuracy,
            'duration': duration,
            'best_trade': best_trade,
            'worst_trade': worst_trade
        }
        
        self.send_alert('performance', performance_data)
        
        # Send completion status
        self.send_alert('status',
                       status='HEALTHY',
                       details=f'Analysis completed: {successful_analyses}/{len(self.stocks)} stocks processed successfully')
        
        # Print summary
        self._print_summary(results_summary, performance_data)
        
        return results_summary
    
    def _print_summary(self, results, performance_data):
        """Print analysis summary with performance metrics"""
        print("\n" + "="*60)
        print("ENHANCED ANALYSIS SUMMARY")
        print("="*60)
        
        if not results:
            print("No successful analyses completed.")
            return
        
        print(f"{'Stock':<12} {'Return':<10} {'Win Rate':<10} {'Trades':<8} {'ML Acc':<8} {'Next Day':<10}")
        print("-" * 60)
        
        for stock, data in results.items():
            backtest = data['backtest_results']
            ml_results = data['ml_results']
            prediction = data['prediction']
            
            return_str = f"{backtest['total_return']:.1%}"
            win_rate_str = f"{backtest['win_rate']:.1f}%"
            trades_str = str(backtest['total_trades'])
            ml_acc_str = f"{ml_results['accuracy']:.3f}" if ml_results else "N/A"
            next_day_str = "UP" if prediction and prediction['prediction'] == 1 else "DOWN" if prediction else "N/A"
            
            print(f"{stock:<12} {return_str:<10} {win_rate_str:<10} {trades_str:<8} {ml_acc_str:<8} {next_day_str:<10}")
        
        print("\n" + "="*60)
        print("OVERALL PERFORMANCE")
        print("="*60)
        print(f"Total Trades: {performance_data['total_trades']}")
        print(f"Total P&L: ₹{performance_data['total_pnl']:+,.2f}")
        print(f"Overall Win Rate: {performance_data['win_rate']:.1f}%")
        print(f"Average ML Accuracy: {performance_data['ml_accuracy']:.1%}")
        print(f"Analysis Duration: {performance_data['duration']:.1f} seconds")
        print(f"Best Trade: ₹{performance_data['best_trade']:+.2f}")
        print(f"Worst Trade: ₹{performance_data['worst_trade']:+.2f}")
        
        print(f"\n✅ Analysis complete! {'📱 Telegram alerts sent!' if self.telegram_alerts.is_configured() else ''}")
        print("📊 Check Google Sheets for detailed logs.")

def main():
    """Main function"""
    system = EnhancedAlgoTradingSystem()
    results = system.run_analysis()
    return results

if __name__ == "__main__":
    main()