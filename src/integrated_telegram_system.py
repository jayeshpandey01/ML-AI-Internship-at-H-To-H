#!/usr/bin/env python3
"""
Integrated Telegram System - Enhanced Automated System with Telegram Alerts
Combines the automated algo system with comprehensive Telegram notifications
"""

import sys
import os
import time
import schedule
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
import traceback
import json

# Add src directory to path
sys.path.append('src')

from data_ingestion import EnhancedDataIngestion
from ml_enhanced_strategy import MLEnhancedTradingStrategy
from google_sheets_automation import GoogleSheetsAutomation
from telegram_alerts import TelegramAlertsSystem

# Load environment variables
load_dotenv('config/config.env')

class IntegratedTelegramSystem:
    def __init__(self):
        """
        Initialize the Integrated System with Telegram Alerts
        """
        # Set up comprehensive logging
        self.setup_logging()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 Initializing Integrated Telegram System")
        
        # Initialize system components
        try:
            self.data_ingestion = EnhancedDataIngestion()
            self.ml_strategy = MLEnhancedTradingStrategy(
                ml_model_type='random_forest',
                ml_confidence_threshold=0.6,
                rsi_oversold=35,
                rsi_overbought=65,
                position_size=10000,
                holding_period=5
            )
            self.sheets_automation = GoogleSheetsAutomation()
            self.telegram_alerts = TelegramAlertsSystem()
            
            # Get stocks from config
            self.stocks = os.getenv('NIFTY_STOCKS', 'RELIANCE.NS,HDFCBANK.NS,INFY.NS').split(',')
            
            # Telegram alert settings
            self.enable_trade_alerts = os.getenv('TELEGRAM_ENABLE_TRADE_ALERTS', 'true').lower() == 'true'
            self.enable_error_alerts = os.getenv('TELEGRAM_ENABLE_ERROR_ALERTS', 'true').lower() == 'true'
            self.enable_status_alerts = os.getenv('TELEGRAM_ENABLE_STATUS_ALERTS', 'true').lower() == 'true'
            self.enable_ml_alerts = os.getenv('TELEGRAM_ENABLE_ML_ALERTS', 'true').lower() == 'true'
            
            # System status tracking
            self.last_run_time = None
            self.run_count = 0
            self.error_count = 0
            self.success_count = 0
            
            self.logger.info(f"✅ System initialized successfully")
            self.logger.info(f"📊 Monitoring stocks: {', '.join(self.stocks)}")
            self.logger.info(f"📱 Telegram alerts: {'✅ Enabled' if self.telegram_alerts.is_configured() else '❌ Disabled'}")
            
            # Send startup notification
            if self.enable_status_alerts and self.telegram_alerts.is_configured():
                self.telegram_alerts.send_system_status_alert(
                    'STARTED',
                    f'Monitoring {len(self.stocks)} stocks with ML-enhanced strategy'
                )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize system: {e}")
            if self.enable_error_alerts and hasattr(self, 'telegram_alerts') and self.telegram_alerts.is_configured():
                self.telegram_alerts.send_error_alert(
                    'System Initialization Error',
                    str(e),
                    'System Startup'
                )
            raise
    
    def setup_logging(self):
        """Set up comprehensive logging system"""
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        file_handler = logging.FileHandler(
            f'logs/integrated_system_{datetime.now().strftime("%Y%m%d")}.log'
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format))
        
        error_handler = logging.FileHandler(
            f'logs/integrated_errors_{datetime.now().strftime("%Y%m%d")}.log'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter(log_format))
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        
        logging.basicConfig(
            level=logging.INFO,
            handlers=[file_handler, error_handler, console_handler]
        )
        
        print(f"📝 Logging configured - Files: logs/integrated_system_{datetime.now().strftime('%Y%m%d')}.log")
    
    def auto_triggered_function(self):
        """Main orchestration function with Telegram integration"""
        run_id = f"RUN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger.info(f"🔄 Starting automated run: {run_id}")
        
        try:
            self.run_count += 1
            start_time = datetime.now()
            
            # Step 1: Fetch latest stock data
            self.logger.info("📊 Step 1: Fetching latest stock data...")
            stock_data = self._fetch_latest_data()
            
            if not stock_data:
                raise Exception("No stock data available")
            
            # Step 2: Run trading strategy and generate signals
            self.logger.info("🎯 Step 2: Running ML-enhanced trading strategy...")
            strategy_results = self._run_trading_strategy_with_alerts(stock_data)
            
            # Step 3: Log results to Google Sheets
            self.logger.info("📈 Step 3: Logging results to Google Sheets...")
            self._log_to_google_sheets(strategy_results)
            
            # Step 4: Generate and send performance summary
            self._generate_and_send_summary(run_id, strategy_results, start_time)
            
            self.success_count += 1
            self.last_run_time = datetime.now()
            
            self.logger.info(f"✅ Automated run completed successfully: {run_id}")
            
            # Send success notification
            if self.enable_status_alerts and self.telegram_alerts.is_configured():
                total_trades = sum(r.get('total_trades', 0) for r in strategy_results.values())
                total_pnl = sum(r.get('total_pnl', 0) for r in strategy_results.values())
                
                self.telegram_alerts.send_system_status_alert(
                    'HEALTHY',
                    f'Run completed: {total_trades} trades, ₹{total_pnl:+.2f} P&L'
                )
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"❌ Automated run failed: {run_id} - {str(e)}")
            self.logger.error(f"📋 Error traceback: {traceback.format_exc()}")
            
            # Send error alert
            if self.enable_error_alerts and self.telegram_alerts.is_configured():
                self.telegram_alerts.send_error_alert(
                    'Automated Run Failed',
                    str(e),
                    'Main System'
                )
            
            self.logger.info("🔄 System will continue with next scheduled run")
    
    def _fetch_latest_data(self):
        """Fetch latest stock data with error handling and alerts"""
        try:
            self.logger.info("📡 Fetching data from multiple sources...")
            stock_data = {}
            
            for stock in self.stocks:
                try:
                    data = self.data_ingestion.load_saved_data(stock, 'processed')
                    
                    if data is None or len(data) < 50:
                        self.logger.info(f"🔄 Fetching fresh data for {stock}...")
                        fresh_data = self.data_ingestion.fetch_yfinance_data(stock, period='6mo')
                        
                        if fresh_data is not None:
                            processed_data = self.data_ingestion.calculate_technical_indicators(fresh_data)
                            self.data_ingestion.save_processed_data(processed_data, stock)
                            stock_data[stock] = processed_data
                            self.logger.info(f"✅ Fresh data fetched and processed for {stock}")
                        else:
                            self.logger.warning(f"⚠️ Failed to fetch fresh data for {stock}")
                            if self.enable_error_alerts and self.telegram_alerts.is_configured():
                                self.telegram_alerts.send_error_alert(
                                    'Data Fetch Failed',
                                    f'Could not fetch data for {stock}',
                                    'Data Ingestion'
                                )
                    else:
                        stock_data[stock] = data
                        self.logger.info(f"📂 Using existing processed data for {stock}")
                        
                except Exception as e:
                    self.logger.error(f"❌ Error processing {stock}: {e}")
                    if self.enable_error_alerts and self.telegram_alerts.is_configured():
                        self.telegram_alerts.send_error_alert(
                            'Data Processing Error',
                            f'Error processing {stock}: {str(e)}',
                            'Data Processing'
                        )
                    continue
            
            self.logger.info(f"📊 Data fetching completed: {len(stock_data)} stocks processed")
            return stock_data
            
        except Exception as e:
            self.logger.error(f"❌ Critical error in data fetching: {e}")
            if self.enable_error_alerts and self.telegram_alerts.is_configured():
                self.telegram_alerts.send_error_alert(
                    'Critical Data Error',
                    str(e),
                    'Data Fetching'
                )
            raise
    
    def _run_trading_strategy_with_alerts(self, stock_data):
        """Run ML-enhanced trading strategy with Telegram alerts for signals"""
        try:
            strategy_results = {}
            
            for stock, data in stock_data.items():
                try:
                    self.logger.info(f"🎯 Running strategy for {stock}...")
                    
                    # Run ML-enhanced strategy
                    results = self.ml_strategy.backtest_enhanced_strategy(data, stock)
                    
                    # Send trade alerts for new trades
                    if self.enable_trade_alerts and self.telegram_alerts.is_configured():
                        self._send_trade_alerts(stock, results)
                    
                    # Send ML prediction alerts
                    if self.enable_ml_alerts and self.telegram_alerts.is_configured():
                        self._send_ml_prediction_alert(stock, data)
                    
                    self.logger.info(f"📊 {stock} Results: {results['total_trades']} trades, "
                                   f"{results['win_ratio']:.1f}% win rate, "
                                   f"₹{results['total_pnl']:+.2f} P&L")
                    
                    strategy_results[stock] = results
                    
                    # Save individual results
                    if results['total_trades'] > 0:
                        results['trades_df'].to_csv(
                            f'data/{stock.replace(".", "_")}_integrated_trades_{datetime.now().strftime("%Y%m%d")}.csv',
                            index=False
                        )
                        self.logger.info(f"💾 Trade details saved for {stock}")
                    
                except Exception as e:
                    self.logger.error(f"❌ Strategy execution failed for {stock}: {e}")
                    if self.enable_error_alerts and self.telegram_alerts.is_configured():
                        self.telegram_alerts.send_error_alert(
                            'Strategy Execution Error',
                            f'Error running strategy for {stock}: {str(e)}',
                            'Trading Strategy'
                        )
                    continue
            
            self.logger.info(f"🎯 Strategy execution completed for {len(strategy_results)} stocks")
            return strategy_results
            
        except Exception as e:
            self.logger.error(f"❌ Critical error in strategy execution: {e}")
            if self.enable_error_alerts and self.telegram_alerts.is_configured():
                self.telegram_alerts.send_error_alert(
                    'Critical Strategy Error',
                    str(e),
                    'Strategy Execution'
                )
            raise
    
    def _send_trade_alerts(self, stock, results):
        """Send Telegram alerts for trading signals"""
        try:
            if 'trades_df' in results and not results['trades_df'].empty:
                # Send alerts for recent trades (last 5)
                recent_trades = results['trades_df'].tail(5)
                
                for _, trade in recent_trades.iterrows():
                    # Determine if this is entry or exit
                    if pd.notna(trade.get('buy_reason')):
                        # Entry signal
                        self.telegram_alerts.send_trade_signal_alert(
                            stock=stock,
                            signal_type='BUY',
                            price=trade['entry_price'],
                            reason=trade['buy_reason'],
                            ml_confidence=trade.get('entry_ml_confidence')
                        )
                    
                    if pd.notna(trade.get('sell_reason')):
                        # Exit signal with P&L
                        profit_emoji = "💰" if trade['profitable'] else "📉"
                        self.telegram_alerts.send_message(f"""
{profit_emoji} <b>TRADE COMPLETED</b> {profit_emoji}

📊 <b>Stock:</b> {stock}
🔴 <b>Action:</b> SELL
💰 <b>Price:</b> ₹{trade['exit_price']:.2f}
📈 <b>P&L:</b> ₹{trade['pnl']:+.2f} ({trade['pnl_percent']:+.2f}%)
📅 <b>Days Held:</b> {trade['days_held']}
📋 <b>Reason:</b> {trade['sell_reason']}
⏰ <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")
                    
                    time.sleep(1)  # Rate limiting
                    
        except Exception as e:
            self.logger.error(f"Error sending trade alerts for {stock}: {e}")
    
    def _send_ml_prediction_alert(self, stock, data):
        """Send ML prediction alert for next-day movement"""
        try:
            if hasattr(self.ml_strategy.ml_system, 'is_trained') and self.ml_strategy.ml_system.is_trained:
                prediction = self.ml_strategy.ml_system.predict_next_day(data, stock)
                
                if prediction and prediction['confidence'] > 0.6:  # Only send high-confidence predictions
                    self.telegram_alerts.send_ml_prediction_alert(
                        stock=stock,
                        prediction=prediction['direction'],
                        confidence=prediction['confidence'],
                        current_price=data['Close'].iloc[-1]
                    )
                    
        except Exception as e:
            self.logger.error(f"Error sending ML prediction alert for {stock}: {e}")
    
    def _log_to_google_sheets(self, strategy_results):
        """Log results to Google Sheets with error handling"""
        try:
            if not self.sheets_automation.authenticate():
                self.logger.warning("⚠️ Google Sheets authentication failed - skipping sheets update")
                return
            
            self.logger.info("📈 Updating Google Sheets...")
            
            for stock, results in strategy_results.items():
                try:
                    self.sheets_automation.automate_full_update(results, stock)
                    self.logger.info(f"✅ Google Sheets updated for {stock}")
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed to update Google Sheets for {stock}: {e}")
                    if self.enable_error_alerts and self.telegram_alerts.is_configured():
                        self.telegram_alerts.send_error_alert(
                            'Google Sheets Error',
                            f'Failed to update sheets for {stock}: {str(e)}',
                            'Google Sheets'
                        )
                    continue
            
        except Exception as e:
            self.logger.error(f"❌ Critical error in Google Sheets logging: {e}")
    
    def _generate_and_send_summary(self, run_id, strategy_results, start_time):
        """Generate run summary and send performance alert"""
        try:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Calculate summary metrics
            total_trades = sum(r.get('total_trades', 0) for r in strategy_results.values())
            total_pnl = sum(r.get('total_pnl', 0) for r in strategy_results.values())
            winning_trades = sum(r.get('winning_trades', 0) for r in strategy_results.values())
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Get best and worst trades
            all_trades = []
            for results in strategy_results.values():
                if 'trades_df' in results and not results['trades_df'].empty:
                    all_trades.extend(results['trades_df']['pnl'].tolist())
            
            best_trade = max(all_trades) if all_trades else 0
            worst_trade = min(all_trades) if all_trades else 0
            
            # Calculate average ML accuracy
            ml_accuracies = [r.get('ml_accuracy', 0) for r in strategy_results.values() if 'ml_accuracy' in r]
            avg_ml_accuracy = sum(ml_accuracies) / len(ml_accuracies) if ml_accuracies else 0
            
            summary = {
                'run_id': run_id,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'stocks_processed': len(strategy_results),
                'total_trades': total_trades,
                'total_pnl': total_pnl,
                'win_rate': win_rate,
                'ml_accuracy': avg_ml_accuracy,
                'best_trade': best_trade,
                'worst_trade': worst_trade,
                'system_stats': {
                    'total_runs': self.run_count,
                    'success_runs': self.success_count,
                    'error_runs': self.error_count,
                    'success_rate': (self.success_count / self.run_count * 100) if self.run_count > 0 else 0
                }
            }
            
            # Save summary to file
            summary_file = f'logs/integrated_summary_{run_id}.json'
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.logger.info(f"📋 Run summary saved: {summary_file}")
            self.logger.info(f"⏱️ Run duration: {duration:.2f} seconds")
            
            # Send performance summary via Telegram
            if self.enable_status_alerts and self.telegram_alerts.is_configured():
                self.telegram_alerts.send_performance_summary({
                    'total_trades': total_trades,
                    'total_pnl': total_pnl,
                    'win_rate': win_rate,
                    'ml_accuracy': avg_ml_accuracy,
                    'duration': duration,
                    'best_trade': best_trade,
                    'worst_trade': worst_trade
                })
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate run summary: {e}")
    
    def run_once(self):
        """Run the system once with Telegram notifications"""
        self.logger.info("🔄 Manual execution requested")
        self.auto_triggered_function()
    
    def start_scheduled_execution(self, schedule_time="09:30"):
        """Start scheduled execution with Telegram notifications"""
        self.logger.info(f"⏰ Starting scheduled execution at {schedule_time} daily")
        
        # Schedule daily execution
        schedule.every().day.at(schedule_time).do(self.auto_triggered_function)
        schedule.every().day.at("15:30").do(self._market_close_update)
        schedule.every().monday.at("08:00").do(self._weekly_summary)
        schedule.every(30).minutes.do(self._system_health_check)
        
        print(f"📅 Scheduled Tasks with Telegram Alerts:")
        print(f"   🌅 Daily Trading: {schedule_time}")
        print(f"   🌆 Market Close Update: 15:30")
        print(f"   📊 Weekly Summary: Monday 08:00")
        print(f"   💚 Health Check: Every 30 minutes")
        
        # Send startup notification
        if self.enable_status_alerts and self.telegram_alerts.is_configured():
            self.telegram_alerts.send_system_status_alert(
                'STARTED',
                f'Scheduled execution started at {schedule_time}'
            )
        
        try:
            self.logger.info("🚀 Scheduler started - Press Ctrl+C to stop")
            
            while True:
                schedule.run_pending()
                time.sleep(60)
                
        except KeyboardInterrupt:
            self.logger.info("⏹️ Scheduler stopped by user")
            if self.enable_status_alerts and self.telegram_alerts.is_configured():
                self.telegram_alerts.send_system_status_alert('STOPPED', 'System stopped by user')
        except Exception as e:
            self.logger.error(f"❌ Scheduler error: {e}")
            if self.enable_error_alerts and self.telegram_alerts.is_configured():
                self.telegram_alerts.send_error_alert(
                    'Scheduler Error',
                    str(e),
                    'Scheduler'
                )
            raise
    
    def _market_close_update(self):
        """Market close update with Telegram notifications"""
        self.logger.info("🌆 Running market close update...")
        try:
            predictions_sent = 0
            for stock in self.stocks:
                data = self.data_ingestion.load_saved_data(stock, 'processed')
                if data is not None and hasattr(self.ml_strategy.ml_system, 'is_trained'):
                    if self.ml_strategy.ml_system.is_trained:
                        prediction = self.ml_strategy.ml_system.predict_next_day(data, stock)
                        if prediction and self.enable_ml_alerts and self.telegram_alerts.is_configured():
                            self.telegram_alerts.send_ml_prediction_alert(
                                stock=stock,
                                prediction=prediction['direction'],
                                confidence=prediction['confidence'],
                                current_price=data['Close'].iloc[-1]
                            )
                            predictions_sent += 1
                            time.sleep(1)  # Rate limiting
            
            self.logger.info(f"✅ Market close update completed - {predictions_sent} predictions sent")
            
        except Exception as e:
            self.logger.error(f"❌ Market close update failed: {e}")
            if self.enable_error_alerts and self.telegram_alerts.is_configured():
                self.telegram_alerts.send_error_alert(
                    'Market Close Update Error',
                    str(e),
                    'Market Close Update'
                )
    
    def _weekly_summary(self):
        """Generate weekly performance summary with Telegram notification"""
        self.logger.info("📊 Generating weekly summary...")
        try:
            summary = {
                'week_ending': datetime.now().strftime('%Y-%m-%d'),
                'system_stats': {
                    'total_runs': self.run_count,
                    'successful_runs': self.success_count,
                    'failed_runs': self.error_count,
                    'success_rate': (self.success_count / self.run_count * 100) if self.run_count > 0 else 0
                },
                'last_run': self.last_run_time.isoformat() if self.last_run_time else None
            }
            
            weekly_file = f'logs/weekly_summary_{datetime.now().strftime("%Y%m%d")}.json'
            with open(weekly_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.logger.info(f"📋 Weekly summary saved: {weekly_file}")
            
            # Send weekly summary via Telegram
            if self.enable_status_alerts and self.telegram_alerts.is_configured():
                self.telegram_alerts.send_message(f"""
📊 <b>WEEKLY SUMMARY</b> 📊

📅 <b>Week Ending:</b> {summary['week_ending']}
🔄 <b>Total Runs:</b> {summary['system_stats']['total_runs']}
✅ <b>Successful Runs:</b> {summary['system_stats']['successful_runs']}
❌ <b>Failed Runs:</b> {summary['system_stats']['failed_runs']}
📈 <b>Success Rate:</b> {summary['system_stats']['success_rate']:.1f}%
⏰ <b>Last Run:</b> {summary['last_run'] or 'Never'}

🚀 System is operating smoothly!
""")
            
        except Exception as e:
            self.logger.error(f"❌ Weekly summary failed: {e}")
    
    def _system_health_check(self):
        """Perform system health check with Telegram alerts"""
        try:
            health_status = {
                'timestamp': datetime.now().isoformat(),
                'total_runs': self.run_count,
                'success_rate': (self.success_count / self.run_count * 100) if self.run_count > 0 else 0,
                'last_run': self.last_run_time.isoformat() if self.last_run_time else None,
                'telegram_configured': self.telegram_alerts.is_configured()
            }
            
            self.logger.info(f"💚 System Health: {health_status['success_rate']:.1f}% success rate, "
                           f"{self.run_count} total runs")
            
            # Send health alert if there are issues
            if health_status['success_rate'] < 80 and self.run_count > 5:
                if self.enable_status_alerts and self.telegram_alerts.is_configured():
                    self.telegram_alerts.send_system_status_alert(
                        'WARNING',
                        f'Low success rate: {health_status["success_rate"]:.1f}%'
                    )
            
        except Exception as e:
            self.logger.error(f"❌ Health check failed: {e}")

def main():
    """Main function to start the integrated system"""
    print("🚀 INTEGRATED TELEGRAM SYSTEM - STEP 7")
    print("=" * 80)
    
    try:
        # Initialize the integrated system
        system = IntegratedTelegramSystem()
        
        # Check command line arguments
        if len(sys.argv) > 1:
            command = sys.argv[1].lower()
            
            if command == 'once':
                print("🔄 Running system once with Telegram alerts...")
                system.run_once()
                
            elif command == 'schedule':
                schedule_time = sys.argv[2] if len(sys.argv) > 2 else "09:30"
                print(f"⏰ Starting scheduled execution with Telegram alerts at {schedule_time}...")
                system.start_scheduled_execution(schedule_time)
                
            elif command == 'test':
                print("🧪 Testing Telegram alerts...")
                system.telegram_alerts.send_test_message()
                
            else:
                print("❌ Unknown command. Use 'once', 'schedule [time]', or 'test'")
                
        else:
            print("🔄 Running system once (default) with Telegram alerts...")
            print("💡 Use 'python src/integrated_telegram_system.py schedule' for continuous operation")
            print("💡 Use 'python src/integrated_telegram_system.py test' to test Telegram alerts")
            system.run_once()
        
    except KeyboardInterrupt:
        print("\n⏹️ System stopped by user")
    except Exception as e:
        print(f"❌ System error: {e}")
        logging.error(f"Critical system error: {e}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()