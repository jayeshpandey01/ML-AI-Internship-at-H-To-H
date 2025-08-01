#!/usr/bin/env python3
"""
Google Sheets Automation Module
Comprehensive integration with Google Sheets API for trade logging and analytics
"""

import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
import json
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv('config/config.env')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoogleSheetsAutomation:
    def __init__(self):
        """
        Initialize Google Sheets Automation
        """
        self.credentials_path = os.getenv('GOOGLE_SHEETS_CREDENTIALS_PATH', 'config/google_credentials.json')
        self.sheet_id = os.getenv('GOOGLE_SHEET_ID')
        self.gc = None
        self.sheet = None
        self.rate_limit_delay = 1  # Seconds between API calls
        
        print(f"📊 Google Sheets Automation Initialized:")
        print(f"   Credentials Path: {self.credentials_path}")
        print(f"   Sheet ID: {self.sheet_id}")
        
    def authenticate(self):
        """
        Authenticate with Google Sheets API
        """
        try:
            # Define the scope
            scope = [
                'https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive',
                'https://www.googleapis.com/auth/spreadsheets'
            ]
            
            # Check if credentials file exists
            if not os.path.exists(self.credentials_path):
                logger.error(f"Credentials file not found: {self.credentials_path}")
                print(f"❌ Credentials file not found: {self.credentials_path}")
                print("📋 To set up Google Sheets integration:")
                print("   1. Go to Google Cloud Console")
                print("   2. Enable Google Sheets API")
                print("   3. Create service account credentials")
                print("   4. Download JSON file and place in config/")
                print("   5. Update GOOGLE_SHEET_ID in config/config.env")
                return False
            
            # Authenticate
            creds = Credentials.from_service_account_file(self.credentials_path, scopes=scope)
            self.gc = gspread.authorize(creds)
            
            # Open the spreadsheet
            if not self.sheet_id:
                logger.error("Google Sheet ID not provided")
                print("❌ Google Sheet ID not provided in config/config.env")
                return False
            
            self.sheet = self.gc.open_by_key(self.sheet_id)
            
            logger.info("Successfully authenticated with Google Sheets")
            print("✅ Successfully authenticated with Google Sheets")
            return True
            
        except Exception as e:
            logger.error(f"Error authenticating with Google Sheets: {e}")
            print(f"❌ Error authenticating with Google Sheets: {e}")
            return False
    
    def setup_worksheets(self):
        """
        Set up the required worksheets with proper headers
        """
        if not self.authenticate():
            return False
        
        try:
            print("🔧 Setting up worksheets...")
            
            # 1. Trade Log Worksheet
            self._setup_trade_log_worksheet()
            
            # 2. P&L Summary Worksheet
            self._setup_pnl_summary_worksheet()
            
            # 3. Win Ratio Worksheet
            self._setup_win_ratio_worksheet()
            
            # 4. ML Analytics Worksheet
            self._setup_ml_analytics_worksheet()
            
            # 5. Portfolio Dashboard Worksheet
            self._setup_portfolio_dashboard_worksheet()
            
            print("✅ All worksheets set up successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up worksheets: {e}")
            print(f"❌ Error setting up worksheets: {e}")
            return False
    
    def _setup_trade_log_worksheet(self):
        """Set up Trade Log worksheet"""
        try:
            worksheet = self.sheet.worksheet('Trade Log')
        except gspread.WorksheetNotFound:
            worksheet = self.sheet.add_worksheet(title='Trade Log', rows=1000, cols=20)
        
        # Headers for trade log
        headers = [
            'Timestamp', 'Stock', 'Signal_Type', 'Entry_Date', 'Exit_Date',
            'Entry_Price', 'Exit_Price', 'Shares', 'Days_Held', 'Investment',
            'Exit_Value', 'PnL', 'PnL_Percent', 'Entry_RSI', 'Exit_RSI',
            'ML_Prediction', 'ML_Confidence', 'Buy_Reason', 'Sell_Reason', 'Profitable'
        ]
        
        worksheet.update('A1:T1', [headers])
        
        # Format headers
        worksheet.format('A1:T1', {
            'backgroundColor': {'red': 0.2, 'green': 0.6, 'blue': 1.0},
            'textFormat': {'bold': True, 'foregroundColor': {'red': 1, 'green': 1, 'blue': 1}}
        })
        
        time.sleep(self.rate_limit_delay)
        print("   ✅ Trade Log worksheet configured")
    
    def _setup_pnl_summary_worksheet(self):
        """Set up P&L Summary worksheet"""
        try:
            worksheet = self.sheet.worksheet('P&L Summary')
        except gspread.WorksheetNotFound:
            worksheet = self.sheet.add_worksheet(title='P&L Summary', rows=100, cols=15)
        
        # Headers for P&L summary
        headers = [
            'Stock', 'Strategy_Type', 'Total_Trades', 'Winning_Trades', 'Losing_Trades',
            'Win_Ratio', 'Total_PnL', 'Avg_PnL_Trade', 'Strategy_Return', 'Market_Return',
            'Alpha', 'Sharpe_Ratio', 'ML_Accuracy', 'Last_Updated', 'Status'
        ]
        
        worksheet.update('A1:O1', [headers])
        
        # Format headers
        worksheet.format('A1:O1', {
            'backgroundColor': {'red': 0.0, 'green': 0.7, 'blue': 0.3},
            'textFormat': {'bold': True, 'foregroundColor': {'red': 1, 'green': 1, 'blue': 1}}
        })
        
        time.sleep(self.rate_limit_delay)
        print("   ✅ P&L Summary worksheet configured")
    
    def _setup_win_ratio_worksheet(self):
        """Set up Win Ratio worksheet"""
        try:
            worksheet = self.sheet.worksheet('Win Ratio')
        except gspread.WorksheetNotFound:
            worksheet = self.sheet.add_worksheet(title='Win Ratio', rows=100, cols=10)
        
        # Headers for win ratio
        headers = [
            'Stock', 'Strategy_Type', 'Total_Trades', 'Winning_Trades',
            'Win_Percentage', 'Best_Trade', 'Worst_Trade', 'Avg_Hold_Days',
            'Last_Updated', 'Performance_Grade'
        ]
        
        worksheet.update('A1:J1', [headers])
        
        # Format headers
        worksheet.format('A1:J1', {
            'backgroundColor': {'red': 1.0, 'green': 0.6, 'blue': 0.0},
            'textFormat': {'bold': True, 'foregroundColor': {'red': 1, 'green': 1, 'blue': 1}}
        })
        
        time.sleep(self.rate_limit_delay)
        print("   ✅ Win Ratio worksheet configured")
    
    def _setup_ml_analytics_worksheet(self):
        """Set up ML Analytics worksheet"""
        try:
            worksheet = self.sheet.worksheet('ML Analytics')
        except gspread.WorksheetNotFound:
            worksheet = self.sheet.add_worksheet(title='ML Analytics', rows=200, cols=12)
        
        # Headers for ML analytics
        headers = [
            'Date', 'Stock', 'Close_Price', 'RSI', 'ML_Prediction',
            'ML_Confidence', 'Prediction_Direction', 'Actual_Direction',
            'Prediction_Correct', 'Traditional_Signal', 'Enhanced_Signal', 'Notes'
        ]
        
        worksheet.update('A1:L1', [headers])
        
        # Format headers
        worksheet.format('A1:L1', {
            'backgroundColor': {'red': 0.6, 'green': 0.2, 'blue': 1.0},
            'textFormat': {'bold': True, 'foregroundColor': {'red': 1, 'green': 1, 'blue': 1}}
        })
        
        time.sleep(self.rate_limit_delay)
        print("   ✅ ML Analytics worksheet configured")
    
    def _setup_portfolio_dashboard_worksheet(self):
        """Set up Portfolio Dashboard worksheet"""
        try:
            worksheet = self.sheet.worksheet('Portfolio Dashboard')
        except gspread.WorksheetNotFound:
            worksheet = self.sheet.add_worksheet(title='Portfolio Dashboard', rows=50, cols=8)
        
        # Create dashboard structure
        dashboard_data = [
            ['ALGO-TRADING PORTFOLIO DASHBOARD', '', '', '', '', '', '', ''],
            ['', '', '', '', '', '', '', ''],
            ['Portfolio Metrics', 'Value', '', 'Strategy Comparison', 'Traditional', 'ML-Enhanced', 'Improvement', ''],
            ['Total Investment', '₹0', '', 'Total Trades', '0', '0', '0%', ''],
            ['Current Value', '₹0', '', 'Win Ratio', '0%', '0%', '0%', ''],
            ['Total P&L', '₹0', '', 'Total P&L', '₹0', '₹0', '0%', ''],
            ['Portfolio Return', '0%', '', 'Avg P&L/Trade', '₹0', '₹0', '0%', ''],
            ['', '', '', '', '', '', '', ''],
            ['Stock Performance', '', '', '', '', '', '', ''],
            ['Stock', 'Trades', 'Win%', 'P&L', 'Return%', 'ML Acc%', 'Status', ''],
        ]
        
        worksheet.update('A1:H10', dashboard_data)
        
        # Format dashboard
        worksheet.format('A1:H1', {
            'backgroundColor': {'red': 0.1, 'green': 0.1, 'blue': 0.1},
            'textFormat': {'bold': True, 'fontSize': 14, 'foregroundColor': {'red': 1, 'green': 1, 'blue': 1}}
        })
        
        worksheet.format('A3:H3', {
            'backgroundColor': {'red': 0.8, 'green': 0.8, 'blue': 0.8},
            'textFormat': {'bold': True}
        })
        
        worksheet.format('A9:H10', {
            'backgroundColor': {'red': 0.9, 'green': 0.9, 'blue': 0.9},
            'textFormat': {'bold': True}
        })
        
        time.sleep(self.rate_limit_delay)
        print("   ✅ Portfolio Dashboard worksheet configured")
    
    def log_trade_signal(self, trade_data):
        """
        Log individual trade signal to Trade Log worksheet
        
        Parameters:
        - trade_data: Dictionary containing trade information
        """
        try:
            worksheet = self.sheet.worksheet('Trade Log')
            
            # Prepare row data
            row_data = [
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  # Timestamp
                trade_data.get('stock', ''),
                trade_data.get('signal_type', ''),
                trade_data.get('entry_date', ''),
                trade_data.get('exit_date', ''),
                trade_data.get('entry_price', 0),
                trade_data.get('exit_price', 0),
                trade_data.get('shares', 0),
                trade_data.get('days_held', 0),
                trade_data.get('investment', 0),
                trade_data.get('exit_value', 0),
                trade_data.get('pnl', 0),
                trade_data.get('pnl_percent', 0),
                trade_data.get('entry_rsi', ''),
                trade_data.get('exit_rsi', ''),
                trade_data.get('ml_prediction', ''),
                trade_data.get('ml_confidence', ''),
                trade_data.get('buy_reason', ''),
                trade_data.get('sell_reason', ''),
                trade_data.get('profitable', False)
            ]
            
            worksheet.append_row(row_data)
            time.sleep(self.rate_limit_delay)
            
            logger.info(f"Trade logged: {trade_data.get('stock')} - {trade_data.get('signal_type')}")
            print(f"📝 Trade logged: {trade_data.get('stock')} - {trade_data.get('signal_type')}")
            
        except Exception as e:
            logger.error(f"Error logging trade: {e}")
            print(f"❌ Error logging trade: {e}")
    
    def update_pnl_summary(self, stock, strategy_results):
        """
        Update P&L Summary worksheet with strategy results
        
        Parameters:
        - stock: Stock symbol
        - strategy_results: Dictionary containing strategy performance metrics
        """
        try:
            worksheet = self.sheet.worksheet('P&L Summary')
            
            # Check if stock already exists
            try:
                cell = worksheet.find(stock)
                row_num = cell.row
                update_existing = True
            except gspread.CellNotFound:
                row_num = len(worksheet.get_all_values()) + 1
                update_existing = False
            
            # Prepare row data
            row_data = [
                stock,
                strategy_results.get('strategy_type', 'ML-Enhanced'),
                strategy_results.get('total_trades', 0),
                strategy_results.get('winning_trades', 0),
                strategy_results.get('losing_trades', 0),
                f"{strategy_results.get('win_ratio', 0):.1f}%",
                f"₹{strategy_results.get('total_pnl', 0):,.2f}",
                f"₹{strategy_results.get('avg_pnl_per_trade', 0):,.2f}",
                f"{strategy_results.get('strategy_return', 0):+.2f}%",
                f"{strategy_results.get('market_return', 0):+.2f}%",
                f"{strategy_results.get('strategy_return', 0) - strategy_results.get('market_return', 0):+.2f}%",
                f"{strategy_results.get('sharpe_ratio', 0):.3f}",
                f"{strategy_results.get('ml_accuracy', 0):.1%}" if 'ml_accuracy' in strategy_results else 'N/A',
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Active' if strategy_results.get('total_trades', 0) > 0 else 'No Trades'
            ]
            
            if update_existing:
                worksheet.update(f'A{row_num}:O{row_num}', [row_data])
            else:
                worksheet.append_row(row_data)
            
            time.sleep(self.rate_limit_delay)
            
            logger.info(f"P&L summary updated for {stock}")
            print(f"📊 P&L summary updated for {stock}")
            
        except Exception as e:
            logger.error(f"Error updating P&L summary: {e}")
            print(f"❌ Error updating P&L summary: {e}")
    
    def update_win_ratio(self, stock, strategy_results):
        """
        Update Win Ratio worksheet
        
        Parameters:
        - stock: Stock symbol
        - strategy_results: Dictionary containing strategy performance metrics
        """
        try:
            worksheet = self.sheet.worksheet('Win Ratio')
            
            # Check if stock already exists
            try:
                cell = worksheet.find(stock)
                row_num = cell.row
                update_existing = True
            except gspread.CellNotFound:
                row_num = len(worksheet.get_all_values()) + 1
                update_existing = False
            
            # Calculate performance grade
            win_ratio = strategy_results.get('win_ratio', 0)
            if win_ratio >= 80:
                grade = 'A+'
            elif win_ratio >= 70:
                grade = 'A'
            elif win_ratio >= 60:
                grade = 'B+'
            elif win_ratio >= 50:
                grade = 'B'
            else:
                grade = 'C'
            
            # Get best and worst trades
            trades_df = strategy_results.get('trades_df', pd.DataFrame())
            best_trade = trades_df['pnl'].max() if not trades_df.empty else 0
            worst_trade = trades_df['pnl'].min() if not trades_df.empty else 0
            
            # Prepare row data
            row_data = [
                stock,
                strategy_results.get('strategy_type', 'ML-Enhanced'),
                strategy_results.get('total_trades', 0),
                strategy_results.get('winning_trades', 0),
                f"{win_ratio:.1f}%",
                f"₹{best_trade:+.2f}",
                f"₹{worst_trade:+.2f}",
                f"{strategy_results.get('avg_holding_period', 0):.1f}",
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                grade
            ]
            
            if update_existing:
                worksheet.update(f'A{row_num}:J{row_num}', [row_data])
            else:
                worksheet.append_row(row_data)
            
            time.sleep(self.rate_limit_delay)
            
            logger.info(f"Win ratio updated for {stock}")
            print(f"🎯 Win ratio updated for {stock}")
            
        except Exception as e:
            logger.error(f"Error updating win ratio: {e}")
            print(f"❌ Error updating win ratio: {e}")
    
    def log_ml_analytics(self, stock, ml_data):
        """
        Log ML analytics data
        
        Parameters:
        - stock: Stock symbol
        - ml_data: Dictionary containing ML prediction data
        """
        try:
            worksheet = self.sheet.worksheet('ML Analytics')
            
            # Prepare row data
            row_data = [
                ml_data.get('date', datetime.now().strftime('%Y-%m-%d')),
                stock,
                ml_data.get('close_price', 0),
                ml_data.get('rsi', ''),
                ml_data.get('ml_prediction', ''),
                f"{ml_data.get('ml_confidence', 0):.1%}",
                ml_data.get('prediction_direction', ''),
                ml_data.get('actual_direction', ''),
                ml_data.get('prediction_correct', ''),
                ml_data.get('traditional_signal', ''),
                ml_data.get('enhanced_signal', ''),
                ml_data.get('notes', '')
            ]
            
            worksheet.append_row(row_data)
            time.sleep(self.rate_limit_delay)
            
            logger.info(f"ML analytics logged for {stock}")
            print(f"🤖 ML analytics logged for {stock}")
            
        except Exception as e:
            logger.error(f"Error logging ML analytics: {e}")
            print(f"❌ Error logging ML analytics: {e}")
    
    def update_portfolio_dashboard(self, portfolio_summary):
        """
        Update Portfolio Dashboard with overall portfolio metrics
        
        Parameters:
        - portfolio_summary: Dictionary containing portfolio-wide metrics
        """
        try:
            worksheet = self.sheet.worksheet('Portfolio Dashboard')
            
            # Update portfolio metrics
            updates = [
                ('B4', f"₹{portfolio_summary.get('total_investment', 0):,.0f}"),
                ('B5', f"₹{portfolio_summary.get('current_value', 0):,.0f}"),
                ('B6', f"₹{portfolio_summary.get('total_pnl', 0):+,.2f}"),
                ('B7', f"{portfolio_summary.get('portfolio_return', 0):+.2f}%"),
                
                # Strategy comparison
                ('E4', str(portfolio_summary.get('traditional_trades', 0))),
                ('F4', str(portfolio_summary.get('ml_enhanced_trades', 0))),
                ('G4', f"{portfolio_summary.get('trade_improvement', 0):+.0f}%"),
                
                ('E5', f"{portfolio_summary.get('traditional_win_ratio', 0):.1f}%"),
                ('F5', f"{portfolio_summary.get('ml_enhanced_win_ratio', 0):.1f}%"),
                ('G5', f"{portfolio_summary.get('win_ratio_improvement', 0):+.1f}%"),
                
                ('E6', f"₹{portfolio_summary.get('traditional_pnl', 0):+,.0f}"),
                ('F6', f"₹{portfolio_summary.get('ml_enhanced_pnl', 0):+,.0f}"),
                ('G6', f"{portfolio_summary.get('pnl_improvement', 0):+.0f}%"),
                
                ('E7', f"₹{portfolio_summary.get('traditional_avg_pnl', 0):+,.0f}"),
                ('F7', f"₹{portfolio_summary.get('ml_enhanced_avg_pnl', 0):+,.0f}"),
                ('G7', f"{portfolio_summary.get('avg_pnl_improvement', 0):+.0f}%"),
            ]
            
            # Batch update for efficiency
            for cell, value in updates:
                worksheet.update(cell, value)
                time.sleep(self.rate_limit_delay)
            
            logger.info("Portfolio dashboard updated")
            print("📈 Portfolio dashboard updated")
            
        except Exception as e:
            logger.error(f"Error updating portfolio dashboard: {e}")
            print(f"❌ Error updating portfolio dashboard: {e}")
    
    def batch_log_trades(self, trades_df, stock):
        """
        Batch log multiple trades for efficiency
        
        Parameters:
        - trades_df: DataFrame containing trade data
        - stock: Stock symbol
        """
        try:
            if trades_df.empty:
                print(f"⚠️ No trades to log for {stock}")
                return
            
            print(f"📝 Batch logging {len(trades_df)} trades for {stock}...")
            
            for _, trade in trades_df.iterrows():
                trade_data = {
                    'stock': stock,
                    'signal_type': 'ML-Enhanced',
                    'entry_date': trade.get('entry_date', ''),
                    'exit_date': trade.get('exit_date', ''),
                    'entry_price': trade.get('entry_price', 0),
                    'exit_price': trade.get('exit_price', 0),
                    'shares': trade.get('shares', 0),
                    'days_held': trade.get('days_held', 0),
                    'investment': trade.get('investment', 0),
                    'exit_value': trade.get('exit_value', 0),
                    'pnl': trade.get('pnl', 0),
                    'pnl_percent': trade.get('pnl_percent', 0),
                    'entry_rsi': trade.get('entry_rsi', ''),
                    'exit_rsi': trade.get('exit_rsi', ''),
                    'ml_prediction': trade.get('entry_ml_prediction', ''),
                    'ml_confidence': f"{trade.get('entry_ml_confidence', 0):.1%}" if pd.notna(trade.get('entry_ml_confidence')) else '',
                    'buy_reason': trade.get('buy_reason', ''),
                    'sell_reason': trade.get('sell_reason', ''),
                    'profitable': trade.get('profitable', False)
                }
                
                self.log_trade_signal(trade_data)
            
            print(f"✅ Successfully logged {len(trades_df)} trades for {stock}")
            
        except Exception as e:
            logger.error(f"Error batch logging trades: {e}")
            print(f"❌ Error batch logging trades: {e}")
    
    def automate_full_update(self, strategy_results, stock):
        """
        Automate complete update of all worksheets
        
        Parameters:
        - strategy_results: Complete strategy results dictionary
        - stock: Stock symbol
        """
        try:
            print(f"\n🔄 Automating full Google Sheets update for {stock}...")
            
            # 1. Log all trades
            if 'trades_df' in strategy_results and not strategy_results['trades_df'].empty:
                self.batch_log_trades(strategy_results['trades_df'], stock)
            
            # 2. Update P&L Summary
            self.update_pnl_summary(stock, strategy_results)
            
            # 3. Update Win Ratio
            self.update_win_ratio(stock, strategy_results)
            
            # 4. Log ML Analytics (if available)
            if 'ml_results' in strategy_results:
                ml_data = {
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'close_price': strategy_results.get('current_price', 0),
                    'ml_prediction': strategy_results.get('ml_results', {}).get('test_accuracy', 0),
                    'ml_confidence': strategy_results.get('ml_results', {}).get('test_accuracy', 0),
                    'notes': f"Model: {strategy_results.get('ml_results', {}).get('model_type', 'N/A')}"
                }
                self.log_ml_analytics(stock, ml_data)
            
            print(f"✅ Full Google Sheets update completed for {stock}")
            
        except Exception as e:
            logger.error(f"Error in automated update: {e}")
            print(f"❌ Error in automated update: {e}")
    
    def handle_api_errors(self, func, *args, **kwargs):
        """
        Handle API rate limits and connectivity issues
        
        Parameters:
        - func: Function to execute
        - args, kwargs: Function arguments
        """
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            
            except gspread.exceptions.APIError as e:
                if 'RATE_LIMIT_EXCEEDED' in str(e):
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Rate limit exceeded, waiting {wait_time} seconds...")
                    print(f"⏳ Rate limit exceeded, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"API Error: {e}")
                    print(f"❌ API Error: {e}")
                    break
            
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                print(f"❌ Unexpected error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    break
        
        logger.error(f"Failed to execute {func.__name__} after {max_retries} attempts")
        print(f"❌ Failed to execute {func.__name__} after {max_retries} attempts")
        return None

def main():
    """Test Google Sheets Automation"""
    print("🚀 GOOGLE SHEETS AUTOMATION TEST")
    print("=" * 60)
    
    # Initialize automation
    sheets_automation = GoogleSheetsAutomation()
    
    # Test authentication and setup
    if sheets_automation.setup_worksheets():
        print("✅ Google Sheets setup successful!")
        
        # Test with sample data
        sample_trade = {
            'stock': 'RELIANCE.NS',
            'signal_type': 'ML-Enhanced BUY',
            'entry_date': '2025-08-01',
            'exit_date': '2025-08-05',
            'entry_price': 1400.50,
            'exit_price': 1450.25,
            'shares': 7.14,
            'days_held': 4,
            'investment': 10000,
            'exit_value': 10355,
            'pnl': 355,
            'pnl_percent': 3.55,
            'entry_rsi': 32.5,
            'exit_rsi': 45.2,
            'ml_prediction': 1,
            'ml_confidence': 0.85,
            'buy_reason': 'ML-only BUY (85%)',
            'sell_reason': 'Holding Period Exceeded',
            'profitable': True
        }
        
        # Test trade logging
        sheets_automation.log_trade_signal(sample_trade)
        
        # Test P&L summary
        sample_results = {
            'strategy_type': 'ML-Enhanced',
            'total_trades': 5,
            'winning_trades': 4,
            'losing_trades': 1,
            'win_ratio': 80.0,
            'total_pnl': 1500,
            'avg_pnl_per_trade': 300,
            'strategy_return': 15.0,
            'market_return': 10.0,
            'sharpe_ratio': 1.25,
            'ml_accuracy': 0.65
        }
        
        sheets_automation.update_pnl_summary('RELIANCE.NS', sample_results)
        sheets_automation.update_win_ratio('RELIANCE.NS', sample_results)
        
        print("🎉 Google Sheets automation test completed!")
        
    else:
        print("❌ Google Sheets setup failed. Please check credentials and configuration.")

if __name__ == "__main__":
    main()