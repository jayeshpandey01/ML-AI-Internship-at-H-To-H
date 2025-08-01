import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv('config/config.env')

class SheetsLogger:
    def __init__(self):
        self.credentials_path = os.getenv('GOOGLE_SHEETS_CREDENTIALS_PATH')
        self.sheet_id = os.getenv('GOOGLE_SHEET_ID')
        self.gc = None
        self.sheet = None
        
    def authenticate(self):
        """Authenticate with Google Sheets API"""
        try:
            scope = [
                'https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive'
            ]
            
            if not os.path.exists(self.credentials_path):
                print(f"Credentials file not found: {self.credentials_path}")
                print("Please download your Google Service Account credentials and place them in the config folder")
                return False
            
            creds = Credentials.from_service_account_file(self.credentials_path, scopes=scope)
            self.gc = gspread.authorize(creds)
            self.sheet = self.gc.open_by_key(self.sheet_id)
            
            print("Successfully authenticated with Google Sheets")
            return True
            
        except Exception as e:
            print(f"Error authenticating with Google Sheets: {e}")
            return False
    
    def setup_sheets(self):
        """Set up the required worksheets"""
        if not self.authenticate():
            return False
        
        try:
            # Create or get Trade Log worksheet
            try:
                trade_log = self.sheet.worksheet('Trade Log')
            except gspread.WorksheetNotFound:
                trade_log = self.sheet.add_worksheet(title='Trade Log', rows=1000, cols=10)
                
            # Set up Trade Log headers
            headers = ['Timestamp', 'Stock', 'Action', 'Price', 'Quantity', 'RSI', 'MA_Short', 'MA_Long', 'ML_Prediction', 'Confidence']
            trade_log.update('A1:J1', [headers])
            
            # Create or get P&L Summary worksheet
            try:
                pnl_summary = self.sheet.worksheet('P&L Summary')
            except gspread.WorksheetNotFound:
                pnl_summary = self.sheet.add_worksheet(title='P&L Summary', rows=100, cols=8)
                
            # Set up P&L Summary headers
            pnl_headers = ['Stock', 'Total_Trades', 'Winning_Trades', 'Win_Rate', 'Total_Return', 'Sharpe_Ratio', 'Final_Value', 'Last_Updated']
            pnl_summary.update('A1:H1', [pnl_headers])
            
            # Create or get Analytics worksheet
            try:
                analytics = self.sheet.worksheet('Analytics')
            except gspread.WorksheetNotFound:
                analytics = self.sheet.add_worksheet(title='Analytics', rows=100, cols=6)
                
            # Set up Analytics headers
            analytics_headers = ['Date', 'Stock', 'Close_Price', 'RSI', 'Signal', 'ML_Prediction']
            analytics.update('A1:F1', [analytics_headers])
            
            print("Worksheets set up successfully")
            return True
            
        except Exception as e:
            print(f"Error setting up worksheets: {e}")
            return False
    
    def log_trade(self, stock, action, price, quantity=1, rsi=None, ma_short=None, ma_long=None, ml_prediction=None, confidence=None):
        """Log a trade to the Trade Log worksheet"""
        try:
            trade_log = self.sheet.worksheet('Trade Log')
            
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            row_data = [
                timestamp, stock, action, price, quantity,
                rsi or '', ma_short or '', ma_long or '',
                ml_prediction or '', confidence or ''
            ]
            
            trade_log.append_row(row_data)
            print(f"Trade logged: {action} {stock} at {price}")
            
        except Exception as e:
            print(f"Error logging trade: {e}")
    
    def update_pnl_summary(self, stock, backtest_results):
        """Update P&L summary for a stock"""
        try:
            pnl_summary = self.sheet.worksheet('P&L Summary')
            
            # Find if stock already exists
            stock_cells = pnl_summary.findall(stock)
            
            row_data = [
                stock,
                backtest_results['total_trades'],
                int(backtest_results['win_rate'] * backtest_results['total_trades'] / 100),
                f"{backtest_results['win_rate']:.2f}%",
                f"{backtest_results['total_return']:.4f}",
                f"{backtest_results['sharpe_ratio']:.4f}",
                f"{backtest_results['final_portfolio_value']:.2f}",
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ]
            
            if stock_cells:
                # Update existing row
                row_num = stock_cells[0].row
                pnl_summary.update(f'A{row_num}:H{row_num}', [row_data])
            else:
                # Add new row
                pnl_summary.append_row(row_data)
            
            print(f"P&L summary updated for {stock}")
            
        except Exception as e:
            print(f"Error updating P&L summary: {e}")
    
    def log_analytics_data(self, stock, df):
        """Log analytics data for a stock"""
        try:
            analytics = self.sheet.worksheet('Analytics')
            
            # Clear existing data for this stock
            stock_cells = analytics.findall(stock)
            if stock_cells:
                # Clear rows with this stock (simplified approach)
                pass
            
            # Add latest data points (last 30 days)
            recent_data = df.tail(30)
            
            for idx, row in recent_data.iterrows():
                row_data = [
                    idx.strftime('%Y-%m-%d'),
                    stock,
                    row['Close'],
                    row.get('RSI', ''),
                    row.get('Signal', ''),
                    row.get('ML_Prediction', '')
                ]
                analytics.append_row(row_data)
            
            print(f"Analytics data logged for {stock}")
            
        except Exception as e:
            print(f"Error logging analytics data: {e}")
    
    def get_trade_history(self, stock=None):
        """Get trade history from Google Sheets"""
        try:
            trade_log = self.sheet.worksheet('Trade Log')
            records = trade_log.get_all_records()
            
            df = pd.DataFrame(records)
            if stock:
                df = df[df['Stock'] == stock]
            
            return df
            
        except Exception as e:
            print(f"Error getting trade history: {e}")
            return pd.DataFrame()