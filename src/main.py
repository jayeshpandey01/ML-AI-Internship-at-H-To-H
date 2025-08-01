#!/usr/bin/env python3
"""
Algo-Trading System Main Script
Fetches stock data, implements RSI + MA crossover strategy, uses ML predictions,
and logs results to Google Sheets.
"""

import os
import sys
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api_call import DataFetcher
from trading_strategy import TradingStrategy
from ml_predictor import MLPredictor
from sheets_integration import SheetsLogger

# Load environment variables
load_dotenv('config/config.env')

class AlgoTradingSystem:
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.strategy = TradingStrategy()
        self.ml_predictor = MLPredictor()
        self.sheets_logger = SheetsLogger()
        
        # Get stock symbols from config
        self.stocks = os.getenv('NIFTY_STOCKS', 'RELIANCE.NS,HDFCBANK.NS,INFY.NS').split(',')
        
    def run_analysis(self):
        """Run complete analysis for all stocks"""
        print("=" * 60)
        print("ALGO-TRADING SYSTEM ANALYSIS")
        print("=" * 60)
        print(f"Analyzing stocks: {', '.join(self.stocks)}")
        print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Set up Google Sheets
        if not self.sheets_logger.setup_sheets():
            print("Warning: Google Sheets setup failed. Continuing without logging.")
        
        results_summary = {}
        
        for stock in self.stocks:
            print(f"\n{'='*40}")
            print(f"ANALYZING {stock}")
            print(f"{'='*40}")
            
            try:
                # Fetch stock data
                print(f"1. Fetching data for {stock}...")
                data = self.data_fetcher.fetch_yfinance_data(stock, period='6mo', interval='1d')
                
                if data is None or data.empty:
                    print(f"❌ No data available for {stock}")
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
                else:
                    print("❌ ML model training failed")
                    prediction = None
                
                # Log to Google Sheets
                print(f"4. Logging results to Google Sheets...")
                try:
                    # Update P&L summary
                    self.sheets_logger.update_pnl_summary(stock, backtest_results)
                    
                    # Log recent trades
                    trades = backtest_results['trades']
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
                    print(f"❌ Error logging to Google Sheets: {e}")
                
                # Store results
                results_summary[stock] = {
                    'backtest_results': backtest_results,
                    'ml_results': ml_results,
                    'prediction': prediction
                }
                
            except Exception as e:
                print(f"❌ Error analyzing {stock}: {e}")
                continue
        
        # Print summary
        self._print_summary(results_summary)
        
        return results_summary
    
    def _print_summary(self, results):
        """Print analysis summary"""
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
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
        
        print("\n✅ Analysis complete! Check Google Sheets for detailed logs.")

def main():
    """Main function"""
    system = AlgoTradingSystem()
    results = system.run_analysis()
    return results

if __name__ == "__main__":
    main()