#!/usr/bin/env python3
"""
ML-Enhanced Trading Strategy
Integrates machine learning predictions with traditional RSI + MA crossover strategy
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
sys.path.append('src')

from ml_trading_system import MLTradingSystem
from practical_trading_strategy import PracticalTradingStrategy

class MLEnhancedTradingStrategy:
    def __init__(self, ml_model_type='random_forest', ml_confidence_threshold=0.6,
                 rsi_oversold=35, rsi_overbought=65, position_size=10000, holding_period=5):
        """
        ML-Enhanced Trading Strategy
        
        Parameters:
        - ml_model_type: Type of ML model to use
        - ml_confidence_threshold: Minimum confidence for ML predictions
        - Other parameters: Same as PracticalTradingStrategy
        """
        self.ml_system = MLTradingSystem(model_type=ml_model_type)
        self.traditional_strategy = PracticalTradingStrategy(
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought,
            position_size=position_size,
            holding_period=holding_period
        )
        
        self.ml_confidence_threshold = ml_confidence_threshold
        self.position_size = position_size
        self.holding_period = holding_period
        
        print(f"\n🤖 ML-Enhanced Strategy Configuration:")
        print(f"   ML Model: {ml_model_type.replace('_', ' ').title()}")
        print(f"   ML Confidence Threshold: {ml_confidence_threshold:.1%}")
        print(f"   Position Size: ₹{position_size:,}")
        print(f"   Max Holding Period: {holding_period} days")
        
    def generate_enhanced_signals(self, df):
        """
        Generate enhanced signals combining traditional strategy with ML predictions
        
        Enhanced Buy Signal:
        - Traditional: RSI < 35 AND 20-DMA > 50-DMA
        - ML Enhancement: ML predicts UP with confidence > threshold
        
        Enhanced Sell Signal:
        - Traditional: RSI > 65 OR bearish crossover
        - ML Enhancement: ML predicts DOWN with confidence > threshold
        """
        # Get traditional signals
        df = self.traditional_strategy.generate_signals(df)
        
        # Prepare ML features and get predictions for each day
        df_ml = self.ml_system.prepare_ml_features(df)
        
        # Initialize ML prediction columns
        df['ML_Prediction'] = np.nan
        df['ML_Confidence'] = np.nan
        df['ML_Direction'] = ''
        df['Enhanced_Signal'] = 0
        df['Enhanced_Buy_Signal'] = False
        df['Enhanced_Sell_Signal'] = False
        df['Enhanced_Signal_Reason'] = ''
        
        # Generate ML predictions for each valid data point
        if self.ml_system.is_trained:
            feature_data = df_ml[self.ml_system.feature_columns]
            
            # Handle infinite values and missing values
            feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
            feature_data = feature_data.fillna(feature_data.mean())
            
            # Get predictions for all valid points
            valid_indices = ~(feature_data.isnull().any(axis=1) | np.isinf(feature_data).any(axis=1))
            if valid_indices.sum() > 0:
                valid_data = feature_data[valid_indices]
                
                # Additional check for infinite values
                if np.isinf(valid_data.values).any():
                    valid_data = valid_data.replace([np.inf, -np.inf], np.nan).fillna(valid_data.mean())
                
                scaled_data = self.ml_system.scaler.transform(valid_data)
                
                predictions = self.ml_system.model.predict(scaled_data)
                probabilities = self.ml_system.model.predict_proba(scaled_data)
                
                # Store predictions
                df.loc[valid_indices, 'ML_Prediction'] = predictions
                df.loc[valid_indices, 'ML_Confidence'] = np.max(probabilities, axis=1)
                df.loc[valid_indices, 'ML_Direction'] = ['UP' if p == 1 else 'DOWN' for p in predictions]
        
        # Generate enhanced signals
        for i in range(len(df)):
            traditional_buy = df['Buy_Signal'].iloc[i]
            traditional_sell = df['Sell_Signal'].iloc[i]
            ml_prediction = df['ML_Prediction'].iloc[i]
            ml_confidence = df['ML_Confidence'].iloc[i]
            ml_direction = df['ML_Direction'].iloc[i]
            
            # Skip if ML data is not available
            if pd.isna(ml_prediction) or pd.isna(ml_confidence):
                continue
            
            # Enhanced Buy Signal
            if traditional_buy:
                if ml_prediction == 1 and ml_confidence >= self.ml_confidence_threshold:
                    df.loc[df.index[i], 'Enhanced_Signal'] = 1
                    df.loc[df.index[i], 'Enhanced_Buy_Signal'] = True
                    df.loc[df.index[i], 'Enhanced_Signal_Reason'] = f"Traditional BUY + ML {ml_direction} ({ml_confidence:.1%})"
                else:
                    df.loc[df.index[i], 'Enhanced_Signal_Reason'] = f"Traditional BUY rejected by ML {ml_direction} ({ml_confidence:.1%})"
            
            # Enhanced Sell Signal
            elif traditional_sell:
                if ml_prediction == 0 and ml_confidence >= self.ml_confidence_threshold:
                    df.loc[df.index[i], 'Enhanced_Signal'] = -1
                    df.loc[df.index[i], 'Enhanced_Sell_Signal'] = True
                    df.loc[df.index[i], 'Enhanced_Signal_Reason'] = f"Traditional SELL + ML {ml_direction} ({ml_confidence:.1%})"
                else:
                    df.loc[df.index[i], 'Enhanced_Signal_Reason'] = f"Traditional SELL rejected by ML {ml_direction} ({ml_confidence:.1%})"
            
            # ML-only signals (when ML is very confident but no traditional signal)
            elif ml_confidence >= 0.8:  # Very high confidence threshold
                if ml_prediction == 1:
                    df.loc[df.index[i], 'Enhanced_Signal'] = 1
                    df.loc[df.index[i], 'Enhanced_Buy_Signal'] = True
                    df.loc[df.index[i], 'Enhanced_Signal_Reason'] = f"ML-only BUY ({ml_confidence:.1%})"
                elif ml_prediction == 0:
                    df.loc[df.index[i], 'Enhanced_Signal'] = -1
                    df.loc[df.index[i], 'Enhanced_Sell_Signal'] = True
                    df.loc[df.index[i], 'Enhanced_Signal_Reason'] = f"ML-only SELL ({ml_confidence:.1%})"
        
        return df
    
    def backtest_enhanced_strategy(self, df, stock_symbol='STOCK'):
        """
        Backtest the ML-enhanced strategy
        """
        print(f"\n🔄 Backtesting ML-enhanced strategy for {stock_symbol}...")
        
        # First train the ML model
        print("🤖 Training ML model...")
        ml_results = self.ml_system.train_model(df, stock_symbol)
        self.ml_system.evaluate_model_performance(ml_results)
        
        # Generate enhanced signals
        df = self.generate_enhanced_signals(df)
        
        # Initialize tracking variables
        trades = []
        current_position = None
        portfolio_value = 0
        total_invested = 0
        
        # Track daily portfolio
        daily_portfolio = []
        
        for i, (date, row) in enumerate(df.iterrows()):
            daily_value = portfolio_value
            
            # Check for enhanced buy signal
            if row['Enhanced_Buy_Signal'] and current_position is None:
                shares = self.position_size / row['Close']
                
                current_position = {
                    'entry_date': date,
                    'entry_price': row['Close'],
                    'shares': shares,
                    'entry_rsi': row['RSI'],
                    'entry_ml_prediction': row['ML_Prediction'],
                    'entry_ml_confidence': row['ML_Confidence'],
                    'signal_reason': row['Enhanced_Signal_Reason'],
                    'days_held': 0
                }
                
                total_invested += self.position_size
                print(f"📈 ENHANCED BUY: {date.date()} at ₹{row['Close']:.2f} ({shares:.2f} shares)")
                print(f"    Reason: {row['Enhanced_Signal_Reason']}")
            
            # Check for enhanced sell signal or holding period exceeded
            elif current_position is not None:
                current_position['days_held'] += 1
                
                should_sell = (row['Enhanced_Sell_Signal'] or 
                             current_position['days_held'] >= self.holding_period)
                
                if should_sell:
                    # Calculate P&L
                    exit_value = current_position['shares'] * row['Close']
                    pnl = exit_value - self.position_size
                    pnl_percent = (pnl / self.position_size) * 100
                    
                    # Determine sell reason
                    if row['Enhanced_Sell_Signal']:
                        sell_reason = row['Enhanced_Signal_Reason']
                    else:
                        sell_reason = f'Holding Period Exceeded ({self.holding_period} days)'
                    
                    # Record trade
                    trade = {
                        'stock': stock_symbol,
                        'entry_date': current_position['entry_date'],
                        'exit_date': date,
                        'entry_price': current_position['entry_price'],
                        'exit_price': row['Close'],
                        'shares': current_position['shares'],
                        'days_held': current_position['days_held'],
                        'investment': self.position_size,
                        'exit_value': exit_value,
                        'pnl': pnl,
                        'pnl_percent': pnl_percent,
                        'entry_rsi': current_position['entry_rsi'],
                        'exit_rsi': row['RSI'],
                        'entry_ml_prediction': current_position['entry_ml_prediction'],
                        'entry_ml_confidence': current_position['entry_ml_confidence'],
                        'exit_ml_prediction': row['ML_Prediction'],
                        'exit_ml_confidence': row['ML_Confidence'],
                        'buy_reason': current_position['signal_reason'],
                        'sell_reason': sell_reason,
                        'profitable': pnl > 0
                    }
                    
                    trades.append(trade)
                    portfolio_value += exit_value
                    
                    status = "✅ PROFIT" if pnl > 0 else "❌ LOSS"
                    print(f"📉 ENHANCED SELL: {date.date()} at ₹{row['Close']:.2f}")
                    print(f"    Reason: {sell_reason}")
                    print(f"    P&L: ₹{pnl:+.2f} ({pnl_percent:+.2f}%) {status}")
                    
                    current_position = None
                
                else:
                    daily_value = current_position['shares'] * row['Close']
            
            # Record daily portfolio value
            daily_portfolio.append({
                'date': date,
                'portfolio_value': daily_value,
                'close_price': row['Close'],
                'rsi': row['RSI'],
                'ml_prediction': row['ML_Prediction'],
                'ml_confidence': row['ML_Confidence'],
                'has_position': current_position is not None
            })
        
        # Handle remaining open position
        if current_position is not None:
            final_price = df['Close'].iloc[-1]
            final_date = df.index[-1]
            exit_value = current_position['shares'] * final_price
            pnl = exit_value - self.position_size
            pnl_percent = (pnl / self.position_size) * 100
            
            trade = {
                'stock': stock_symbol,
                'entry_date': current_position['entry_date'],
                'exit_date': final_date,
                'entry_price': current_position['entry_price'],
                'exit_price': final_price,
                'shares': current_position['shares'],
                'days_held': current_position['days_held'],
                'investment': self.position_size,
                'exit_value': exit_value,
                'pnl': pnl,
                'pnl_percent': pnl_percent,
                'entry_rsi': current_position['entry_rsi'],
                'exit_rsi': df['RSI'].iloc[-1],
                'entry_ml_prediction': current_position['entry_ml_prediction'],
                'entry_ml_confidence': current_position['entry_ml_confidence'],
                'exit_ml_prediction': df['ML_Prediction'].iloc[-1],
                'exit_ml_confidence': df['ML_Confidence'].iloc[-1],
                'buy_reason': current_position['signal_reason'],
                'sell_reason': 'End of Period',
                'profitable': pnl > 0
            }
            
            trades.append(trade)
            portfolio_value += exit_value
            
            status = "✅ PROFIT" if pnl > 0 else "❌ LOSS"
            print(f"📉 FINAL ENHANCED SELL: {final_date.date()} at ₹{final_price:.2f}")
            print(f"    P&L: ₹{pnl:+.2f} ({pnl_percent:+.2f}%) {status}")
        
        # Calculate performance metrics
        trades_df = pd.DataFrame(trades)
        daily_portfolio_df = pd.DataFrame(daily_portfolio)
        
        if len(trades) > 0:
            total_pnl = trades_df['pnl'].sum()
            total_trades = len(trades)
            winning_trades = len(trades_df[trades_df['profitable']])
            win_ratio = (winning_trades / total_trades) * 100
            avg_pnl = trades_df['pnl'].mean()
            avg_holding_period = trades_df['days_held'].mean()
            
            returns = trades_df['pnl_percent'].values / 100
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0
            
        else:
            total_pnl = 0
            total_trades = 0
            winning_trades = 0
            win_ratio = 0
            avg_pnl = 0
            avg_holding_period = 0
            sharpe_ratio = 0
        
        # Market performance
        market_return = ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100
        
        results = {
            'stock': stock_symbol,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_ratio': win_ratio,
            'total_pnl': total_pnl,
            'avg_pnl_per_trade': avg_pnl,
            'avg_holding_period': avg_holding_period,
            'total_invested': total_invested,
            'final_portfolio_value': portfolio_value,
            'strategy_return': (portfolio_value / total_invested - 1) * 100 if total_invested > 0 else 0,
            'market_return': market_return,
            'sharpe_ratio': sharpe_ratio,
            'ml_accuracy': ml_results['test_accuracy'],
            'trades_df': trades_df,
            'daily_portfolio_df': daily_portfolio_df,
            'signals_df': df,
            'ml_results': ml_results
        }
        
        return results
    
    def print_enhanced_summary(self, results):
        """Print comprehensive ML-enhanced strategy summary"""
        print(f"\n{'='*70}")
        print(f"ML-ENHANCED STRATEGY SUMMARY: {results['stock']}")
        print(f"{'='*70}")
        
        print(f"🤖 ML Model Performance:")
        print(f"   Model Accuracy: {results['ml_accuracy']:.1%}")
        print(f"   Confidence Threshold: {self.ml_confidence_threshold:.1%}")
        
        print(f"\n📊 Trading Performance:")
        print(f"   Total Trades: {results['total_trades']}")
        print(f"   Winning Trades: {results['winning_trades']}")
        print(f"   Losing Trades: {results['losing_trades']}")
        print(f"   Win Ratio: {results['win_ratio']:.1f}%")
        
        print(f"\n💰 Financial Performance:")
        print(f"   Total P&L: ₹{results['total_pnl']:+,.2f}")
        print(f"   Average P&L per Trade: ₹{results['avg_pnl_per_trade']:+,.2f}")
        print(f"   Strategy Return: {results['strategy_return']:+.2f}%")
        print(f"   Market Return: {results['market_return']:+.2f}%")
        print(f"   Alpha (vs Market): {results['strategy_return'] - results['market_return']:+.2f}%")
        
        print(f"\n📈 Risk Metrics:")
        print(f"   Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print(f"   Average Holding Period: {results['avg_holding_period']:.1f} days")
        
        if results['total_trades'] > 0:
            trades_df = results['trades_df']
            print(f"\n🔍 Trade Details:")
            print(f"   Best Trade: ₹{trades_df['pnl'].max():+.2f} ({trades_df['pnl_percent'].max():+.2f}%)")
            print(f"   Worst Trade: ₹{trades_df['pnl'].min():+.2f} ({trades_df['pnl_percent'].min():+.2f}%)")
            print(f"   Average ML Entry Confidence: {trades_df['entry_ml_confidence'].mean():.1%}")

def main():
    """Test the ML-Enhanced Trading Strategy"""
    from data_ingestion import EnhancedDataIngestion
    
    print("🚀 ML-ENHANCED TRADING STRATEGY TEST")
    print("=" * 70)
    
    # Initialize components
    ingestion = EnhancedDataIngestion()
    
    # Test with RELIANCE data
    test_stock = 'RELIANCE.NS'
    print(f"Testing ML-enhanced strategy with {test_stock}...")
    
    # Load processed data
    data = ingestion.load_saved_data(test_stock, 'processed')
    
    if data is None:
        print("❌ No processed data found. Please run data ingestion first.")
        return
    
    # Initialize ML-enhanced strategy
    ml_strategy = MLEnhancedTradingStrategy(
        ml_model_type='random_forest',
        ml_confidence_threshold=0.6,
        rsi_oversold=35,
        rsi_overbought=65,
        position_size=10000,
        holding_period=5
    )
    
    # Run backtest
    results = ml_strategy.backtest_enhanced_strategy(data, test_stock)
    
    # Print summary
    ml_strategy.print_enhanced_summary(results)
    
    # Save results
    if results['total_trades'] > 0:
        results['trades_df'].to_csv(f'data/{test_stock.replace(".", "_")}_ml_enhanced_trades.csv', index=False)
        print(f"\n💾 ML-enhanced trade details saved to data/{test_stock.replace('.', '_')}_ml_enhanced_trades.csv")
    
    print(f"\n🎉 ML-Enhanced Trading Strategy test completed!")

if __name__ == "__main__":
    main()