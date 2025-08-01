#!/usr/bin/env python3
"""
Comprehensive Backtesting System
Tests the enhanced trading strategy on all NIFTY 50 stocks
"""

import sys
import os
sys.path.append('src')

from src.data_ingestion import EnhancedDataIngestion
from src.enhanced_trading_strategy import EnhancedTradingStrategy
import pandas as pd
from datetime import datetime

def run_comprehensive_backtest():
    """Run comprehensive backtest on all configured stocks"""
    print("🚀 COMPREHENSIVE BACKTESTING SYSTEM")
    print("=" * 80)
    
    # Initialize components
    ingestion = EnhancedDataIngestion()
    strategy = EnhancedTradingStrategy(
        position_size=10000,  # ₹10,000 per trade
        holding_period=5      # Maximum 5 days holding
    )
    
    # Get stocks from config
    stocks = os.getenv('NIFTY_STOCKS', 'RELIANCE.NS,HDFCBANK.NS,INFY.NS').split(',')
    
    print(f"📊 Testing strategy on {len(stocks)} stocks:")
    for i, stock in enumerate(stocks, 1):
        print(f"   {i}. {stock}")
    
    print(f"\n⚙️ Strategy Configuration:")
    print(f"   Position Size: ₹{strategy.position_size:,} per trade")
    print(f"   Max Holding Period: {strategy.holding_period} days")
    print(f"   RSI Oversold: < {strategy.rsi_oversold}")
    print(f"   RSI Overbought: > {strategy.rsi_overbought}")
    print(f"   Moving Averages: {strategy.ma_short}-DMA vs {strategy.ma_long}-DMA")
    
    # Fetch data for all stocks
    print(f"\n{'='*60}")
    print("FETCHING DATA")
    print(f"{'='*60}")
    
    stock_data, summary = ingestion.run_complete_ingestion(
        stocks=stocks,
        period='6mo',
        calculate_indicators=True
    )
    
    if not stock_data:
        print("❌ No data available for backtesting")
        return
    
    # Run backtests
    print(f"\n{'='*60}")
    print("RUNNING BACKTESTS")
    print(f"{'='*60}")
    
    all_results = {}
    all_trades = []
    
    for stock in stocks:
        if stock in stock_data:
            print(f"\n{'='*50}")
            print(f"BACKTESTING: {stock}")
            print(f"{'='*50}")
            
            data = stock_data[stock]
            results = strategy.backtest_strategy(data, stock)
            
            # Store results
            all_results[stock] = results
            
            # Collect all trades
            if results['total_trades'] > 0:
                trades_df = results['trades_df'].copy()
                all_trades.append(trades_df)
            
            # Print individual summary
            strategy.print_backtest_summary(results)
        else:
            print(f"❌ No data available for {stock}")
    
    # Combine all trades
    if all_trades:
        combined_trades = pd.concat(all_trades, ignore_index=True)
        combined_trades.to_csv('data/all_trades_combined.csv', index=False)
        print(f"\n💾 All trades saved to data/all_trades_combined.csv")
    
    # Generate comparative analysis
    generate_comparative_analysis(all_results)
    
    # Generate portfolio summary
    generate_portfolio_summary(all_results, combined_trades if all_trades else pd.DataFrame())
    
    return all_results

def generate_comparative_analysis(all_results):
    """Generate comparative analysis across all stocks"""
    print(f"\n{'='*80}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'='*80}")
    
    if not all_results:
        print("❌ No results to analyze")
        return
    
    # Create comparison DataFrame
    comparison_data = []
    
    for stock, results in all_results.items():
        comparison_data.append({
            'Stock': stock,
            'Total_Trades': results['total_trades'],
            'Win_Ratio': f"{results['win_ratio']:.1f}%",
            'Total_PnL': f"₹{results['total_pnl']:+,.0f}",
            'Avg_PnL_Trade': f"₹{results['avg_pnl_per_trade']:+,.0f}",
            'Strategy_Return': f"{results['strategy_return']:+.2f}%",
            'Market_Return': f"{results['market_return']:+.2f}%",
            'Alpha': f"{results['strategy_return'] - results['market_return']:+.2f}%",
            'Max_Drawdown': f"{results['max_drawdown']:.2f}%",
            'Sharpe_Ratio': f"{results['sharpe_ratio']:.3f}",
            'Avg_Hold_Days': f"{results['avg_holding_period']:.1f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print("📊 Performance Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Save comparison
    comparison_df.to_csv('data/strategy_comparison.csv', index=False)
    print(f"\n💾 Comparison saved to data/strategy_comparison.csv")
    
    # Identify best performers
    print(f"\n🏆 PERFORMANCE HIGHLIGHTS:")
    
    # Best overall return
    best_return = max(all_results.items(), key=lambda x: x[1]['strategy_return'])
    print(f"   📈 Best Strategy Return: {best_return[0]} ({best_return[1]['strategy_return']:+.2f}%)")
    
    # Best win ratio
    best_winrate = max(all_results.items(), key=lambda x: x[1]['win_ratio'])
    print(f"   🎯 Best Win Ratio: {best_winrate[0]} ({best_winrate[1]['win_ratio']:.1f}%)")
    
    # Most active trading
    most_trades = max(all_results.items(), key=lambda x: x[1]['total_trades'])
    print(f"   🔄 Most Active: {most_trades[0]} ({most_trades[1]['total_trades']} trades)")
    
    # Best alpha (vs market)
    best_alpha = max(all_results.items(), 
                    key=lambda x: x[1]['strategy_return'] - x[1]['market_return'])
    alpha_value = best_alpha[1]['strategy_return'] - best_alpha[1]['market_return']
    print(f"   ⚡ Best Alpha: {best_alpha[0]} ({alpha_value:+.2f}% vs market)")

def generate_portfolio_summary(all_results, combined_trades):
    """Generate overall portfolio summary"""
    print(f"\n{'='*80}")
    print("PORTFOLIO SUMMARY")
    print(f"{'='*80}")
    
    if not all_results:
        print("❌ No results to summarize")
        return
    
    # Calculate portfolio metrics
    total_trades = sum(r['total_trades'] for r in all_results.values())
    total_pnl = sum(r['total_pnl'] for r in all_results.values())
    total_invested = sum(r['total_invested'] for r in all_results.values())
    total_winning = sum(r['winning_trades'] for r in all_results.values())
    
    portfolio_return = (total_pnl / total_invested * 100) if total_invested > 0 else 0
    overall_win_ratio = (total_winning / total_trades * 100) if total_trades > 0 else 0
    
    # Market comparison
    market_returns = [r['market_return'] for r in all_results.values()]
    avg_market_return = sum(market_returns) / len(market_returns) if market_returns else 0
    
    print(f"📊 Overall Portfolio Performance:")
    print(f"   Total Trades Executed: {total_trades}")
    print(f"   Total Amount Invested: ₹{total_invested:,.0f}")
    print(f"   Total P&L: ₹{total_pnl:+,.2f}")
    print(f"   Portfolio Return: {portfolio_return:+.2f}%")
    print(f"   Overall Win Ratio: {overall_win_ratio:.1f}%")
    print(f"   Average Market Return: {avg_market_return:+.2f}%")
    print(f"   Portfolio Alpha: {portfolio_return - avg_market_return:+.2f}%")
    
    if not combined_trades.empty:
        print(f"\n📈 Trade Statistics:")
        print(f"   Best Single Trade: ₹{combined_trades['pnl'].max():+.2f}")
        print(f"   Worst Single Trade: ₹{combined_trades['pnl'].min():+.2f}")
        print(f"   Average Trade P&L: ₹{combined_trades['pnl'].mean():+.2f}")
        print(f"   Average Holding Period: {combined_trades['days_held'].mean():.1f} days")
        
        # Trade distribution by stock
        print(f"\n📊 Trades by Stock:")
        trade_counts = combined_trades['stock'].value_counts()
        for stock, count in trade_counts.items():
            stock_pnl = combined_trades[combined_trades['stock'] == stock]['pnl'].sum()
            print(f"   {stock}: {count} trades, ₹{stock_pnl:+.2f} P&L")
    
    # Strategy effectiveness analysis
    print(f"\n🎯 Strategy Effectiveness:")
    
    effective_stocks = [stock for stock, results in all_results.items() 
                       if results['strategy_return'] > results['market_return']]
    
    print(f"   Stocks Outperforming Market: {len(effective_stocks)}/{len(all_results)}")
    if effective_stocks:
        print(f"   Outperforming Stocks: {', '.join(effective_stocks)}")
    
    # Risk assessment
    drawdowns = [r['max_drawdown'] for r in all_results.values() if r['max_drawdown'] != 0]
    if drawdowns:
        avg_drawdown = sum(drawdowns) / len(drawdowns)
        max_drawdown = min(drawdowns)  # Most negative
        print(f"   Average Max Drawdown: {avg_drawdown:.2f}%")
        print(f"   Worst Drawdown: {max_drawdown:.2f}%")
    
    # Save portfolio summary
    portfolio_summary = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_trades': total_trades,
        'total_invested': total_invested,
        'total_pnl': total_pnl,
        'portfolio_return': portfolio_return,
        'win_ratio': overall_win_ratio,
        'avg_market_return': avg_market_return,
        'portfolio_alpha': portfolio_return - avg_market_return,
        'stocks_analyzed': len(all_results),
        'effective_stocks': len(effective_stocks)
    }
    
    portfolio_df = pd.DataFrame([portfolio_summary])
    portfolio_df.to_csv('data/portfolio_summary.csv', index=False)
    print(f"\n💾 Portfolio summary saved to data/portfolio_summary.csv")

def main():
    """Main backtesting function"""
    try:
        results = run_comprehensive_backtest()
        
        print(f"\n🎉 BACKTESTING COMPLETED SUCCESSFULLY!")
        print(f"\n📁 Generated Files:")
        print(f"   📊 data/strategy_comparison.csv - Strategy performance comparison")
        print(f"   📈 data/all_trades_combined.csv - All trade details")
        print(f"   📋 data/portfolio_summary.csv - Overall portfolio summary")
        print(f"   💾 data/*_trades.csv - Individual stock trade details")
        
        return results
        
    except Exception as e:
        print(f"❌ Backtesting failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()