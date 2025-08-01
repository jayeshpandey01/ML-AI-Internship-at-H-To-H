#!/usr/bin/env python3
"""
System Monitor and Status Dashboard
Monitors the automated algo trading system performance and health
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class SystemMonitor:
    def __init__(self):
        """
        Initialize System Monitor
        """
        self.logs_dir = 'logs'
        self.data_dir = 'data'
        
        # Ensure directories exist
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        print("📊 System Monitor Initialized")
        print(f"   Logs Directory: {self.logs_dir}")
        print(f"   Data Directory: {self.data_dir}")
    
    def get_system_status(self):
        """
        Get current system status and health metrics
        """
        print("\n🔍 SYSTEM STATUS REPORT")
        print("=" * 60)
        
        try:
            # Check for recent log files
            log_files = glob.glob(f"{self.logs_dir}/algo_system_*.log")
            error_files = glob.glob(f"{self.logs_dir}/algo_errors_*.log")
            summary_files = glob.glob(f"{self.logs_dir}/run_summary_*.json")
            
            print(f"📁 File Status:")
            print(f"   System Logs: {len(log_files)} files")
            print(f"   Error Logs: {len(error_files)} files")
            print(f"   Run Summaries: {len(summary_files)} files")
            
            # Analyze recent runs
            if summary_files:
                self._analyze_recent_runs(summary_files)
            else:
                print("⚠️ No run summaries found - system may not have run yet")
            
            # Check data files
            self._check_data_files()
            
            # Check error patterns
            if error_files:
                self._analyze_error_patterns(error_files)
            
        except Exception as e:
            print(f"❌ Error getting system status: {e}")
    
    def _analyze_recent_runs(self, summary_files):
        """
        Analyze recent system runs
        """
        print(f"\n📊 Recent Runs Analysis:")
        
        try:
            # Load recent summaries
            summaries = []
            for file in sorted(summary_files)[-10:]:  # Last 10 runs
                try:
                    with open(file, 'r') as f:
                        summary = json.load(f)
                        summaries.append(summary)
                except Exception as e:
                    print(f"⚠️ Could not read {file}: {e}")
            
            if not summaries:
                print("❌ No valid run summaries found")
                return
            
            # Calculate metrics
            total_runs = len(summaries)
            total_trades = sum(s.get('total_trades', 0) for s in summaries)
            total_pnl = sum(s.get('total_pnl', 0) for s in summaries)
            avg_duration = sum(s.get('duration_seconds', 0) for s in summaries) / total_runs
            
            print(f"   Recent Runs: {total_runs}")
            print(f"   Total Trades: {total_trades}")
            print(f"   Total P&L: ₹{total_pnl:+,.2f}")
            print(f"   Average Duration: {avg_duration:.2f} seconds")
            
            # Latest run details
            if summaries:
                latest = summaries[-1]
                print(f"\n📋 Latest Run ({latest.get('run_id', 'Unknown')}):")
                print(f"   Time: {latest.get('start_time', 'Unknown')}")
                print(f"   Stocks Processed: {latest.get('stocks_processed', 0)}")
                print(f"   Trades Generated: {latest.get('total_trades', 0)}")
                print(f"   P&L: ₹{latest.get('total_pnl', 0):+.2f}")
                print(f"   Win Ratio: {latest.get('overall_win_ratio', 0):.1f}%")
                
                # System stats
                if 'system_stats' in latest:
                    stats = latest['system_stats']
                    print(f"   Success Rate: {stats.get('success_rate', 0):.1f}%")
                    print(f"   Total System Runs: {stats.get('total_runs', 0)}")
            
        except Exception as e:
            print(f"❌ Error analyzing runs: {e}")
    
    def _check_data_files(self):
        """
        Check data file status
        """
        print(f"\n💾 Data Files Status:")
        
        try:
            # Check for processed data
            processed_files = glob.glob(f"{self.data_dir}/*_processed_data.csv")
            trade_files = glob.glob(f"{self.data_dir}/*_trades.csv")
            model_files = glob.glob(f"{self.data_dir}/*.pkl")
            
            print(f"   Processed Data Files: {len(processed_files)}")
            print(f"   Trade Log Files: {len(trade_files)}")
            print(f"   ML Model Files: {len(model_files)}")
            
            # Check file ages
            if processed_files:
                latest_data = max(processed_files, key=os.path.getmtime)
                age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(latest_data))
                print(f"   Latest Data Age: {age.days} days, {age.seconds//3600} hours")
            
        except Exception as e:
            print(f"❌ Error checking data files: {e}")
    
    def _analyze_error_patterns(self, error_files):
        """
        Analyze error patterns from log files
        """
        print(f"\n🚨 Error Analysis:")
        
        try:
            error_count = 0
            error_types = {}
            
            for error_file in error_files[-3:]:  # Last 3 error files
                try:
                    with open(error_file, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            if 'ERROR' in line:
                                error_count += 1
                                # Extract error type (simplified)
                                if 'API' in line:
                                    error_types['API_Error'] = error_types.get('API_Error', 0) + 1
                                elif 'Network' in line or 'Connection' in line:
                                    error_types['Network_Error'] = error_types.get('Network_Error', 0) + 1
                                elif 'Google Sheets' in line:
                                    error_types['Sheets_Error'] = error_types.get('Sheets_Error', 0) + 1
                                else:
                                    error_types['Other_Error'] = error_types.get('Other_Error', 0) + 1
                except Exception:
                    continue
            
            print(f"   Total Errors (recent): {error_count}")
            if error_types:
                print(f"   Error Breakdown:")
                for error_type, count in error_types.items():
                    print(f"     {error_type}: {count}")
            
        except Exception as e:
            print(f"❌ Error analyzing error patterns: {e}")
    
    def generate_performance_report(self):
        """
        Generate comprehensive performance report
        """
        print("\n📈 PERFORMANCE REPORT")
        print("=" * 60)
        
        try:
            # Load all trade data
            trade_files = glob.glob(f"{self.data_dir}/*_trades.csv")
            
            if not trade_files:
                print("❌ No trade files found for performance analysis")
                return
            
            all_trades = []
            for file in trade_files:
                try:
                    df = pd.read_csv(file)
                    all_trades.append(df)
                except Exception as e:
                    print(f"⚠️ Could not read {file}: {e}")
            
            if not all_trades:
                print("❌ No valid trade data found")
                return
            
            # Combine all trades
            combined_trades = pd.concat(all_trades, ignore_index=True)
            
            # Performance metrics
            total_trades = len(combined_trades)
            profitable_trades = len(combined_trades[combined_trades['profitable'] == True])
            win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
            
            total_pnl = combined_trades['pnl'].sum()
            avg_pnl = combined_trades['pnl'].mean()
            best_trade = combined_trades['pnl'].max()
            worst_trade = combined_trades['pnl'].min()
            
            print(f"📊 Overall Performance:")
            print(f"   Total Trades: {total_trades}")
            print(f"   Profitable Trades: {profitable_trades}")
            print(f"   Win Rate: {win_rate:.1f}%")
            print(f"   Total P&L: ₹{total_pnl:+,.2f}")
            print(f"   Average P&L: ₹{avg_pnl:+,.2f}")
            print(f"   Best Trade: ₹{best_trade:+,.2f}")
            print(f"   Worst Trade: ₹{worst_trade:+,.2f}")
            
            # Performance by stock
            if 'stock' in combined_trades.columns:
                print(f"\n📈 Performance by Stock:")
                stock_performance = combined_trades.groupby('stock').agg({
                    'pnl': ['count', 'sum', 'mean'],
                    'profitable': 'sum'
                }).round(2)
                
                for stock in combined_trades['stock'].unique():
                    stock_data = combined_trades[combined_trades['stock'] == stock]
                    stock_trades = len(stock_data)
                    stock_wins = len(stock_data[stock_data['profitable'] == True])
                    stock_win_rate = (stock_wins / stock_trades * 100) if stock_trades > 0 else 0
                    stock_pnl = stock_data['pnl'].sum()
                    
                    print(f"   {stock}: {stock_trades} trades, {stock_win_rate:.1f}% win rate, ₹{stock_pnl:+.2f}")
            
            # Save performance report
            report_data = {
                'report_date': datetime.now().isoformat(),
                'total_trades': int(total_trades),
                'win_rate': float(win_rate),
                'total_pnl': float(total_pnl),
                'avg_pnl': float(avg_pnl),
                'best_trade': float(best_trade),
                'worst_trade': float(worst_trade)
            }
            
            with open(f"{self.logs_dir}/performance_report_{datetime.now().strftime('%Y%m%d')}.json", 'w') as f:
                json.dump(report_data, f, indent=2)
            
            print(f"\n💾 Performance report saved to {self.logs_dir}/")
            
        except Exception as e:
            print(f"❌ Error generating performance report: {e}")
    
    def check_system_health(self):
        """
        Comprehensive system health check
        """
        print("\n💚 SYSTEM HEALTH CHECK")
        print("=" * 60)
        
        health_score = 100
        issues = []
        
        try:
            # Check 1: Recent activity
            log_files = glob.glob(f"{self.logs_dir}/algo_system_*.log")
            if log_files:
                latest_log = max(log_files, key=os.path.getmtime)
                age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(latest_log))
                
                if age.days > 1:
                    health_score -= 20
                    issues.append(f"No recent activity ({age.days} days)")
                else:
                    print("✅ Recent system activity detected")
            else:
                health_score -= 30
                issues.append("No system logs found")
            
            # Check 2: Data freshness
            data_files = glob.glob(f"{self.data_dir}/*_processed_data.csv")
            if data_files:
                latest_data = max(data_files, key=os.path.getmtime)
                age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(latest_data))
                
                if age.days > 7:
                    health_score -= 15
                    issues.append(f"Data is stale ({age.days} days)")
                else:
                    print("✅ Data is reasonably fresh")
            else:
                health_score -= 25
                issues.append("No processed data found")
            
            # Check 3: Error frequency
            error_files = glob.glob(f"{self.logs_dir}/algo_errors_*.log")
            if error_files:
                recent_errors = 0
                for error_file in error_files:
                    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(error_file))
                    if age.days <= 1:  # Errors in last day
                        try:
                            with open(error_file, 'r') as f:
                                recent_errors += len([line for line in f if 'ERROR' in line])
                        except:
                            pass
                
                if recent_errors > 10:
                    health_score -= 20
                    issues.append(f"High error rate ({recent_errors} recent errors)")
                elif recent_errors > 0:
                    health_score -= 5
                    print(f"⚠️ Some recent errors ({recent_errors})")
                else:
                    print("✅ No recent errors")
            
            # Check 4: Model files
            model_files = glob.glob(f"{self.data_dir}/*.pkl")
            if len(model_files) >= 3:
                print("✅ ML models are available")
            else:
                health_score -= 10
                issues.append("Missing ML model files")
            
            # Overall health assessment
            print(f"\n🏥 Overall Health Score: {health_score}/100")
            
            if health_score >= 90:
                print("💚 System Status: EXCELLENT")
            elif health_score >= 70:
                print("💛 System Status: GOOD")
            elif health_score >= 50:
                print("🧡 System Status: FAIR")
            else:
                print("❤️ System Status: NEEDS ATTENTION")
            
            if issues:
                print(f"\n⚠️ Issues Found:")
                for issue in issues:
                    print(f"   • {issue}")
            
        except Exception as e:
            print(f"❌ Error during health check: {e}")
    
    def cleanup_old_files(self, days_to_keep=30):
        """
        Clean up old log and temporary files
        """
        print(f"\n🧹 CLEANUP - Removing files older than {days_to_keep} days")
        print("=" * 60)
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            removed_count = 0
            
            # Clean log files
            for log_file in glob.glob(f"{self.logs_dir}/*.log") + glob.glob(f"{self.logs_dir}/*.json"):
                try:
                    file_age = datetime.fromtimestamp(os.path.getmtime(log_file))
                    if file_age < cutoff_date:
                        os.remove(log_file)
                        removed_count += 1
                        print(f"🗑️ Removed: {log_file}")
                except Exception as e:
                    print(f"⚠️ Could not remove {log_file}: {e}")
            
            print(f"✅ Cleanup completed: {removed_count} files removed")
            
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")

def main():
    """
    Main function for system monitoring
    """
    print("🔍 SYSTEM MONITOR - ALGO TRADING SYSTEM")
    print("=" * 80)
    
    monitor = SystemMonitor()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'status':
            monitor.get_system_status()
        elif command == 'health':
            monitor.check_system_health()
        elif command == 'performance':
            monitor.generate_performance_report()
        elif command == 'cleanup':
            days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
            monitor.cleanup_old_files(days)
        elif command == 'all':
            monitor.get_system_status()
            monitor.check_system_health()
            monitor.generate_performance_report()
        else:
            print("❌ Unknown command. Use: status, health, performance, cleanup, or all")
    else:
        # Default: show status and health
        monitor.get_system_status()
        monitor.check_system_health()

if __name__ == "__main__":
    import sys
    main()