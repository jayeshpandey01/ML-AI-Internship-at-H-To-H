#!/usr/bin/env python3
"""
ML Trading System - Advanced Machine Learning Integration
Implements comprehensive ML features, model training, and prediction integration
"""

import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

class MLTradingSystem:
    def __init__(self, model_type='random_forest', test_size=0.33):
        """
        Initialize ML Trading System
        
        Parameters:
        - model_type: 'random_forest', 'logistic_regression', or 'decision_tree'
        - test_size: Proportion of data for testing (0.33 = last 2 months of 6-month data)
        """
        self.model_type = model_type
        self.test_size = test_size
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        
        # Initialize model based on type
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == 'decision_tree':
            self.model = DecisionTreeClassifier(random_state=42, max_depth=8)
        else:
            raise ValueError("Model type must be 'random_forest', 'logistic_regression', or 'decision_tree'")
        
        print(f"🤖 ML Trading System Initialized:")
        print(f"   Model Type: {model_type.replace('_', ' ').title()}")
        print(f"   Test Size: {test_size:.1%} (last {test_size*6:.1f} months)")
        
    def prepare_ml_features(self, df):
        """
        Prepare comprehensive ML features from stock data
        
        Features include:
        - Technical indicators: RSI, MACD, 20-DMA, 50-DMA, Volume
        - Lagged returns: 1-day, 3-day, 5-day returns
        - Price ratios and volatility measures
        - Volume indicators
        """
        df = df.copy()
        
        print("🔧 Preparing ML features...")
        
        # Technical Indicators
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        
        # MACD components
        macd_indicator = ta.trend.MACD(df['Close'])
        df['MACD'] = macd_indicator.macd()
        df['MACD_Signal'] = macd_indicator.macd_signal()
        df['MACD_Histogram'] = macd_indicator.macd_diff()
        
        # Moving Averages
        df['MA_20'] = df['Close'].rolling(window=20).mean()  # 20-DMA
        df['MA_50'] = df['Close'].rolling(window=50).mean()  # 50-DMA
        df['MA_10'] = df['Close'].rolling(window=10).mean()  # Additional short-term MA
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Price-based features
        df['Price_Change'] = df['Close'].pct_change()
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Open_Ratio'] = df['Close'] / df['Open']
        
        # Lagged returns (1-day, 3-day, 5-day)
        df['Return_1D'] = df['Close'].pct_change(1)
        df['Return_3D'] = df['Close'].pct_change(3)
        df['Return_5D'] = df['Close'].pct_change(5)
        
        # Volatility measures
        df['Volatility_10D'] = df['Price_Change'].rolling(window=10).std()
        df['Volatility_20D'] = df['Price_Change'].rolling(window=20).std()
        
        # Moving average ratios
        df['Price_MA20_Ratio'] = df['Close'] / df['MA_20']
        df['Price_MA50_Ratio'] = df['Close'] / df['MA_50']
        df['MA20_MA50_Ratio'] = df['MA_20'] / df['MA_50']
        
        # Bollinger Bands
        bb_indicator = ta.volatility.BollingerBands(df['Close'])
        df['BB_Upper'] = bb_indicator.bollinger_hband()
        df['BB_Lower'] = bb_indicator.bollinger_lband()
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Average True Range
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        
        # Stochastic Oscillator
        stoch_indicator = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = stoch_indicator.stoch()
        df['Stoch_D'] = stoch_indicator.stoch_signal()
        
        # Additional lag features
        for lag in [1, 2, 3, 5]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'RSI_Lag_{lag}'] = df['RSI'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
        
        print(f"✅ Created {len([col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']])} features")
        
        return df
    
    def create_target_variable(self, df):
        """
        Create binary target variable for next-day price movement
        1 = price increases next day, 0 = price decreases next day
        """
        df = df.copy()
        
        # Create next day close price
        df['Next_Close'] = df['Close'].shift(-1)
        
        # Binary target: 1 if next day price increases, 0 if decreases
        df['Target'] = (df['Next_Close'] > df['Close']).astype(int)
        
        # Remove the last row (no next day data)
        df = df[:-1]
        
        print(f"📊 Target variable created:")
        print(f"   Up days (1): {df['Target'].sum()} ({df['Target'].mean()*100:.1f}%)")
        print(f"   Down days (0): {len(df) - df['Target'].sum()} ({(1-df['Target'].mean())*100:.1f}%)")
        
        return df
    
    def prepare_training_data(self, df):
        """
        Prepare data for ML training by selecting features and handling missing values
        """
        # Create features and target
        df = self.prepare_ml_features(df)
        df = self.create_target_variable(df)
        
        # Select feature columns (exclude OHLCV and target)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Next_Close', 'Target']
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        print(f"🎯 Selected {len(self.feature_columns)} features for training:")
        for i, feature in enumerate(self.feature_columns[:10], 1):
            print(f"   {i:2d}. {feature}")
        if len(self.feature_columns) > 10:
            print(f"   ... and {len(self.feature_columns) - 10} more features")
        
        # Prepare feature matrix and target vector
        X = df[self.feature_columns]
        y = df['Target']
        
        # Handle missing values and infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
        
        # Remove any remaining NaN or infinite rows
        valid_indices = ~(X.isnull().any(axis=1) | y.isnull() | np.isinf(X).any(axis=1))
        X = X[valid_indices]
        y = y[valid_indices]
        
        # Additional check for infinite values
        if np.isinf(X.values).any():
            print("⚠️ Warning: Infinite values detected, replacing with column means...")
            X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())
        
        print(f"📈 Training data prepared:")
        print(f"   Samples: {len(X)}")
        print(f"   Features: {len(self.feature_columns)}")
        print(f"   Missing values handled: {df.isnull().sum().sum()} → 0")
        
        return X, y, df
    
    def train_model(self, df, stock_symbol='STOCK'):
        """
        Train ML model with time-based train/test split
        """
        print(f"\n🤖 Training ML model for {stock_symbol}...")
        
        # Prepare training data
        X, y, processed_df = self.prepare_training_data(df)
        
        if len(X) < 50:
            raise ValueError(f"Insufficient data for training: {len(X)} samples (minimum 50 required)")
        
        # Time-based split (first 4 months for training, last 2 months for testing)
        split_index = int(len(X) * (1 - self.test_size))
        
        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]
        
        print(f"📊 Data split:")
        print(f"   Training: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"   Testing: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print(f"🔄 Training {self.model_type.replace('_', ' ').title()} model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Calculate performance metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        
        # Classification report
        test_report = classification_report(y_test, y_test_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        
        # Feature importance (for tree-based models)
        feature_importance = {}
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        self.is_trained = True
        
        # Prepare results
        results = {
            'stock': stock_symbol,
            'model_type': self.model_type,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': test_report,
            'confusion_matrix': cm,
            'feature_importance': feature_importance,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'total_features': len(self.feature_columns),
            'processed_data': processed_df
        }
        
        return results
    
    def evaluate_model_performance(self, results):
        """Print comprehensive model evaluation"""
        print(f"\n{'='*60}")
        print(f"ML MODEL EVALUATION: {results['stock']}")
        print(f"{'='*60}")
        
        print(f"🤖 Model Configuration:")
        print(f"   Model Type: {results['model_type'].replace('_', ' ').title()}")
        print(f"   Training Samples: {results['train_samples']}")
        print(f"   Testing Samples: {results['test_samples']}")
        print(f"   Total Features: {results['total_features']}")
        
        print(f"\n📊 Performance Metrics:")
        print(f"   Training Accuracy: {results['train_accuracy']:.3f} ({results['train_accuracy']*100:.1f}%)")
        print(f"   Testing Accuracy: {results['test_accuracy']:.3f} ({results['test_accuracy']*100:.1f}%)")
        print(f"   Cross-Validation: {results['cv_mean']:.3f} ± {results['cv_std']:.3f}")
        
        # Check for overfitting
        overfitting = results['train_accuracy'] - results['test_accuracy']
        if overfitting > 0.1:
            print(f"   ⚠️ Potential Overfitting: {overfitting:.3f}")
        else:
            print(f"   ✅ Good Generalization: {overfitting:.3f}")
        
        print(f"\n📈 Classification Report:")
        report = results['classification_report']
        print(f"   Precision (Down): {report['0']['precision']:.3f}")
        print(f"   Precision (Up): {report['1']['precision']:.3f}")
        print(f"   Recall (Down): {report['0']['recall']:.3f}")
        print(f"   Recall (Up): {report['1']['recall']:.3f}")
        print(f"   F1-Score: {report['macro avg']['f1-score']:.3f}")
        
        print(f"\n🎯 Confusion Matrix:")
        cm = results['confusion_matrix']
        print(f"   True Negatives (Down→Down): {cm[0,0]}")
        print(f"   False Positives (Down→Up): {cm[0,1]}")
        print(f"   False Negatives (Up→Down): {cm[1,0]}")
        print(f"   True Positives (Up→Up): {cm[1,1]}")
        
        # Feature importance
        if results['feature_importance']:
            print(f"\n🔍 Top 10 Most Important Features:")
            for i, (feature, importance) in enumerate(list(results['feature_importance'].items())[:10], 1):
                print(f"   {i:2d}. {feature}: {importance:.4f}")
    
    def predict_next_day(self, df, stock_symbol='STOCK'):
        """
        Predict next-day price movement for the most recent data
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        print(f"\n🔮 Generating next-day prediction for {stock_symbol}...")
        
        # Prepare features for the latest data point
        df_features = self.prepare_ml_features(df)
        
        # Get the latest complete data point
        latest_data = df_features[self.feature_columns].iloc[-1:].fillna(df_features[self.feature_columns].mean())
        
        # Scale features
        latest_scaled = self.scaler.transform(latest_data)
        
        # Make prediction
        prediction = self.model.predict(latest_scaled)[0]
        prediction_proba = self.model.predict_proba(latest_scaled)[0]
        
        # Prepare prediction results
        prediction_result = {
            'stock': stock_symbol,
            'prediction_date': df.index[-1].strftime('%Y-%m-%d'),
            'current_price': df['Close'].iloc[-1],
            'prediction': prediction,  # 1 for up, 0 for down
            'probability_down': prediction_proba[0],
            'probability_up': prediction_proba[1],
            'confidence': max(prediction_proba),
            'direction': 'UP' if prediction == 1 else 'DOWN'
        }
        
        print(f"📊 Prediction Results:")
        print(f"   Current Price: ₹{prediction_result['current_price']:.2f}")
        print(f"   Prediction: {prediction_result['direction']}")
        print(f"   Confidence: {prediction_result['confidence']:.1%}")
        print(f"   Probability Up: {prediction_result['probability_up']:.1%}")
        print(f"   Probability Down: {prediction_result['probability_down']:.1%}")
        
        return prediction_result
    
    def save_model(self, filepath, results):
        """Save trained model and metadata"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'results': results
        }
        
        joblib.dump(model_data, filepath)
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model and metadata"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.model_type = model_data['model_type']
        self.is_trained = True
        
        print(f"📂 Model loaded from {filepath}")
        return model_data['results']

def main():
    """Test the ML Trading System"""
    from data_ingestion import EnhancedDataIngestion
    
    print("🚀 ML TRADING SYSTEM TEST")
    print("=" * 60)
    
    # Initialize components
    ingestion = EnhancedDataIngestion()
    
    # Test different models
    models = ['random_forest', 'logistic_regression', 'decision_tree']
    test_stock = 'RELIANCE.NS'
    
    # Load data
    print(f"Loading data for {test_stock}...")
    data = ingestion.load_saved_data(test_stock, 'processed')
    
    if data is None:
        print("❌ No processed data found. Please run data ingestion first.")
        return
    
    # Test each model
    for model_type in models:
        print(f"\n{'='*70}")
        print(f"TESTING {model_type.upper().replace('_', ' ')} MODEL")
        print(f"{'='*70}")
        
        try:
            # Initialize ML system
            ml_system = MLTradingSystem(model_type=model_type, test_size=0.33)
            
            # Train model
            results = ml_system.train_model(data, test_stock)
            
            # Evaluate performance
            ml_system.evaluate_model_performance(results)
            
            # Make prediction
            prediction = ml_system.predict_next_day(data, test_stock)
            
            # Save model
            model_filename = f'data/{test_stock.replace(".", "_")}_{model_type}_model.pkl'
            ml_system.save_model(model_filename, results)
            
        except Exception as e:
            print(f"❌ Error with {model_type}: {e}")
    
    print(f"\n🎉 ML Trading System testing completed!")

if __name__ == "__main__":
    main()