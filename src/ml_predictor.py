import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import ta

class MLPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def create_features(self, df):
        """Create features for ML model"""
        df = df.copy()
        
        # Technical indicators
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        df['MACD'] = ta.trend.MACD(df['Close']).macd()
        df['MACD_Signal'] = ta.trend.MACD(df['Close']).macd_signal()
        df['BB_Upper'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
        df['BB_Lower'] = ta.volatility.BollingerBands(df['Close']).bollinger_lband()
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        
        # Price-based features
        df['Price_Change'] = df['Close'].pct_change()
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Moving averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        
        # Relative position indicators
        df['Close_MA5_Ratio'] = df['Close'] / df['MA_5']
        df['Close_MA10_Ratio'] = df['Close'] / df['MA_10']
        df['Close_MA20_Ratio'] = df['Close'] / df['MA_20']
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
            df[f'RSI_Lag_{lag}'] = df['RSI'].shift(lag)
        
        return df
    
    def create_target(self, df):
        """Create target variable (next day price movement)"""
        df = df.copy()
        df['Next_Close'] = df['Close'].shift(-1)
        df['Target'] = (df['Next_Close'] > df['Close']).astype(int)
        return df
    
    def prepare_data(self, df):
        """Prepare data for training"""
        df = self.create_features(df)
        df = self.create_target(df)
        
        # Select feature columns
        feature_cols = [col for col in df.columns if col not in 
                       ['Open', 'High', 'Low', 'Close', 'Volume', 'Next_Close', 'Target']]
        
        # Remove rows with NaN values
        df_clean = df.dropna()
        
        if len(df_clean) < 10:
            raise ValueError(f"Insufficient data for training: {len(df_clean)} records (minimum 10 required)")
        
        X = df_clean[feature_cols]
        y = df_clean['Target']
        
        return X, y, feature_cols
    
    def train_model(self, df):
        """Train the ML model"""
        try:
            X, y, feature_cols = self.prepare_data(df)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.is_trained = True
            self.feature_cols = feature_cols
            
            return {
                'accuracy': accuracy,
                'classification_report': classification_report(y_test, y_pred),
                'feature_importance': dict(zip(feature_cols, self.model.feature_importances_))
            }
            
        except Exception as e:
            print(f"Error training model: {e}")
            return None
    
    def predict_next_day(self, df):
        """Predict next day price movement"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            df_features = self.create_features(df)
            
            # Get the latest data point
            latest_data = df_features[self.feature_cols].iloc[-1:].fillna(0)
            
            # Scale features
            latest_scaled = self.scaler.transform(latest_data)
            
            # Make prediction
            prediction = self.model.predict(latest_scaled)[0]
            probability = self.model.predict_proba(latest_scaled)[0]
            
            return {
                'prediction': prediction,  # 1 for up, 0 for down
                'probability_down': probability[0],
                'probability_up': probability[1],
                'confidence': max(probability)
            }
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None