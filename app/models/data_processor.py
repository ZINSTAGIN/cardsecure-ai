"""
=============================================================
FIXED Data Processor for CardSecure AI
Credit Card Fraud Detection
=============================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os
from config import Config


class DataProcessor:
    """
    Handles all data preprocessing - FIXED VERSION
    """
    
    def __init__(self):
        self.scaler = None
        self.feature_columns = Config.FEATURE_COLUMNS
        self.scaler_path = os.path.join(Config.MODEL_PATH, 'scaler.pkl')
        self._load_scaler()
    
    def _load_scaler(self):
        """Load the fitted scaler"""
        try:
            if os.path.exists(self.scaler_path) and os.path.getsize(self.scaler_path) > 0:
                self.scaler = joblib.load(self.scaler_path)
            else:
                self.scaler = None
        except Exception as e:
            print(f"⚠️ Scaler load warning: {e}")
            self.scaler = None
    
    def load_and_validate(self, filepath):
        """Load and validate data from file"""
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported file format")
        
        return df
    
    def prepare_single_transaction(self, data):
        """
        Prepare a single transaction for prediction - FIXED
        """
        # Create DataFrame
        features = pd.DataFrame([data])
        
        # Ensure all columns exist with default 0
        for col in self.feature_columns:
            if col not in features.columns:
                features[col] = 0.0
        
        # Convert to float
        for col in self.feature_columns:
            features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0.0)
        
        # Reorder columns
        features = features[self.feature_columns]
        
        # Scale if scaler is available and fitted
        if self.scaler is not None:
            try:
                features_scaled = self.scaler.transform(features)
                features = pd.DataFrame(features_scaled, columns=self.feature_columns)
            except Exception as e:
                print(f"⚠️ Scaling skipped: {e}")
        
        return features
    
    def prepare_batch_data(self, df):
        """
        Prepare batch data for prediction - FIXED
        """
        # Select available feature columns
        available_cols = [col for col in self.feature_columns if col in df.columns]
        features = df[available_cols].copy()
        
        # Add missing columns with 0
        for col in self.feature_columns:
            if col not in features.columns:
                features[col] = 0.0
        
        # Convert to float
        for col in self.feature_columns:
            features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0.0)
        
        # Reorder columns
        features = features[self.feature_columns]
        
        # Scale if scaler is available
        if self.scaler is not None:
            try:
                features_scaled = self.scaler.transform(features)
                features = pd.DataFrame(features_scaled, columns=self.feature_columns)
            except Exception as e:
                print(f"⚠️ Batch scaling skipped: {e}")
        
        return features
    
    @staticmethod
    def load_raw_data(filepath):
        """Load raw credit card data for training"""
        df = pd.read_csv(filepath)
        return df
    
    def preprocess_for_training(self, df, target_column='Class'):
        """
        Full preprocessing pipeline for model training - FIXED
        """
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Fill missing values
        X = X.fillna(0)
        
        # Create and fit scaler
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Save scaler
        os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
        joblib.dump(self.scaler, self.scaler_path)
        print(f"   💾 Scaler saved")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            test_size=Config.TEST_SIZE,
            random_state=Config.RANDOM_STATE,
            stratify=y
        )
        
        # Apply SMOTE for balancing
        try:
            smote = SMOTE(sampling_strategy=0.7, random_state=Config.RANDOM_STATE)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            print(f"   ✅ SMOTE applied: {len(X_train)} → {len(X_train_resampled)} samples")
        except Exception as e:
            print(f"   ⚠️ SMOTE skipped: {e}")
            X_train_resampled, y_train_resampled = X_train, y_train
        
        return X_train_resampled, X_test, y_train_resampled, y_test