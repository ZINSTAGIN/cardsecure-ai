"""
Configuration settings for the Credit Card Fraud Detection Application
"""

import os
from datetime import timedelta

class Config:
    """Base configuration class"""
    
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-super-secret-key-change-in-production'
    DEBUG = False
    TESTING = False
    
    # Changed App Name
    APP_NAME = "CardSecure AI"
    APP_VERSION = "2.0.0"
    
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
    
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')
    DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    
    DATABASE_URI = os.environ.get('DATABASE_URI') or 'sqlite:///fraud_detection.db'
    
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    
    FEATURE_COLUMNS = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
                       'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
                       'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27',
                       'V28', 'Amount']
    
    FRAUD_THRESHOLD = 0.5
    LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'app.log')
    LOG_LEVEL = 'INFO'


class DevelopmentConfig(Config):
    DEBUG = True
    ENV = 'development'


class ProductionConfig(Config):
    DEBUG = False
    ENV = 'production'


class TestingConfig(Config):
    TESTING = True
    DEBUG = True


config_dict = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}