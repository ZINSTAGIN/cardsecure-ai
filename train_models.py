"""
=============================================================
FIXED Model Training Script for CardSecure AI
Credit Card Fraud Detection - With Clear Fraud Patterns
=============================================================
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config


def create_fraud_sample():
    """
    Create a FRAUD transaction with DISTINCT patterns
    Key indicators: V1 negative, V2 positive, V4 positive, V14 very negative, V17 very negative
    """
    return {
        'Time': np.random.uniform(0, 172800),
        'V1': np.random.uniform(-4.0, -2.0),      # FRAUD: Very Negative
        'V2': np.random.uniform(2.0, 4.5),        # FRAUD: Very Positive
        'V3': np.random.uniform(-3.0, -1.0),      # FRAUD: Negative
        'V4': np.random.uniform(2.5, 5.5),        # FRAUD: Very Positive
        'V5': np.random.uniform(-3.0, -1.0),
        'V6': np.random.uniform(-2.5, -0.5),
        'V7': np.random.uniform(2.0, 4.5),        # FRAUD: Positive
        'V8': np.random.uniform(-4.0, -1.5),
        'V9': np.random.uniform(0.5, 2.5),
        'V10': np.random.uniform(-3.0, -0.5),
        'V11': np.random.uniform(-2.0, 0.0),
        'V12': np.random.uniform(-2.0, 1.0),
        'V13': np.random.uniform(-1.5, 0.5),
        'V14': np.random.uniform(-6.0, -3.0),     # FRAUD: VERY NEGATIVE (KEY!)
        'V15': np.random.uniform(-1.0, 1.0),
        'V16': np.random.uniform(-4.0, -1.0),
        'V17': np.random.uniform(-5.0, -2.0),     # FRAUD: VERY NEGATIVE (KEY!)
        'V18': np.random.uniform(-2.0, 0.0),
        'V19': np.random.uniform(0.0, 2.0),
        'V20': np.random.uniform(0.0, 1.0),
        'V21': np.random.uniform(0.0, 1.0),
        'V22': np.random.uniform(-1.5, 0.0),
        'V23': np.random.uniform(-0.5, 0.5),
        'V24': np.random.uniform(-1.0, 0.0),
        'V25': np.random.uniform(0.0, 1.0),
        'V26': np.random.uniform(-0.5, 0.5),
        'V27': np.random.uniform(0.0, 0.5),
        'V28': np.random.uniform(-0.5, 0.0),
        'Amount': np.random.uniform(500, 5000),   # FRAUD: High Amount
        'Class': 1  # FRAUD
    }


def create_legitimate_sample():
    """
    Create a LEGITIMATE transaction with NORMAL patterns
    All values close to zero, normal amounts
    """
    return {
        'Time': np.random.uniform(0, 172800),
        'V1': np.random.uniform(-1.0, 1.0),       # LEGIT: Near Zero
        'V2': np.random.uniform(-1.0, 1.0),       # LEGIT: Near Zero
        'V3': np.random.uniform(-0.5, 2.0),       # LEGIT: Slightly Positive
        'V4': np.random.uniform(-1.0, 1.0),       # LEGIT: Near Zero
        'V5': np.random.uniform(-1.0, 1.0),
        'V6': np.random.uniform(-1.0, 1.0),
        'V7': np.random.uniform(-1.0, 1.0),       # LEGIT: Near Zero
        'V8': np.random.uniform(-0.5, 0.5),
        'V9': np.random.uniform(-1.0, 1.0),
        'V10': np.random.uniform(-1.0, 1.0),
        'V11': np.random.uniform(-1.5, 1.5),
        'V12': np.random.uniform(-1.5, 1.5),
        'V13': np.random.uniform(-1.5, 0.5),
        'V14': np.random.uniform(-1.0, 2.0),      # LEGIT: Normal Range
        'V15': np.random.uniform(-1.0, 1.0),
        'V16': np.random.uniform(-1.0, 1.0),
        'V17': np.random.uniform(-1.0, 1.0),      # LEGIT: Normal Range
        'V18': np.random.uniform(-1.0, 1.0),
        'V19': np.random.uniform(-1.0, 1.0),
        'V20': np.random.uniform(-0.5, 0.5),
        'V21': np.random.uniform(-0.5, 0.5),
        'V22': np.random.uniform(-1.0, 1.0),
        'V23': np.random.uniform(-0.5, 0.5),
        'V24': np.random.uniform(-0.5, 0.5),
        'V25': np.random.uniform(-0.5, 0.5),
        'V26': np.random.uniform(-0.5, 0.5),
        'V27': np.random.uniform(-0.3, 0.3),
        'V28': np.random.uniform(-0.3, 0.3),
        'Amount': np.random.uniform(1, 300),      # LEGIT: Low Amount
        'Class': 0  # LEGITIMATE
    }


def create_training_dataset():
    """Create balanced training dataset with clear patterns"""
    
    print("=" * 60)
    print("   CREATING TRAINING DATASET WITH CLEAR FRAUD PATTERNS")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create samples
    n_fraud = 3000
    n_legitimate = 7000
    
    print(f"\n📊 Generating {n_fraud + n_legitimate} samples...")
    print(f"   ├── 🚨 Fraud samples: {n_fraud}")
    print(f"   └── ✅ Legitimate samples: {n_legitimate}")
    
    samples = []
    
    # Generate fraud samples
    print("\n⏳ Generating fraud samples...")
    for i in range(n_fraud):
        samples.append(create_fraud_sample())
        if (i + 1) % 1000 == 0:
            print(f"   Fraud: {i + 1}/{n_fraud}")
    
    # Generate legitimate samples
    print("\n⏳ Generating legitimate samples...")
    for i in range(n_legitimate):
        samples.append(create_legitimate_sample())
        if (i + 1) % 2000 == 0:
            print(f"   Legitimate: {i + 1}/{n_legitimate}")
    
    # Create DataFrame
    df = pd.DataFrame(samples)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    data_dir = os.path.join(Config.DATA_PATH, 'raw')
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, 'creditcard.csv')
    df.to_csv(filepath, index=False)
    
    print(f"\n✅ Dataset saved: {filepath}")
    
    # Print sample statistics
    print("\n📈 Feature ranges for FRAUD vs LEGITIMATE:")
    print("-" * 50)
    fraud_df = df[df['Class'] == 1]
    legit_df = df[df['Class'] == 0]
    
    key_features = ['V1', 'V2', 'V4', 'V14', 'V17', 'Amount']
    for feat in key_features:
        f_mean = fraud_df[feat].mean()
        l_mean = legit_df[feat].mean()
        print(f"   {feat:8s} | Fraud: {f_mean:8.2f} | Legit: {l_mean:8.2f}")
    
    return filepath


def train_all_models():
    """Main training function"""
    
    print("\n")
    print("=" * 60)
    print("   CARDSECURE AI - MODEL TRAINING")
    print("=" * 60)
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Create directories
    os.makedirs(Config.MODEL_PATH, exist_ok=True)
    
    # Create fresh training dataset
    data_path = create_training_dataset()
    
    # Import modules
    from app.models.data_processor import DataProcessor
    from app.models.ensemble_model import FraudDetectionEnsemble
    
    # Load and preprocess data
    print("\n" + "=" * 60)
    print("   PREPROCESSING DATA")
    print("=" * 60)
    
    processor = DataProcessor()
    df = processor.load_raw_data(data_path)
    
    print(f"\n📊 Dataset loaded:")
    print(f"   ├── Total samples: {len(df):,}")
    print(f"   ├── Fraud samples: {int(df['Class'].sum()):,} ({df['Class'].mean()*100:.1f}%)")
    print(f"   └── Legitimate samples: {len(df) - int(df['Class'].sum()):,}")
    
    # Preprocess
    X_train, X_test, y_train, y_test = processor.preprocess_for_training(df)
    
    print(f"\n📦 After preprocessing:")
    print(f"   ├── Training set: {len(X_train):,} samples")
    print(f"   ├── Test set: {len(X_test):,} samples")
    print(f"   └── Training fraud ratio: {y_train.mean()*100:.1f}%")
    
    # Train models
    print("\n" + "=" * 60)
    print("   TRAINING MODELS")
    print("=" * 60)
    
    ensemble = FraudDetectionEnsemble()
    metrics = ensemble.train(X_train, y_train, X_test, y_test)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("   TRAINING COMPLETE - FINAL RESULTS")
    print("=" * 60)
    
    print(f"\n{'Model':<25} {'Accuracy':>10} {'F1-Score':>10} {'ROC-AUC':>10}")
    print("-" * 57)
    
    sorted_models = sorted(metrics.items(), key=lambda x: x[1].get('f1_score', 0), reverse=True)
    for model_name, model_metrics in sorted_models:
        acc = model_metrics.get('accuracy', 0) * 100
        f1 = model_metrics.get('f1_score', 0)
        roc = model_metrics.get('roc_auc', 0)
        print(f"{model_name:<25} {acc:>9.2f}% {f1:>10.4f} {roc:>10.4f}")
    
    print("\n" + "=" * 60)
    print(f"   ✅ Models saved to: {Config.MODEL_PATH}")
    print(f"   ✅ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("\n")
    
    return metrics


if __name__ == '__main__':
    try:
        train_all_models()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()