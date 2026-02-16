"""
Ensemble Model for CardSecure AI
Credit Card Fraud Detection - Fixed Prediction Logic
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    average_precision_score
)

import xgboost as xgb
import lightgbm as lgb

from config import Config


class FraudDetectionEnsemble:
    """
    Ensemble of multiple ML models for fraud detection
    """
    
    def __init__(self):
        self.models = {}
        self.ensemble_model = None
        self.stacking_model = None
        self.model_metrics = {}
        self.feature_importance = {}
        self.is_loaded = False
        
        self.model_path = Config.MODEL_PATH
        self.metrics_path = os.path.join(self.model_path, 'model_metrics.json')
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all models"""
        
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=Config.RANDOM_STATE,
            n_jobs=-1
        )
        
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=3,
            random_state=Config.RANDOM_STATE,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )
        
        self.models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=Config.RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        )
        
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=Config.RANDOM_STATE
        )
        
        self.models['extra_trees'] = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            class_weight='balanced',
            random_state=Config.RANDOM_STATE,
            n_jobs=-1
        )
        
        self.models['logistic_regression'] = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=Config.RANDOM_STATE,
            solver='saga',
            n_jobs=-1
        )
        
        self.models['neural_network'] = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size=256,
            learning_rate='adaptive',
            max_iter=300,
            random_state=Config.RANDOM_STATE,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        self.models['adaboost'] = AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=Config.RANDOM_STATE
        )
    
    def _create_voting_ensemble(self):
        """Create soft voting ensemble"""
        estimators = [
            ('rf', self.models['random_forest']),
            ('xgb', self.models['xgboost']),
            ('lgb', self.models['lightgbm']),
            ('gb', self.models['gradient_boosting']),
            ('et', self.models['extra_trees']),
        ]
        
        self.ensemble_model = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )
        
        return self.ensemble_model
    
    def _create_stacking_ensemble(self):
        """Create stacking ensemble"""
        base_estimators = [
            ('rf', self.models['random_forest']),
            ('xgb', self.models['xgboost']),
            ('lgb', self.models['lightgbm']),
            ('et', self.models['extra_trees'])
        ]
        
        meta_learner = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=Config.RANDOM_STATE
        )
        
        self.stacking_model = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=3,
            n_jobs=-1,
            passthrough=False
        )
        
        return self.stacking_model
    
    def train(self, X_train, y_train, X_test, y_test):
        """Train all models"""
        
        print("\n" + "=" * 60)
        print("   TRAINING INDIVIDUAL MODELS")
        print("=" * 60)
        
        for name, model in self.models.items():
            print(f"\n🔄 Training {name}...")
            try:
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                if hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(X_test)[:, 1]
                else:
                    y_prob = y_pred.astype(float)
                
                self.model_metrics[name] = self._calculate_metrics(y_test, y_pred, y_prob)
                
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = dict(zip(
                        Config.FEATURE_COLUMNS,
                        model.feature_importances_.tolist()
                    ))
                
                acc = self.model_metrics[name]['accuracy']
                f1 = self.model_metrics[name]['f1_score']
                roc = self.model_metrics[name]['roc_auc']
                
                print(f"   ✅ Accuracy: {acc:.4f}")
                print(f"   ✅ F1-Score: {f1:.4f}")
                print(f"   ✅ ROC-AUC:  {roc:.4f}")
                
                self._save_model(model, name)
                
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
        
        # Train voting ensemble
        print(f"\n🔄 Training Voting Ensemble...")
        try:
            self._create_voting_ensemble()
            self.ensemble_model.fit(X_train, y_train)
            
            y_pred = self.ensemble_model.predict(X_test)
            y_prob = self.ensemble_model.predict_proba(X_test)[:, 1]
            
            self.model_metrics['voting_ensemble'] = self._calculate_metrics(y_test, y_pred, y_prob)
            self._save_model(self.ensemble_model, 'voting_ensemble')
            
            print(f"   ✅ F1-Score: {self.model_metrics['voting_ensemble']['f1_score']:.4f}")
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
        
        # Train stacking ensemble
        print(f"\n🔄 Training Stacking Ensemble...")
        try:
            self._create_stacking_ensemble()
            self.stacking_model.fit(X_train, y_train)
            
            y_pred = self.stacking_model.predict(X_test)
            y_prob = self.stacking_model.predict_proba(X_test)[:, 1]
            
            self.model_metrics['stacking_ensemble'] = self._calculate_metrics(y_test, y_pred, y_prob)
            self._save_model(self.stacking_model, 'stacking_ensemble')
            
            print(f"   ✅ F1-Score: {self.model_metrics['stacking_ensemble']['f1_score']:.4f}")
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
        
        self._save_metrics()
        
        return self.model_metrics
    
    def _calculate_metrics(self, y_true, y_pred, y_prob):
        """Calculate metrics"""
        return {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_true, y_prob)),
            'average_precision': float(average_precision_score(y_true, y_prob)),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
    
    def _save_model(self, model, name):
        """Save model"""
        filepath = os.path.join(self.model_path, f'{name}_model.pkl')
        joblib.dump(model, filepath)
        print(f"   💾 Saved: {name}_model.pkl")
    
    def _save_metrics(self):
        """Save metrics"""
        with open(self.metrics_path, 'w') as f:
            json.dump({
                'metrics': self.model_metrics,
                'feature_importance': self.feature_importance,
                'timestamp': datetime.now().isoformat()
            }, f, indent=4)
    
    def load_models(self):
        """Load trained models"""
        
        model_files = {
            'random_forest': 'random_forest_model.pkl',
            'xgboost': 'xgboost_model.pkl',
            'lightgbm': 'lightgbm_model.pkl',
            'gradient_boosting': 'gradient_boosting_model.pkl',
            'extra_trees': 'extra_trees_model.pkl',
            'logistic_regression': 'logistic_regression_model.pkl',
            'neural_network': 'neural_network_model.pkl',
            'adaboost': 'adaboost_model.pkl',
        }
        
        loaded = 0
        for name, filename in model_files.items():
            filepath = os.path.join(self.model_path, filename)
            if os.path.exists(filepath):
                try:
                    self.models[name] = joblib.load(filepath)
                    loaded += 1
                except:
                    pass
        
        # Load ensembles
        voting_path = os.path.join(self.model_path, 'voting_ensemble_model.pkl')
        if os.path.exists(voting_path):
            try:
                self.ensemble_model = joblib.load(voting_path)
                loaded += 1
            except:
                pass
        
        stacking_path = os.path.join(self.model_path, 'stacking_ensemble_model.pkl')
        if os.path.exists(stacking_path):
            try:
                self.stacking_model = joblib.load(stacking_path)
                loaded += 1
            except:
                pass
        
        # Load metrics
        if os.path.exists(self.metrics_path):
            try:
                with open(self.metrics_path, 'r') as f:
                    data = json.load(f)
                    self.model_metrics = data.get('metrics', {})
                    self.feature_importance = data.get('feature_importance', {})
            except:
                pass
        
        self.is_loaded = True
        print(f"   ✅ Loaded {loaded} models")
    
    def predict(self, features):
        """Make prediction - FIXED LOGIC"""
        if not self.is_loaded:
            self.load_models()
        
        model_predictions = {}
        all_probs = []
        fraud_votes = 0
        total_votes = 0
        
        # Get predictions from all models
        for name, model in self.models.items():
            if model is not None:
                try:
                    pred = int(model.predict(features)[0])
                    
                    if hasattr(model, 'predict_proba'):
                        prob = float(model.predict_proba(features)[0][1])
                    else:
                        prob = float(pred)
                    
                    model_predictions[name] = {
                        'prediction': pred,
                        'probability': prob
                    }
                    
                    all_probs.append(prob)
                    total_votes += 1
                    
                    if pred == 1:
                        fraud_votes += 1
                        
                except Exception as e:
                    continue
        
        # Get ensemble prediction
        ensemble_prob = None
        
        if self.stacking_model is not None:
            try:
                ensemble_prob = float(self.stacking_model.predict_proba(features)[0][1])
            except:
                pass
        
        if ensemble_prob is None and self.ensemble_model is not None:
            try:
                ensemble_prob = float(self.ensemble_model.predict_proba(features)[0][1])
            except:
                pass
        
        # Calculate final prediction
        if all_probs:
            avg_prob = np.mean(all_probs)
            max_prob = np.max(all_probs)
            
            if ensemble_prob is not None:
                final_prob = ensemble_prob
            else:
                final_prob = (avg_prob * 0.6) + (max_prob * 0.4)
            
            # Lower threshold for fraud detection
            fraud_ratio = fraud_votes / total_votes if total_votes > 0 else 0
            
            if final_prob >= 0.35 or fraud_ratio >= 0.5:
                final_pred = 1
            else:
                final_pred = 0
        else:
            final_prob = 0.5
            final_pred = 0
        
        risk_level = self._get_risk_level(final_prob)
        top_features = self._get_top_features()
        
        return {
            'prediction': final_pred,
            'prediction_label': 'Fraudulent' if final_pred == 1 else 'Legitimate',
            'probability': final_prob,
            'risk_level': risk_level,
            'model_predictions': model_predictions,
            'feature_importance': top_features
        }
    
    def batch_predict(self, df):
        """Batch predictions - FIXED"""
        if not self.is_loaded:
            self.load_models()
        
        from app.models.data_processor import DataProcessor
        processor = DataProcessor()
        features = processor.prepare_batch_data(df)
        
        results = []
        
        for idx in range(len(features)):
            row = features.iloc[[idx]]
            
            all_probs = []
            fraud_votes = 0
            total_votes = 0
            
            for name, model in self.models.items():
                if model is not None:
                    try:
                        pred = int(model.predict(row)[0])
                        if hasattr(model, 'predict_proba'):
                            prob = float(model.predict_proba(row)[0][1])
                        else:
                            prob = float(pred)
                        
                        all_probs.append(prob)
                        total_votes += 1
                        if pred == 1:
                            fraud_votes += 1
                    except:
                        continue
            
            if all_probs:
                avg_prob = np.mean(all_probs)
                max_prob = np.max(all_probs)
                final_prob = (avg_prob * 0.6) + (max_prob * 0.4)
                
                fraud_ratio = fraud_votes / total_votes if total_votes > 0 else 0
                
                if final_prob >= 0.35 or fraud_ratio >= 0.5:
                    final_pred = 1
                else:
                    final_pred = 0
            else:
                final_prob = 0.5
                final_pred = 0
            
            amount = 0
            if 'Amount' in df.columns:
                amount = float(df.iloc[idx]['Amount'])
            
            results.append({
                'prediction': final_pred,
                'probability': final_prob,
                'risk_level': self._get_risk_level(final_prob),
                'prediction_label': 'Fraudulent' if final_pred == 1 else 'Legitimate',
                'amount': amount
            })
        
        return pd.DataFrame(results)
    
    def _get_risk_level(self, probability):
        """Get risk level"""
        if probability >= 0.75:
            return 'Critical'
        elif probability >= 0.55:
            return 'High'
        elif probability >= 0.35:
            return 'Medium'
        elif probability >= 0.15:
            return 'Low'
        else:
            return 'Very Low'
    
    def _get_top_features(self, top_n=10):
        """Get top features"""
        if 'xgboost' in self.feature_importance:
            importance = self.feature_importance['xgboost']
        elif 'random_forest' in self.feature_importance:
            importance = self.feature_importance['random_forest']
        else:
            return {}
        
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_features[:top_n])
    
    def get_model_metrics(self):
        """Get metrics"""
        if not self.model_metrics and os.path.exists(self.metrics_path):
            with open(self.metrics_path, 'r') as f:
                data = json.load(f)
                self.model_metrics = data.get('metrics', {})
        return self.model_metrics
    
    def get_feature_importance(self):
        """Get feature importance"""
        return self.feature_importance
    
    def get_comparison_data(self):
        """Get comparison data"""
        metrics = self.get_model_metrics()
        
        comparison = []
        for model_name, model_metrics in metrics.items():
            comparison.append({
                'model': model_name,
                'accuracy': model_metrics.get('accuracy', 0),
                'precision': model_metrics.get('precision', 0),
                'recall': model_metrics.get('recall', 0),
                'f1_score': model_metrics.get('f1_score', 0),
                'roc_auc': model_metrics.get('roc_auc', 0)
            })
        
        return comparison