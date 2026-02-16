"""
Flask Routes for Credit Card Fraud Detection Application
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from flask import (
    Blueprint, render_template, request, jsonify, 
    flash, redirect, url_for, current_app, send_file
)
from werkzeug.utils import secure_filename
from app.models.ensemble_model import FraudDetectionEnsemble
from app.models.data_processor import DataProcessor
from app.utils.helpers import allowed_file, generate_report

# Create blueprints
main_bp = Blueprint('main', __name__)
api_bp = Blueprint('api', __name__)

# Initialize models
fraud_detector = None
data_processor = None

def get_fraud_detector():
    """Lazy loading of fraud detector"""
    global fraud_detector
    if fraud_detector is None:
        fraud_detector = FraudDetectionEnsemble()
        fraud_detector.load_models()
    return fraud_detector

def get_data_processor():
    """Lazy loading of data processor"""
    global data_processor
    if data_processor is None:
        data_processor = DataProcessor()
    return data_processor


# ============== MAIN ROUTES ==============

@main_bp.route('/')
def index():
    """Landing page"""
    return render_template('index.html')


@main_bp.route('/predict')
def predict_page():
    """Single transaction prediction page"""
    return render_template('predict.html')


@main_bp.route('/batch-predict')
def batch_predict_page():
    """Batch prediction page"""
    return render_template('batch_predict.html')


@main_bp.route('/dashboard')
def dashboard():
    """Analytics dashboard"""
    detector = get_fraud_detector()
    metrics = detector.get_model_metrics()
    return render_template('dashboard.html', metrics=metrics)


@main_bp.route('/model-comparison')
def model_comparison():
    """Model comparison page"""
    detector = get_fraud_detector()
    comparison_data = detector.get_comparison_data()
    return render_template('model_comparison.html', data=comparison_data)


@main_bp.route('/analytics')
def analytics():
    """Advanced analytics page"""
    return render_template('analytics.html')


@main_bp.route('/history')
def history():
    """Prediction history page"""
    return render_template('history.html')


@main_bp.route('/about')
def about():
    """About page"""
    return render_template('about.html')


# ============== API ROUTES ==============

@api_bp.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for single transaction prediction
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Get feature values
        processor = get_data_processor()
        features = processor.prepare_single_transaction(data)
        
        # Get prediction from ensemble
        detector = get_fraud_detector()
        result = detector.predict(features)
        
        # Log prediction
        current_app.logger.info(f"Prediction made: {result['prediction']}")
        
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'probability': result['probability'],
            'risk_level': result['risk_level'],
            'model_predictions': result['model_predictions'],
            'feature_importance': result['feature_importance'],
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/batch-predict', methods=['POST'])
def batch_predict():
    """
    API endpoint for batch predictions
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process file
        processor = get_data_processor()
        df = processor.load_and_validate(filepath)
        
        # Get predictions
        detector = get_fraud_detector()
        results = detector.batch_predict(df)
        
        # Generate summary
        summary = {
            'total_transactions': len(results),
            'fraudulent': int(sum(results['prediction'])),
            'legitimate': int(len(results) - sum(results['prediction'])),
            'fraud_percentage': float(sum(results['prediction']) / len(results) * 100),
            'high_risk_count': int(sum(results['risk_level'] == 'High')),
            'medium_risk_count': int(sum(results['risk_level'] == 'Medium')),
            'low_risk_count': int(sum(results['risk_level'] == 'Low'))
        }
        
        # Clean up
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'summary': summary,
            'results': results.to_dict(orient='records')[:100],  # Limit to 100 for display
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/model-metrics', methods=['GET'])
def get_model_metrics():
    """Get model performance metrics"""
    try:
        detector = get_fraud_detector()
        metrics = detector.get_model_metrics()
        return jsonify({
            'success': True,
            'metrics': metrics
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/feature-importance', methods=['GET'])
def get_feature_importance():
    """Get feature importance data"""
    try:
        detector = get_fraud_detector()
        importance = detector.get_feature_importance()
        return jsonify({
            'success': True,
            'feature_importance': importance
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/model-comparison', methods=['GET'])
def get_model_comparison():
    """Get model comparison data"""
    try:
        detector = get_fraud_detector()
        comparison = detector.get_comparison_data()
        return jsonify({
            'success': True,
            'comparison': comparison
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/generate-report', methods=['POST'])
def generate_prediction_report():
    """Generate PDF report for predictions"""
    try:
        data = request.get_json()
        report_path = generate_report(data)
        return send_file(report_path, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': current_app.config['APP_VERSION'],
        'timestamp': datetime.now().isoformat()
    })