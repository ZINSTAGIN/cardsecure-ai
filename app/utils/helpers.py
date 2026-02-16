"""
Helper Utilities for Credit Card Fraud Detection
"""

import os
import json
from datetime import datetime
from config import Config


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def generate_report(data):
    """Generate a prediction report"""
    # Create report directory
    report_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'reports')
    os.makedirs(report_dir, exist_ok=True)
    
    # Generate report filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(report_dir, f'fraud_report_{timestamp}.json')
    
    # Save report
    with open(report_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    return report_path


def format_currency(amount):
    """Format amount as currency"""
    return f"${amount:,.2f}"


def format_percentage(value):
    """Format value as percentage"""
    return f"{value * 100:.2f}%"


def get_risk_color(risk_level):
    """Get color based on risk level"""
    colors = {
        'Critical': '#DC2626',  # Red
        'High': '#F97316',      # Orange
        'Medium': '#FBBF24',    # Yellow
        'Low': '#22C55E',       # Green
        'Very Low': '#10B981'   # Emerald
    }
    return colors.get(risk_level, '#6B7280')


def calculate_statistics(predictions):
    """Calculate summary statistics for predictions"""
    total = len(predictions)
    if total == 0:
        return {}
    
    fraudulent = sum(1 for p in predictions if p['prediction'] == 1)
    legitimate = total - fraudulent
    
    return {
        'total': total,
        'fraudulent': fraudulent,
        'legitimate': legitimate,
        'fraud_rate': fraudulent / total * 100,
        'average_probability': sum(p['probability'] for p in predictions) / total
    }