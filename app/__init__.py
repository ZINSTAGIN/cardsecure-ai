"""
Flask Application Factory
Credit Card Fraud Detection System
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask
from flask_cors import CORS
from config import config_dict


def create_app(config_name='default'):
    """
    Application factory function
    """
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(config_dict[config_name])
    
    # Enable CORS
    CORS(app)
    
    # Create necessary directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODEL_PATH'], exist_ok=True)
    os.makedirs(os.path.dirname(app.config['LOG_FILE']), exist_ok=True)
    
    # Configure logging
    configure_logging(app)
    
    # Register blueprints
    from app.routes import main_bp, api_bp
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Register error handlers
    register_error_handlers(app)
    
    app.logger.info('Credit Card Fraud Detection Application Started')
    
    return app


def configure_logging(app):
    """Configure application logging"""
    if not app.debug and not app.testing:
        file_handler = RotatingFileHandler(
            app.config['LOG_FILE'],
            maxBytes=10240000,
            backupCount=10
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.INFO)


def register_error_handlers(app):
    """Register error handlers"""
    
    @app.errorhandler(404)
    def not_found_error(error):
        from flask import render_template
        return render_template('errors/404.html'), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        from flask import render_template
        return render_template('errors/500.html'), 500