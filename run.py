"""
Application Entry Point
Run this script to start the Flask application
"""

import os
from app import create_app

# Get environment
env = os.environ.get('FLASK_ENV', 'development')

# Create application
app = create_app(env)

if __name__ == '__main__':
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Run application
    print(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║          FraudGuard AI - Credit Card Fraud Detection      ║
    ╠══════════════════════════════════════════════════════════╣
    ║  Environment: {env:<43} ║
    ║  Server:      http://127.0.0.1:{port:<26} ║
    ║  Debug Mode:  {'Enabled' if app.debug else 'Disabled':<43} ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=app.debug
    )