
"""
ISRAA - Data Intelligence Studio
Flask application factory.
"""

import os

from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def create_app():
    """Create and configure the Flask application."""

    app = Flask(__name__)

    app.config.from_object('app.config.Config')

    # Ensure required directories exist
    for folder_key in [
        'UPLOAD_FOLDER',
        'PROCESSED_FOLDER',
        'MODELS_FOLDER',
        'REPORTS_FOLDER'
    ]:
        os.makedirs(app.config[folder_key], exist_ok=True)

    os.makedirs(app.instance_path, exist_ok=True)

    # Initialize extensions
    db.init_app(app)

    # Register blueprints
    from app.routes.main import main_bp
    from app.routes.upload import upload_bp
    from app.routes.explore import explore_bp
    from app.routes.cleaning import cleaning_bp
    from app.routes.preprocessing import preprocessing_bp
    from app.routes.ml import ml_bp
    from app.routes.evaluation import evaluation_bp
    from app.routes.reports import reports_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(upload_bp, url_prefix='/upload')
    app.register_blueprint(explore_bp, url_prefix='/explore')
    app.register_blueprint(cleaning_bp, url_prefix='/cleaning')
    app.register_blueprint(preprocessing_bp, url_prefix='/preprocessing')
    app.register_blueprint(ml_bp, url_prefix='/ml')
    app.register_blueprint(evaluation_bp, url_prefix='/evaluation')
    app.register_blueprint(reports_bp, url_prefix='/reports')

    # Create database tables
    with app.app_context():
        from app.models.dataset import Dataset
        db.create_all()

    return app
