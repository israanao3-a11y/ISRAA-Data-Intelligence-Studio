
"""
Application configuration settings.
"""

import os

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)


class Config:
    """Base configuration."""

    SECRET_KEY = os.environ.get(
        'SECRET_KEY',
        'dev-only-secret-key'
    )

    # Database
    SQLALCHEMY_DATABASE_URI = (
        f"sqlite:///{os.path.join(BASE_DIR, 'instance', 'israa.db')}"
    )

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # File upload
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    PROCESSED_FOLDER = os.path.join(BASE_DIR, 'processed')
    MODELS_FOLDER = os.path.join(BASE_DIR, 'models')
    REPORTS_FOLDER = os.path.join(BASE_DIR, 'reports')

    MAX_CONTENT_LENGTH = 100 * 1024 * 1024

    ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx'}
