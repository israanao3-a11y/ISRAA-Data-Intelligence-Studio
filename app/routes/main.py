
"""
Dashboard / main routes.
Serves the landing page with dataset overview and quick-start actions.
"""

import json

from flask import Blueprint, render_template

from app import db
from app.models.dataset import Dataset


main_bp = Blueprint('main', __name__)


@main_bp.route('/healthz')
def healthz():
    """Health check endpoint for Render."""
    return {'status': 'ok'}, 200


@main_bp.route('/')
def dashboard():
    """Render the main dashboard page with dataset statistics."""

    try:
        datasets = Dataset.query.order_by(
            Dataset.uploaded_at.desc()
        ).all()

        total_datasets = len(datasets)

        total_rows = sum(
            d.num_rows or 0 for d in datasets
        )

        total_columns = sum(
            d.num_columns or 0 for d in datasets
        )

        latest_upload = (
            datasets[0].uploaded_at if datasets else None
        )

        return render_template(
            'dashboard.html',
            datasets=datasets,
            total_datasets=total_datasets,
            total_rows=total_rows,
            total_columns=total_columns,
            latest_upload=latest_upload,
        )

    except Exception as e:
        """Graceful fallback if database query fails."""

        return render_template(
            'dashboard.html',
            datasets=[],
            total_datasets=0,
            total_rows=0,
            total_columns=0,
            latest_upload=None,
            error=str(e),
        )
