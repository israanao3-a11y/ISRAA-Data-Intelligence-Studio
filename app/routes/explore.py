"""
Data exploration routes.
Provides dataset profiling, statistics, distributions, and correlation analysis.
"""
import json
from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
from app import db
from app.models.dataset import Dataset
from app.services.data_loader import load_dataset
from app.services.data_profiler import (
    get_data_summary,
    get_column_statistics,
    get_distribution_data,
    get_correlation_matrix,
)

explore_bp = Blueprint('explore', __name__)


def _load_df(dataset):
    """Load the DataFrame from the most up-to-date file for a dataset."""
    path = dataset.processed_path or dataset.file_path
    return load_dataset(path)


@explore_bp.route('/')
def explore_page():
    """Show dataset picker or redirect to exploration if dataset_id is given."""
    dataset_id = request.args.get('dataset_id', type=int)
    if dataset_id:
        return redirect(url_for('explore.explore_dataset', dataset_id=dataset_id))

    datasets = Dataset.query.order_by(Dataset.uploaded_at.desc()).all()
    return render_template('explore.html', datasets=datasets, dataset=None)


@explore_bp.route('/<int:dataset_id>')
def explore_dataset(dataset_id):
    """Full exploration view for a single dataset."""
    try:
        dataset = Dataset.query.get_or_404(dataset_id)
        df = _load_df(dataset)

        # All datasets for the selector dropdown
        datasets = Dataset.query.order_by(Dataset.uploaded_at.desc()).all()

        # Profiling
        summary = get_data_summary(df)
        col_stats = get_column_statistics(df)
        correlation = get_correlation_matrix(df)

        # Data preview (first 100 rows as list-of-dicts for DataTables)
        preview_df = df.head(100)
        preview_columns = list(preview_df.columns)
        preview_data = preview_df.fillna('').astype(str).values.tolist()

        return render_template(
            'explore.html',
            datasets=datasets,
            dataset=dataset,
            summary=summary,
            col_stats=col_stats,
            correlation=correlation,
            preview_columns=preview_columns,
            preview_data=preview_data,
        )
    except Exception as e:
        flash(f'Error exploring dataset: {str(e)}', 'error')
        return redirect(url_for('explore.explore_page'))


@explore_bp.route('/api/distribution/<int:dataset_id>/<column>')
def get_distribution(dataset_id, column):
    """API endpoint returning JSON distribution data for a column."""
    try:
        dataset = Dataset.query.get_or_404(dataset_id)
        df = _load_df(dataset)
        dist = get_distribution_data(df, column)
        return jsonify(dist)
    except Exception as e:
        return jsonify({'error': str(e)}), 400
