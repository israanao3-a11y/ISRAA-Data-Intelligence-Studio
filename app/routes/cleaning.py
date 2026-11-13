"""
Data cleaning routes.
Handles dataset selection, analysis and cleaning.
"""

import os
import json
import pandas as pd

from flask import (
    Blueprint,
    render_template,
    request,
    flash,
    redirect,
    url_for,
    current_app,
    send_file
)

from app import db
from app.models.dataset import Dataset
from app.services.data_loader import load_dataset


cleaning_bp = Blueprint('cleaning', __name__)


def get_stats(df):
    """Calculate dataset statistics."""

    return {
        'rows': len(df),
        'columns': len(df.columns),
        'missing': int(df.isnull().sum().sum()),
        'duplicates': int(df.duplicated().sum())
    }


def get_preview(df):
    """Prepare dataset preview."""

    preview = df.head(10).copy()

    preview = preview.astype(object).where(
        pd.notna(preview), ''
    )

    return preview.columns.tolist(), preview.values.tolist()


@cleaning_bp.route('/', methods=['GET', 'POST'])
def cleaning_page():

    datasets = Dataset.query.order_by(
        Dataset.uploaded_at.desc()
    ).all()

    stats = None
    preview_columns = []
    preview_data = []
    selected_dataset_id = ''

    if request.method == 'POST':

        dataset_id = request.form.get('dataset_id')

        if not dataset_id:
            flash('Please select a dataset.', 'warning')
            return redirect(url_for('cleaning.cleaning_page'))

        dataset = db.session.get(Dataset, int(dataset_id))

        if dataset is None:
            flash('Dataset not found.', 'error')
            return redirect(url_for('cleaning.cleaning_page'))

        selected_dataset_id = dataset.id

        try:
            # Load original dataset
            df = load_dataset(dataset.file_path)

            # Apply cleaning operations
            if request.form.get('clean'):

                operations = []

                if request.form.get('remove_duplicates'):
                    before = len(df)
                    df = df.drop_duplicates()
                    operations.append(
                        f'Removed {before - len(df)} duplicate rows'
                    )

                if request.form.get('remove_empty'):
                    df = df.dropna(how='all')
                    df = df.dropna(axis=1, how='all')
                    operations.append(
                        'Removed empty rows and columns'
                    )

                if request.form.get('trim_spaces'):
                    for col in df.select_dtypes(
                        include=['object', 'string']
                    ).columns:
                        df[col] = df[col].apply(
                            lambda x: x.strip()
                            if isinstance(x, str) else x
                        )

                    operations.append('Removed unnecessary spaces')

                if request.form.get('handle_missing'):
                    for col in df.columns:

                        if pd.api.types.is_numeric_dtype(df[col]):
                            if df[col].isnull().any():
                                median = df[col].median()
                                if pd.notna(median):
                                    df[col] = df[col].fillna(median)

                        else:
                            if df[col].isnull().any():
                                mode = df[col].mode()
                                if not mode.empty:
                                    df[col] = df[col].fillna(mode[0])

                    operations.append('Handled missing values')

                # Save cleaned dataset
                upload_folder = current_app.config['UPLOAD_FOLDER']
                os.makedirs(upload_folder, exist_ok=True)

                output_name = f'cleaned_{dataset.id}.csv'
                output_path = os.path.join(
                    upload_folder, output_name
                )

                df.to_csv(output_path, index=False)

                dataset.processed_path = output_path
                dataset.processing_log = json.dumps(operations)

                dataset.num_rows = len(df)
                dataset.num_columns = len(df.columns)

                dataset.column_names = json.dumps(
                    df.columns.tolist()
                )

                db.session.commit()

                flash(
                    'Dataset cleaned and saved successfully!',
                    'success'
                )

            # Statistics and preview
            stats = get_stats(df)

            preview_columns, preview_data = get_preview(df)

        except Exception as e:
            db.session.rollback()
            flash(f'Cleaning error: {str(e)}', 'error')

    return render_template(
        'cleaning.html',
        datasets=datasets,
        stats=stats,
        preview_columns=preview_columns,
        preview_data=preview_data,
        selected_dataset_id=selected_dataset_id
    )


@cleaning_bp.route('/download/<int:dataset_id>')
def download_cleaned(dataset_id):

    dataset = db.session.get(Dataset, dataset_id)

    if dataset is None or not dataset.processed_path:
        flash('No cleaned dataset available.', 'warning')
        return redirect(url_for('cleaning.cleaning_page'))

    if not os.path.exists(dataset.processed_path):
        flash('Cleaned file not found.', 'error')
        return redirect(url_for('cleaning.cleaning_page'))

    return send_file(
        dataset.processed_path,
        as_attachment=True,
        download_name=f'{dataset.name}_cleaned.csv'
    )
