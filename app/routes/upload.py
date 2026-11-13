"""
Upload routes.
Handles file upload, validation, metadata extraction, and deletion.
"""
import os
import json
import uuid
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from app import db
from app.models.dataset import Dataset
from app.services.data_loader import load_dataset, get_basic_info

upload_bp = Blueprint('upload', __name__)


def _allowed_file(filename):
    """Check if the file extension is in the allowed set."""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']


@upload_bp.route('/')
def upload_page():
    """Show the upload form and list existing datasets."""
    datasets = Dataset.query.order_by(Dataset.uploaded_at.desc()).all()
    return render_template('upload.html', datasets=datasets)


@upload_bp.route('/', methods=['POST'])
def upload_file():
    """Handle the file upload POST request."""
    # --- Validate presence ---
    if 'file' not in request.files:
        flash('No file part in the request.', 'error')
        return redirect(url_for('upload.upload_page'))

    file = request.files['file']
    if file.filename == '':
        flash('No file selected.', 'error')
        return redirect(url_for('upload.upload_page'))

    # --- Validate extension ---
    if not _allowed_file(file.filename):
        flash('Unsupported file type. Please upload a CSV, XLS, or XLSX file.', 'error')
        return redirect(url_for('upload.upload_page'))

    try:
        # --- Save to disk with a UUID prefix to avoid collisions ---
        original_filename = file.filename
        ext = original_filename.rsplit('.', 1)[1].lower()
        safe_name = f"{uuid.uuid4().hex}_{original_filename}"
        upload_folder = current_app.config['UPLOAD_FOLDER']
        file_path = os.path.join(upload_folder, safe_name)
        file.save(file_path)

        file_size = os.path.getsize(file_path)

        # --- Load & extract metadata ---
        df = load_dataset(file_path)
        info = get_basic_info(df)

        # --- Persist to database ---
        dataset = Dataset(
            name=os.path.splitext(original_filename)[0],
            original_filename=original_filename,
            file_path=file_path,
            file_size=file_size,
            file_type=ext,
            num_rows=info['num_rows'],
            num_columns=info['num_columns'],
            column_names=json.dumps(info['column_names']),
            column_types=json.dumps(info['column_types']),
        )
        db.session.add(dataset)
        db.session.commit()

        flash(f'Dataset "{dataset.name}" uploaded successfully! '
              f'({info["num_rows"]:,} rows × {info["num_columns"]} columns)', 'success')
        return redirect(url_for('explore.explore_dataset', dataset_id=dataset.id))

    except Exception as e:
        db.session.rollback()
        flash(f'Upload failed: {str(e)}', 'error')
        return redirect(url_for('upload.upload_page'))


@upload_bp.route('/delete/<int:dataset_id>', methods=['POST'])
def delete_dataset(dataset_id):
    """Delete a dataset record and its file from disk."""
    try:
        dataset = Dataset.query.get_or_404(dataset_id)

        # Remove files from disk
        for path in (dataset.file_path, dataset.processed_path):
            if path and os.path.exists(path):
                os.remove(path)

        name = dataset.name
        db.session.delete(dataset)
        db.session.commit()

        flash(f'Dataset "{name}" deleted successfully.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Delete failed: {str(e)}', 'error')

    return redirect(url_for('main.dashboard'))
