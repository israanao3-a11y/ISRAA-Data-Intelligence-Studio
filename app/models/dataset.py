"""
Dataset metadata model.
Stores information about uploaded and processed datasets.
"""
from datetime import datetime
from app import db


class Dataset(db.Model):
    """Represents an uploaded dataset and its processing state."""
    __tablename__ = 'datasets'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    file_size = db.Column(db.Integer)  # bytes
    file_type = db.Column(db.String(10))  # csv, xls, xlsx

    # Dataset metadata
    num_rows = db.Column(db.Integer)
    num_columns = db.Column(db.Integer)
    column_names = db.Column(db.Text)  # JSON list
    column_types = db.Column(db.Text)  # JSON dict

    # Processing state
    processed_path = db.Column(db.String(500))  # path to cleaned/processed version
    processing_log = db.Column(db.Text)  # JSON list of operations applied

    # Timestamps
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<Dataset {self.name} ({self.num_rows}x{self.num_columns})>'
