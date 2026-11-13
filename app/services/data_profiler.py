"""
Data profiling and statistical analysis service.
Provides summary statistics, distributions, and correlation analysis.
"""
import numpy as np
import pandas as pd

from app.services.data_loader import detect_column_types


def get_data_summary(df):
    """
    Generate a high-level summary of the dataset.

    Args:
        df: pandas DataFrame.

    Returns:
        dict with summary metrics.
    """
    col_types = detect_column_types(df)
    numeric_cols = [c for c, t in col_types.items() if t == 'numeric']
    categorical_cols = [c for c, t in col_types.items() if t == 'categorical']
    boolean_cols = [c for c, t in col_types.items() if t == 'boolean']
    datetime_cols = [c for c, t in col_types.items() if t == 'datetime']

    total_cells = df.shape[0] * df.shape[1]
    total_missing = int(df.isnull().sum().sum())
    missing_pct = round((total_missing / total_cells * 100), 2) if total_cells > 0 else 0.0
    duplicate_rows = int(df.duplicated().sum())

    memory_bytes = int(df.memory_usage(deep=True).sum())
    if memory_bytes >= 1_048_576:
        memory_str = f"{memory_bytes / 1_048_576:.2f} MB"
    else:
        memory_str = f"{memory_bytes / 1024:.2f} KB"

    return {
        'num_rows': df.shape[0],
        'num_columns': df.shape[1],
        'numeric_columns': len(numeric_cols),
        'categorical_columns': len(categorical_cols),
        'boolean_columns': len(boolean_cols),
        'datetime_columns': len(datetime_cols),
        'total_missing': total_missing,
        'missing_percentage': missing_pct,
        'duplicate_rows': duplicate_rows,
        'memory_usage': memory_str,
        'memory_bytes': memory_bytes,
        'column_type_map': col_types,
    }


def get_column_statistics(df):
    """
    Compute per-column statistics suitable for display in a table.

    Args:
        df: pandas DataFrame.

    Returns:
        list of dicts, one per column, with keys:
        name, dtype, col_type, non_null, unique, missing, missing_pct,
        mean, std, min, max, mode.
    """
    col_types = detect_column_types(df)
    stats = []

    for col in df.columns:
        series = df[col]
        n = len(series)
        non_null = int(series.count())
        missing = int(series.isnull().sum())
        missing_pct = round(missing / n * 100, 1) if n > 0 else 0.0
        unique = int(series.nunique())
        col_type = col_types.get(col, 'text')

        row = {
            'name': col,
            'dtype': str(series.dtype),
            'col_type': col_type,
            'non_null': non_null,
            'unique': unique,
            'missing': missing,
            'missing_pct': missing_pct,
            'mean': None,
            'std': None,
            'min': None,
            'max': None,
            'mode': None,
        }

        if col_type == 'numeric':
            row['mean'] = round(float(series.mean()), 4) if non_null > 0 else None
            row['std'] = round(float(series.std()), 4) if non_null > 1 else None
            row['min'] = _safe_scalar(series.min())
            row['max'] = _safe_scalar(series.max())
        elif col_type in ('categorical', 'boolean', 'text'):
            mode_vals = series.mode()
            row['mode'] = str(mode_vals.iloc[0]) if len(mode_vals) > 0 else None

        stats.append(row)

    return stats


def get_distribution_data(df, column):
    """
    Get distribution data for a single column, suitable for Plotly charts.

    Args:
        df: pandas DataFrame.
        column: column name.

    Returns:
        dict with chart_type and data payload.
    """
    if column not in df.columns:
        return {'error': f'Column "{column}" not found'}

    col_types = detect_column_types(df)
    col_type = col_types.get(column, 'text')
    series = df[column].dropna()

    if col_type == 'numeric':
        values = series.tolist()
        return {
            'chart_type': 'histogram',
            'column': column,
            'col_type': col_type,
            'values': values,
            'stats': {
                'mean': round(float(series.mean()), 4),
                'median': round(float(series.median()), 4),
                'std': round(float(series.std()), 4),
                'skew': round(float(series.skew()), 4),
            },
        }
    else:
        # Categorical / boolean / text — bar chart of top 30 value counts
        counts = series.value_counts().head(30)
        return {
            'chart_type': 'bar',
            'column': column,
            'col_type': col_type,
            'labels': [str(v) for v in counts.index.tolist()],
            'values': counts.values.tolist(),
        }


def get_correlation_matrix(df):
    """
    Compute the correlation matrix for numeric columns.

    Args:
        df: pandas DataFrame.

    Returns:
        dict with columns (list of str), values (2-D list of floats).
        Returns None if fewer than 2 numeric columns.
    """
    numeric_df = df.select_dtypes(include='number')
    if numeric_df.shape[1] < 2:
        return None

    corr = numeric_df.corr()
    # Replace NaN with 0 for JSON serialisation
    corr = corr.fillna(0)

    return {
        'columns': corr.columns.tolist(),
        'values': corr.round(4).values.tolist(),
    }


# ---- helpers ----

def _safe_scalar(val):
    """Convert numpy scalars to Python native types for JSON serialisation."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return round(float(val), 4)
    return val
