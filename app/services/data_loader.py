"""
Data loading and file ingestion service.
Handles reading CSV/Excel files and extracting metadata.
"""
import os
import pandas as pd


def load_dataset(file_path):
    """
    Load a dataset from a file path into a pandas DataFrame.

    Args:
        file_path: Absolute path to the data file (csv, xls, xlsx).

    Returns:
        pandas.DataFrame with the loaded data.

    Raises:
        ValueError: If the file extension is not supported.
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.csv':
        df = pd.read_csv(file_path)
    elif ext in ('.xls', '.xlsx'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    return df


def detect_column_types(df):
    """
    Detect and categorize column types in a DataFrame.

    Args:
        df: pandas DataFrame.

    Returns:
        dict mapping column names to type labels:
        'numeric', 'categorical', 'boolean', 'datetime', or 'text'.
    """
    type_map = {}
    for col in df.columns:
        dtype = df[col].dtype

        if pd.api.types.is_bool_dtype(dtype):
            type_map[col] = 'boolean'
        elif pd.api.types.is_numeric_dtype(dtype):
            # Check if it's really a boolean encoded as int (only 0/1)
            unique_vals = df[col].dropna().unique()
            if set(unique_vals).issubset({0, 1, 0.0, 1.0}) and len(unique_vals) <= 2:
                type_map[col] = 'boolean'
            else:
                type_map[col] = 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            type_map[col] = 'datetime'
        else:
            # Try to parse as datetime
            try:
                pd.to_datetime(df[col].dropna().head(20))
                type_map[col] = 'datetime'
            except (ValueError, TypeError):
                nunique = df[col].nunique()
                # Heuristic: if unique values < 50% of rows and < 100, it's categorical
                if nunique < min(len(df) * 0.5, 100):
                    type_map[col] = 'categorical'
                else:
                    type_map[col] = 'text'

    return type_map


def get_basic_info(df):
    """
    Extract basic metadata from a DataFrame.

    Args:
        df: pandas DataFrame.

    Returns:
        dict with keys: num_rows, num_columns, column_names, column_types,
        memory_usage_bytes.
    """
    column_types = detect_column_types(df)

    return {
        'num_rows': len(df),
        'num_columns': len(df.columns),
        'column_names': list(df.columns),
        'column_types': column_types,
        'memory_usage_bytes': int(df.memory_usage(deep=True).sum()),
    }
