"""
Data cleaning service.
Functions to handle missing values, duplicates, outliers, and text standardization.
"""
import numpy as np
import pandas as pd


def handle_missing_values(df: pd.DataFrame, strategies: dict) -> tuple[pd.DataFrame, list[str]]:
    """Handle missing values per column based on selected strategies.

    Args:
        df: Input DataFrame.
        strategies: Dict mapping column name → strategy.
            Strategies: 'keep', 'drop', 'mean', 'median', 'mode', 'constant'.

    Returns:
        Tuple of (cleaned DataFrame, list of log messages).
    """
    df = df.copy()
    log = []

    for col, strategy in strategies.items():
        if col not in df.columns:
            continue
        missing_before = int(df[col].isnull().sum())
        if missing_before == 0:
            continue

        if strategy == 'keep':
            continue
        elif strategy == 'drop':
            df = df.dropna(subset=[col])
            log.append(f"Dropped rows with missing values in '{col}' ({missing_before} rows affected)")
        elif strategy == 'mean':
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_val = df[col].mean()
                df[col] = df[col].fillna(fill_val)
                log.append(f"Filled '{col}' missing values with mean ({fill_val:.4f})")
            else:
                log.append(f"Cannot apply mean to non-numeric column '{col}' — skipped")
        elif strategy == 'median':
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_val = df[col].median()
                df[col] = df[col].fillna(fill_val)
                log.append(f"Filled '{col}' missing values with median ({fill_val:.4f})")
            else:
                log.append(f"Cannot apply median to non-numeric column '{col}' — skipped")
        elif strategy == 'mode':
            mode_vals = df[col].mode()
            if len(mode_vals) > 0:
                fill_val = mode_vals.iloc[0]
                df[col] = df[col].fillna(fill_val)
                log.append(f"Filled '{col}' missing values with mode ({fill_val})")
            else:
                log.append(f"No mode found for '{col}' — skipped")
        elif strategy == 'constant':
            fill_val = 0 if pd.api.types.is_numeric_dtype(df[col]) else 'Unknown'
            df[col] = df[col].fillna(fill_val)
            log.append(f"Filled '{col}' missing values with constant ({fill_val})")

    return df, log


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Remove duplicate rows, keeping first occurrence.

    Returns:
        Tuple of (deduplicated DataFrame, log messages).
    """
    before = len(df)
    df = df.drop_duplicates(keep='first').reset_index(drop=True)
    removed = before - len(df)
    log = []
    if removed > 0:
        log.append(f"Removed {removed} duplicate rows (kept first occurrence)")
    else:
        log.append("No duplicate rows found")
    return df, log


def handle_outliers(df: pd.DataFrame, method: str = 'clip') -> tuple[pd.DataFrame, list[str]]:
    """Handle outliers in numeric columns.

    Args:
        df: Input DataFrame.
        method: 'clip' to cap at IQR bounds, 'remove' to drop outlier rows, 'none' to skip.

    Returns:
        Tuple of (processed DataFrame, log messages).
    """
    if method == 'none':
        return df.copy(), ["Outlier handling skipped"]

    df = df.copy()
    log = []
    numeric_cols = df.select_dtypes(include='number').columns

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_mask = (df[col] < lower) | (df[col] > upper)
        outlier_count = int(outlier_mask.sum())

        if outlier_count == 0:
            continue

        if method == 'clip':
            df[col] = df[col].clip(lower=lower, upper=upper)
            log.append(f"Clipped {outlier_count} outliers in '{col}' to [{lower:.4f}, {upper:.4f}]")
        elif method == 'remove':
            df = df[~outlier_mask]
            log.append(f"Removed {outlier_count} outlier rows from '{col}'")

    if not log:
        log.append("No outliers detected in numeric columns")

    df = df.reset_index(drop=True)
    return df, log


def standardize_text(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Standardize text columns: strip whitespace and normalize case.

    Returns:
        Tuple of (cleaned DataFrame, log messages).
    """
    df = df.copy()
    log = []
    text_cols = df.select_dtypes(include='object').columns

    for col in text_cols:
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue
        # Strip whitespace
        original = df[col].copy()
        df[col] = df[col].astype(str).str.strip()
        # Replace 'nan' strings back to NaN
        df[col] = df[col].replace('nan', np.nan)
        changed = int((original.fillna('') != df[col].fillna('')).sum())
        if changed > 0:
            log.append(f"Standardized text in '{col}' ({changed} values trimmed)")

    if not log:
        log.append("No text standardization needed")

    return df, log
