
"""
Preprocessing and Feature Engineering Service
ISRAA - Data Intelligence Studio

Provides dataset cleaning, encoding, scaling, feature engineering,
feature selection, outlier detection and train/test splitting.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split


# ============================================================
# VALIDATION
# ============================================================

def validate_dataframe(df):
    """Validate that the input is a usable pandas DataFrame."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("The dataset is empty.")

    if len(df.columns) == 0:
        raise ValueError("The dataset has no columns.")

    return True


def get_numeric_columns(df):
    """Return numeric columns."""

    return df.select_dtypes(include=np.number).columns.tolist()


def get_categorical_columns(df):
    """Return categorical columns."""

    return df.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()


# ============================================================
# MISSING VALUES
# ============================================================

def handle_missing_values(
    df,
    strategy="mean",
    columns=None,
    constant_value=0,
):
    """
    Handle missing values.

    Strategies:
        mean, median, mode, constant,
        drop_rows, drop_columns
    """

    validate_dataframe(df)

    result = df.copy()

    selected = columns or result.columns.tolist()

    selected = [
        col for col in selected
        if col in result.columns
    ]

    if not selected:
        raise ValueError("No valid columns selected.")

    if strategy == "drop_rows":

        result = result.dropna(subset=selected)

    elif strategy == "drop_columns":

        result = result.drop(columns=selected)

    elif strategy == "mean":

        for col in selected:

            if pd.api.types.is_numeric_dtype(result[col]):

                value = result[col].mean()

                if pd.notna(value):
                    result[col] = result[col].fillna(value)

    elif strategy == "median":

        for col in selected:

            if pd.api.types.is_numeric_dtype(result[col]):

                value = result[col].median()

                if pd.notna(value):
                    result[col] = result[col].fillna(value)

    elif strategy == "mode":

        for col in selected:

            modes = result[col].mode(dropna=True)

            if not modes.empty:
                result[col] = result[col].fillna(modes.iloc[0])

    elif strategy == "constant":

        result[selected] = result[selected].fillna(constant_value)

    else:

        raise ValueError("Unsupported missing-value strategy.")

    return result


# ============================================================
# DUPLICATES AND EMPTY DATA
# ============================================================

def remove_duplicates(df):
    """Remove duplicated rows."""

    validate_dataframe(df)

    return df.drop_duplicates().reset_index(drop=True)


def remove_empty_rows(df):
    """Remove rows containing only missing values."""

    validate_dataframe(df)

    return df.dropna(how="all").reset_index(drop=True)


def remove_empty_columns(df):
    """Remove columns containing only missing values."""

    validate_dataframe(df)

    return df.dropna(axis=1, how="all")


def remove_constant_columns(df):
    """Remove columns containing only one unique value."""

    validate_dataframe(df)

    result = df.copy()

    constant_columns = [
        col for col in result.columns
        if result[col].nunique(dropna=False) <= 1
    ]

    return result.drop(columns=constant_columns)


# ============================================================
# DATA TYPE CONVERSION
# ============================================================

def convert_numeric_columns(df, columns=None):
    """Convert selected columns to numeric values."""

    validate_dataframe(df)

    result = df.copy()

    selected = columns or get_categorical_columns(result)

    for col in selected:

        if col in result.columns:

            result[col] = pd.to_numeric(
                result[col],
                errors="coerce",
            )

    return result


def convert_datetime_columns(df, columns):
    """Convert selected columns to datetime."""

    validate_dataframe(df)

    result = df.copy()

    for col in columns:

        if col in result.columns:

            result[col] = pd.to_datetime(
                result[col],
                errors="coerce",
            )

    return result


# ============================================================
# ENCODING
# ============================================================

def one_hot_encode(df, columns=None, drop_first=False):
    """Apply one-hot encoding to categorical columns."""

    validate_dataframe(df)

    result = df.copy()

    selected = columns or get_categorical_columns(result)

    selected = [
        col for col in selected
        if col in result.columns
    ]

    if not selected:
        return result

    return pd.get_dummies(
        result,
        columns=selected,
        drop_first=drop_first,
        dtype=int,
    )


def label_encode(df, columns=None):
    """Apply label encoding to selected categorical columns."""

    validate_dataframe(df)

    result = df.copy()

    selected = columns or get_categorical_columns(result)

    encoders = {}

    for col in selected:

        if col not in result.columns:
            continue

        encoder = LabelEncoder()

        values = (
            result[col]
            .fillna("__MISSING__")
            .astype(str)
        )

        result[col] = encoder.fit_transform(values)

        encoders[col] = encoder

    return result, encoders


# ============================================================
# SCALING
# ============================================================

def scale_features(df, method="standard", columns=None):
    """
    Scale numeric features.

    Methods:
        standard, minmax, robust
    """

    validate_dataframe(df)

    result = df.copy()

    selected = columns or get_numeric_columns(result)

    selected = [
        col for col in selected
        if col in result.columns
        and pd.api.types.is_numeric_dtype(result[col])
    ]

    if not selected:
        raise ValueError(
            "No numeric columns available for scaling."
        )

    if method == "standard":

        scaler = StandardScaler()

    elif method == "minmax":

        scaler = MinMaxScaler()

    elif method == "robust":

        scaler = RobustScaler()

    else:

        raise ValueError("Unsupported scaling method.")

    if result[selected].isna().any().any():
        raise ValueError(
            "Handle missing values before scaling."
        )

    result[selected] = scaler.fit_transform(result[selected])

    return result, scaler


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def extract_date_features(df, columns):
    """Extract useful features from datetime columns."""

    validate_dataframe(df)

    result = df.copy()

    for col in columns:

        if col not in result.columns:
            continue

        dates = pd.to_datetime(
            result[col],
            errors="coerce",
        )

        result[f"{col}_year"] = dates.dt.year
        result[f"{col}_month"] = dates.dt.month
        result[f"{col}_day"] = dates.dt.day
        result[f"{col}_dayofweek"] = dates.dt.dayofweek

    return result


def create_interaction_feature(
    df,
    column_a,
    column_b,
    operation="multiply",
):
    """Create a new feature from two numeric columns."""

    validate_dataframe(df)

    result = df.copy()

    if (
        column_a not in result.columns
        or column_b not in result.columns
    ):
        raise ValueError("Selected columns do not exist.")

    if not (
        pd.api.types.is_numeric_dtype(result[column_a])
        and pd.api.types.is_numeric_dtype(result[column_b])
    ):
        raise ValueError(
            "Interaction features require numeric columns."
        )

    if operation == "multiply":

        name = f"{column_a}_x_{column_b}"

        result[name] = result[column_a] * result[column_b]

    elif operation == "add":

        name = f"{column_a}_plus_{column_b}"

        result[name] = result[column_a] + result[column_b]

    elif operation == "subtract":

        name = f"{column_a}_minus_{column_b}"

        result[name] = result[column_a] - result[column_b]

    elif operation == "divide":

        name = f"{column_a}_div_{column_b}"

        result[name] = (
            result[column_a]
            / result[column_b].replace(0, np.nan)
        )

    else:

        raise ValueError("Unsupported interaction operation.")

    return result


# ============================================================
# FEATURE SELECTION
# ============================================================

def remove_low_variance_features(df, threshold=0.0):
    """Remove numeric features with low variance."""

    validate_dataframe(df)

    result = df.copy()

    numeric_columns = get_numeric_columns(result)

    if not numeric_columns:
        raise ValueError("No numeric features available.")

    threshold = float(threshold)

    if threshold < 0:
        raise ValueError("Variance threshold cannot be negative.")

    # Exclude columns containing only missing values.
    usable_columns = [
        col for col in numeric_columns
        if result[col].notna().any()
    ]

    missing_only_columns = [
        col for col in numeric_columns
        if col not in usable_columns
    ]

    if not usable_columns:

        return (
            result.drop(columns=missing_only_columns),
            [],
            missing_only_columns,
        )

    # Fill missing values temporarily for variance calculation.
    numeric_data = result[usable_columns].copy()

    for col in usable_columns:

        median_value = numeric_data[col].median()

        numeric_data[col] = numeric_data[col].fillna(
            median_value
        )

    selector = VarianceThreshold(threshold=threshold)

    try:

        selector.fit(numeric_data)

        selected_columns = [
            col
            for col, keep in zip(
                usable_columns,
                selector.get_support(),
            )
            if keep
        ]

    except ValueError:

        selected_columns = []

    removed_columns = [
        col for col in numeric_columns
        if col not in selected_columns
    ]

    result = result.drop(columns=removed_columns)

    return result, selected_columns, removed_columns


# ============================================================
# OUTLIER DETECTION
# ============================================================

def detect_outliers_iqr(df, columns=None):
    """Detect outliers using the IQR method."""

    validate_dataframe(df)

    selected = columns or get_numeric_columns(df)

    report = {}

    for col in selected:

        if col not in df.columns:
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        valid_values = df[col].dropna()

        if valid_values.empty:
            continue

        q1 = valid_values.quantile(0.25)
        q3 = valid_values.quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = (
            (df[col] < lower) |
            (df[col] > upper)
        ).sum()

        report[col] = {
            "lower_bound": float(lower),
            "upper_bound": float(upper),
            "outliers_count": int(count),
        }

    return report


def cap_outliers_iqr(df, columns=None):
    """Cap outliers using IQR boundaries."""

    validate_dataframe(df)

    result = df.copy()

    report = detect_outliers_iqr(result, columns)

    for col, values in report.items():

        result[col] = result[col].clip(
            lower=values["lower_bound"],
            upper=values["upper_bound"],
        )

    return result


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_data(
    df,
    target,
    test_size=0.2,
    random_state=42,
    shuffle=True,
):
    """Split dataset into training and testing sets."""

    validate_dataframe(df)

    if not target or target not in df.columns:
        raise ValueError("Target column does not exist.")

    test_size = float(test_size)

    if not 0.05 <= test_size <= 0.5:
        raise ValueError(
            "Test size must be between 5% and 50%."
        )

    if df[target].isna().any():
        raise ValueError(
            "Target column contains missing values."
        )

    X = df.drop(columns=[target])

    y = df[target]

    if X.shape[1] == 0:
        raise ValueError("No feature columns available.")

    if len(df) < 2:
        raise ValueError(
            "Not enough rows to split the dataset."
        )

    if shuffle and y.nunique() > 1:

        counts = y.value_counts()

        if counts.min() >= 2:

            try:

                return _stratified_split(
                    X,
                    y,
                    test_size,
                    random_state,
                )

            except ValueError:
                pass

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=int(random_state),
        shuffle=bool(shuffle),
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def _stratified_split(X, y, test_size, random_state):
    """Perform stratified splitting when class counts allow it."""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=int(random_state),
        shuffle=True,
        stratify=y,
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


# ============================================================
# DATASET REPORT
# ============================================================

def preprocessing_report(df):
    """Generate a detailed preprocessing report."""

    validate_dataframe(df)

    missing = df.isna().sum()

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicates": int(df.duplicated().sum()),
        "missing_total": int(missing.sum()),

        "missing_by_column": {
            col: int(value)
            for col, value in missing.items()
        },

        "numeric_columns": get_numeric_columns(df),

        "categorical_columns": get_categorical_columns(df),

        "memory_usage": int(
            df.memory_usage(deep=True).sum()
        ),
    }


# ============================================================
# RECOMMENDATIONS
# ============================================================

def generate_recommendations(df):
    """Generate preprocessing recommendations based on dataset quality."""

    validate_dataframe(df)

    recommendations = []

    missing_total = int(df.isna().sum().sum())

    duplicates = int(df.duplicated().sum())

    empty_columns = [
        col for col in df.columns
        if df[col].isna().all()
    ]

    constant_columns = [
        col for col in df.columns
        if df[col].nunique(dropna=False) <= 1
    ]

    numeric_columns = get_numeric_columns(df)

    categorical_columns = get_categorical_columns(df)

    # Missing values
    if missing_total > 0:

        recommendations.append(
            f"Handle {missing_total} missing values using "
            "mean, median or mode."
        )

    # Duplicates
    if duplicates > 0:

        recommendations.append(
            f"Remove {duplicates} duplicate rows."
        )

    # Empty columns
    if empty_columns:

        recommendations.append(
            f"Remove {len(empty_columns)} completely empty columns."
        )

    # Constant columns
    if constant_columns:

        recommendations.append(
            f"Remove {len(constant_columns)} constant columns."
        )

    # Categorical encoding
    if categorical_columns:

        recommendations.append(
            "Apply One-Hot Encoding or Label Encoding "
            "to categorical features."
        )

    # Numeric scaling
    if numeric_columns:

        recommendations.append(
            "Consider scaling numerical features "
            "before training machine learning models."
        )

    # Outlier detection
    if numeric_columns:

        outliers = detect_outliers_iqr(
            df,
            numeric_columns,
        )

        total_outliers = sum(
            item["outliers_count"]
            for item in outliers.values()
        )

        if total_outliers > 0:

            recommendations.append(
                f"Review {total_outliers} potential outliers "
                "using the IQR method."
            )

    # Dataset size
    if len(df) < 10:

        recommendations.append(
            "The dataset contains very few rows. "
            "Consider collecting more data before training."
        )

    if not recommendations:

        recommendations.append(
            "Your dataset looks clean. "
            "You can proceed to feature engineering or model training."
        )

    return recommendations


# ============================================================
# AUTOMATIC PREPROCESSING
# ============================================================

def auto_preprocess(df):
    """
    Automatically clean a dataset using basic preprocessing rules.

    Returns:
        processed_df, actions
    """

    validate_dataframe(df)

    result = df.copy()

    actions = []

    # 1. Remove completely empty rows
    before = len(result)

    result = remove_empty_rows(result)

    removed = before - len(result)

    if removed > 0:

        actions.append(
            f"Removed {removed} empty rows"
        )

    # 2. Remove completely empty columns
    before = len(result.columns)

    result = remove_empty_columns(result)

    removed = before - len(result.columns)

    if removed > 0:

        actions.append(
            f"Removed {removed} empty columns"
        )

    # 3. Remove duplicate rows
    before = len(result)

    result = remove_duplicates(result)

    removed = before - len(result)

    if removed > 0:

        actions.append(
            f"Removed {removed} duplicate rows"
        )

    # 4. Remove constant columns
    before = len(result.columns)

    result = remove_constant_columns(result)

    removed = before - len(result.columns)

    if removed > 0:

        actions.append(
            f"Removed {removed} constant columns"
        )

    # 5. Clean text spaces
    for col in get_categorical_columns(result):

        result[col] = result[col].apply(
            lambda value: value.strip()
            if isinstance(value, str)
            else value
        )

    actions.append("Cleaned unnecessary text spaces")

    # 6. Fill missing numeric values with median
    for col in get_numeric_columns(result):

        if result[col].isna().any():

            median_value = result[col].median()

            if pd.notna(median_value):

                result[col] = result[col].fillna(
                    median_value
                )

                actions.append(
                    f"Filled missing values in {col} using median"
                )

    # 7. Fill missing categorical values with mode
    for col in get_categorical_columns(result):

        if result[col].isna().any():

            modes = result[col].mode(dropna=True)

            if not modes.empty:

                result[col] = result[col].fillna(
                    modes.iloc[0]
                )

                actions.append(
                    f"Filled missing values in {col} using mode"
                )

    # 8. Try to convert numeric-looking text columns
    for col in get_categorical_columns(result):

        converted = pd.to_numeric(
            result[col],
            errors="coerce",
        )

        original_non_missing = result[col].notna()

        if (
            original_non_missing.any()
            and converted[original_non_missing].notna().all()
        ):

            result[col] = converted

            actions.append(
                f"Converted {col} to numeric"
            )

    if not actions:

        actions.append(
            "No preprocessing changes were necessary"
        )

    return result, actions


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(df, file_path):
    """Save processed dataset as CSV."""

    validate_dataframe(df)

    path = Path(file_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    return str(path)


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(file_path):
    """Load a CSV or Excel dataset."""

    path = Path(file_path)

    if not path.exists():

        raise FileNotFoundError(
            "Dataset file was not found."
        )

    extension = path.suffix.lower()

    if extension == ".csv":

        df = pd.read_csv(path)

    elif extension in (".xlsx", ".xls"):

        df = pd.read_excel(path)

    else:

        raise ValueError(
            "Unsupported dataset format."
        )

    validate_dataframe(df)

    return df
