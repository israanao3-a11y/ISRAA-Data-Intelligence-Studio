
"""
Preprocessing Routes
ISRAA - Data Intelligence Studio

Handles preprocessing operations, dataset management and downloads.
"""

import uuid
from pathlib import Path

from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    current_app,
    send_file,
)

from app import db
from app.models.dataset import Dataset

from app.services.preprocessing import (
    load_dataset,
    save_dataset,
    handle_missing_values,
    remove_duplicates,
    remove_empty_rows,
    remove_empty_columns,
    remove_constant_columns,
    convert_numeric_columns,
    one_hot_encode,
    label_encode,
    scale_features,
    extract_date_features,
    create_interaction_feature,
    remove_low_variance_features,
    detect_outliers_iqr,
    cap_outliers_iqr,
    split_data,
    preprocessing_report,
    get_numeric_columns,
    get_categorical_columns,
    generate_recommendations,
    auto_preprocess,
)


preprocessing_bp = Blueprint(
    "preprocessing",
    __name__,
    url_prefix="/preprocessing",
)


# ============================================================
# DATASET MANAGEMENT
# ============================================================

def get_storage_directory():
    """Create and return the dataset storage directory."""

    directory = Path(current_app.instance_path) / "datasets"
    directory.mkdir(parents=True, exist_ok=True)

    return directory


def get_current_dataset_record():
    """Get the latest dataset stored in the database."""

    dataset = (
        Dataset.query
        .order_by(Dataset.id.desc())
        .first()
    )

    if dataset is None:
        raise ValueError(
            "No dataset found. Please upload a dataset first."
        )

    return dataset


def get_dataset_original_path(dataset):
    """Return the original uploaded dataset path."""

    if not dataset.file_path:
        raise ValueError(
            "The original dataset path is not available."
        )

    path = Path(dataset.file_path)

    if not path.exists():
        raise FileNotFoundError(
            "The original dataset file does not exist."
        )

    return path


def get_current_dataset_path():
    """
    Return the active dataset path.

    Priority:
    1. Processed dataset
    2. Original uploaded dataset
    """

    dataset = get_current_dataset_record()

    if dataset.processed_path:

        processed_path = Path(dataset.processed_path)

        if processed_path.exists():
            return processed_path

    return get_dataset_original_path(dataset)


def read_current_dataset():
    """Load the active dataset from its stored path."""

    path = get_current_dataset_path()

    return load_dataset(path)


def save_current_dataset(df):
    """Save processed data and update the database record."""

    dataset = get_current_dataset_record()

    filename = f"processed_{uuid.uuid4().hex}.csv"

    output_path = get_storage_directory() / filename

    saved_path = save_dataset(
        df,
        output_path,
    )

    dataset.processed_path = str(saved_path)

    db.session.commit()

    session.modified = True

    return Path(saved_path)


def get_selected_columns():
    """Read selected columns from the submitted form."""

    columns = request.form.getlist("columns")

    if not columns:
        single = request.form.get("column")

        if single:
            columns = [single]

    return columns


# ============================================================
# PAGE RENDERING
# ============================================================

def render_preprocessing_page(
    df=None,
    message=None,
    outlier_report=None,
):
    """Build all information needed by the preprocessing page."""

    try:

        if df is None:
            df = read_current_dataset()

        if df is None or df.empty:
            raise ValueError(
                "The selected dataset is empty."
            )

        report = preprocessing_report(df)

        recommendations = generate_recommendations(df)

        return render_template(
            "preprocessing.html",

            dataset_available=True,

            message=message,

            columns=df.columns.tolist(),

            numeric_columns=get_numeric_columns(df),

            categorical_columns=get_categorical_columns(df),

            preview_columns=df.columns.tolist(),

            preview_rows=df.head(10).fillna("").to_dict(
                orient="records"
            ),

            report=report,

            outlier_report=outlier_report,

            recommendations=recommendations,

            rows_count=len(df),

            columns_count=len(df.columns),

            split_available=bool(
                session.get("last_split_id")
            ),
        )

    except (ValueError, FileNotFoundError):

        return render_template(
            "preprocessing.html",

            dataset_available=False,

            message=message,

            outlier_report=outlier_report,

            recommendations=[],

            split_available=False,
        )


# ============================================================
# MAIN PAGE
# ============================================================

@preprocessing_bp.route("/", methods=["GET"])
def index():
    """Display the preprocessing workspace."""

    return render_preprocessing_page()


# ============================================================
# PREPROCESSING OPERATIONS
# ============================================================

@preprocessing_bp.route("/apply", methods=["POST"])
def apply():
    """Apply the selected preprocessing operation."""

    try:

        df = read_current_dataset()

        action = request.form.get(
            "action", ""
        ).strip()

        columns = get_selected_columns()

        if not action:
            raise ValueError(
                "Please select a preprocessing operation."
            )

        # ----------------------------------------------------
        # AUTOMATIC PREPROCESSING
        # ----------------------------------------------------

        if action == "auto":

            df, actions = auto_preprocess(df)

            save_current_dataset(df)

            if actions:
                message = (
                    "Automatic preprocessing completed: "
                    + ", ".join(actions)
                )
            else:
                message = (
                    "Automatic preprocessing completed. "
                    "No changes were necessary."
                )

            return render_preprocessing_page(
                df=df,
                message=message,
            )

        # ----------------------------------------------------
        # MISSING VALUES
        # ----------------------------------------------------

        elif action == "missing":

            strategy = request.form.get(
                "strategy", "mean"
            )

            constant = request.form.get(
                "constant_value", "0"
            )

            try:
                constant = float(constant)
            except ValueError:
                pass

            df = handle_missing_values(
                df,
                strategy=strategy,
                columns=columns or None,
                constant_value=constant,
            )

            message = "Missing values handled successfully."

        # ----------------------------------------------------
        # DUPLICATES
        # ----------------------------------------------------

        elif action == "duplicates":

            before = len(df)

            df = remove_duplicates(df)

            removed = before - len(df)

            message = (
                f"{removed} duplicate rows removed."
            )

        # ----------------------------------------------------
        # EMPTY ROWS
        # ----------------------------------------------------

        elif action == "empty_rows":

            before = len(df)

            df = remove_empty_rows(df)

            removed = before - len(df)

            message = f"{removed} empty rows removed."

        # ----------------------------------------------------
        # EMPTY COLUMNS
        # ----------------------------------------------------

        elif action == "empty_columns":

            before = len(df.columns)

            df = remove_empty_columns(df)

            removed = before - len(df.columns)

            message = f"{removed} empty columns removed."

        # ----------------------------------------------------
        # CONSTANT COLUMNS
        # ----------------------------------------------------

        elif action == "constant_columns":

            before = len(df.columns)

            df = remove_constant_columns(df)

            removed = before - len(df.columns)

            message = f"{removed} constant columns removed."

        # ----------------------------------------------------
        # NUMERIC CONVERSION
        # ----------------------------------------------------

        elif action == "numeric_conversion":

            df = convert_numeric_columns(
                df,
                columns=columns or None,
            )

            message = "Numeric conversion completed."

        # ----------------------------------------------------
        # ONE HOT ENCODING
        # ----------------------------------------------------

        elif action == "one_hot":

            df = one_hot_encode(
                df,
                columns=columns or None,
                drop_first=(
                    request.form.get("drop_first") == "yes"
                ),
            )

            message = "One-hot encoding completed."

        # ----------------------------------------------------
        # LABEL ENCODING
        # ----------------------------------------------------

        elif action == "label":

            df, encoders = label_encode(
                df,
                columns=columns or None,
            )

            message = "Label encoding completed."

        # ----------------------------------------------------
        # SCALING
        # ----------------------------------------------------

        elif action == "scaling":

            method = request.form.get(
                "method", "standard"
            )

            df, scaler = scale_features(
                df,
                method=method,
                columns=columns or None,
            )

            message = (
                f"{method.title()} scaling completed."
            )

        # ----------------------------------------------------
        # DATE FEATURES
        # ----------------------------------------------------

        elif action == "date_features":

            if not columns:
                raise ValueError(
                    "Select at least one date column."
                )

            df = extract_date_features(
                df,
                columns,
            )

            message = (
                "Date features extracted successfully."
            )

        # ----------------------------------------------------
        # INTERACTION FEATURES
        # ----------------------------------------------------

        elif action == "interaction":

            column_a = request.form.get("column_a")

            column_b = request.form.get("column_b")

            operation = request.form.get(
                "operation", "multiply"
            )

            df = create_interaction_feature(
                df,
                column_a,
                column_b,
                operation,
            )

            message = "New interaction feature created."

        # ----------------------------------------------------
        # LOW VARIANCE
        # ----------------------------------------------------

        elif action == "variance":

            threshold = float(
                request.form.get("threshold", 0.0)
            )

            df, selected, removed = (
                remove_low_variance_features(
                    df,
                    threshold,
                )
            )

            message = (
                "Feature selection completed. "
                f"{len(removed)} features removed."
            )

        # ----------------------------------------------------
        # OUTLIER CAPPING
        # ----------------------------------------------------

        elif action == "outlier_cap":

            df = cap_outliers_iqr(
                df,
                columns=columns or None,
            )

            message = (
                "Outliers capped using the IQR method."
            )

        # ----------------------------------------------------
        # TRAIN / TEST SPLIT
        # ----------------------------------------------------

        elif action == "split":

            target = request.form.get(
                "target", ""
            ).strip()

            if not target:
                raise ValueError(
                    "Please select a target column."
                )

            if target not in df.columns:
                raise ValueError(
                    "Selected target column does not exist."
                )

            test_size = float(
                request.form.get("test_size", 0.2)
            )

            if not 0.05 <= test_size <= 0.5:
                raise ValueError(
                    "Test size must be between 5% and 50%."
                )

            parts = split_data(
                df,
                target,
                test_size=test_size,
            )

            storage = get_storage_directory()

            split_id = uuid.uuid4().hex

            for name, part in parts.items():

                part_path = (
                    storage / f"{name}_{split_id}.csv"
                )

                part.to_csv(
                    part_path,
                    index=False,
                    encoding="utf-8-sig",
                )

            session["last_split_id"] = split_id
            session.modified = True

            message = (
                "Train/test split completed. "
                f"Training rows: {len(parts['X_train'])}, "
                f"testing rows: {len(parts['X_test'])}."
            )

            return render_preprocessing_page(
                df=df,
                message=message,
            )

        else:

            raise ValueError(
                "Unknown preprocessing operation."
            )

        # ----------------------------------------------------
        # SAVE PROCESSED DATASET
        # ----------------------------------------------------

        save_current_dataset(df)

        return render_preprocessing_page(
            df=df,
            message=message,
        )

    except (
        ValueError,
        TypeError,
        FileNotFoundError,
    ) as error:

        db.session.rollback()

        flash(str(error), "danger")

        return redirect(
            url_for("preprocessing.index")
        )

    except Exception as error:

        db.session.rollback()

        current_app.logger.exception(
            "Preprocessing operation failed."
        )

        flash(
            f"An unexpected error occurred: {error}",
            "danger",
        )

        return redirect(
            url_for("preprocessing.index")
        )


# ============================================================
# OUTLIER REPORT
# ============================================================

@preprocessing_bp.route(
    "/outliers",
    methods=["GET"],
)
def outliers():
    """Display outlier detection results."""

    try:

        df = read_current_dataset()

        report = detect_outliers_iqr(df)

        return render_preprocessing_page(
            df=df,
            outlier_report=report,
        )

    except Exception as error:

        flash(str(error), "danger")

        return redirect(
            url_for("preprocessing.index")
        )


# ============================================================
# RESET DATASET
# ============================================================

@preprocessing_bp.route(
    "/reset",
    methods=["POST"],
)
def reset():
    """Restore the original uploaded dataset."""

    try:

        dataset = get_current_dataset_record()

        original_path = get_dataset_original_path(dataset)

        # Remove the processed path from the database.
        dataset.processed_path = None

        db.session.commit()

        # Clear previous split information.
        session.pop("last_split_id", None)

        session.modified = True

        flash(
            "Dataset restored to its original version.",
            "success",
        )

    except Exception as error:

        db.session.rollback()

        flash(str(error), "danger")

    return redirect(
        url_for("preprocessing.index")
    )


# ============================================================
# DOWNLOAD PROCESSED DATASET
# ============================================================

@preprocessing_bp.route(
    "/download",
    methods=["GET"],
)
def download():
    """Download the active dataset."""

    try:

        path = get_current_dataset_path()

        extension = path.suffix.lower()

        if extension == ".csv":

            mimetype = "text/csv"

            download_name = "processed_dataset.csv"

        elif extension == ".xlsx":

            mimetype = (
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            )

            download_name = "processed_dataset.xlsx"

        elif extension == ".xls":

            mimetype = "application/vnd.ms-excel"

            download_name = "processed_dataset.xls"

        else:

            raise ValueError(
                "Unsupported download format."
            )

        return send_file(
            path,
            as_attachment=True,
            download_name=download_name,
            mimetype=mimetype,
        )

    except Exception as error:

        flash(str(error), "danger")

        return redirect(
            url_for("preprocessing.index")
        )


# ============================================================
# DOWNLOAD TRAIN / TEST FILES
# ============================================================

@preprocessing_bp.route(
    "/download-split/<part>",
    methods=["GET"],
)
def download_split(part):
    """Download one of the generated train/test datasets."""

    allowed = {
        "X_train",
        "X_test",
        "y_train",
        "y_test",
    }

    if part not in allowed:

        flash(
            "Invalid split file.",
            "danger",
        )

        return redirect(
            url_for("preprocessing.index")
        )

    split_id = session.get("last_split_id")

    if not split_id:

        flash(
            "No train/test split has been generated.",
            "danger",
        )

        return redirect(
            url_for("preprocessing.index")
        )

    path = (
        get_storage_directory()
        / f"{part}_{split_id}.csv"
    )

    if not path.exists():

        flash(
            "Split file not found.",
            "danger",
        )

        return redirect(
            url_for("preprocessing.index")
        )

    return send_file(
        path,
        as_attachment=True,
        download_name=f"{part}.csv",
        mimetype="text/csv",
    )
