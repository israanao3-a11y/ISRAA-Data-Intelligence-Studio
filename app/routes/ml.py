
"""
Machine Learning Routes
ISRAA - Data Intelligence Studio

Training, evaluation, model persistence and dataset analysis.
"""

import os
import json
import time
import copy
import uuid
import traceback

from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from flask import (
    Blueprint,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from app.models.dataset import Dataset


ml_bp = Blueprint("ml", __name__)


# ============================================================
# DATASET LOADING
# ============================================================

def _load_dataframe(dataset):

    path = dataset.processed_path or dataset.file_path

    if not path or not os.path.isfile(path):
        raise FileNotFoundError("Dataset file not found.")

    extension = os.path.splitext(path)[1].lower()

    if extension == ".csv":
        return pd.read_csv(path)

    if extension in (".xlsx", ".xls"):
        return pd.read_excel(path)

    raise ValueError("Unsupported dataset format.")


# ============================================================
# TRAINING RESULTS STORAGE
# ============================================================

def _get_results_folder():

    models_folder = current_app.config.get("MODELS_FOLDER")

    if not models_folder:
        raise ValueError("MODELS_FOLDER is not configured.")

    results_folder = os.path.join(
        models_folder,
        "training_results"
    )

    os.makedirs(results_folder, exist_ok=True)

    return results_folder


def _save_ml_results(results):

    """
    Save training results outside Flask session.
    Session stores only the JSON filename.
    """

    results_folder = _get_results_folder()

    filename = f"ml_results_{uuid.uuid4().hex}.json"

    final_path = os.path.join(
        results_folder,
        filename
    )

    temporary_path = final_path + ".tmp"

    try:

        with open(
            temporary_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                results,
                file,
                ensure_ascii=False,
                default=str,
                separators=(",", ":")
            )

        os.replace(
            temporary_path,
            final_path
        )

    except Exception:

        if os.path.exists(temporary_path):
            os.remove(temporary_path)

        raise

    session.pop("ml_results", None)

    session["ml_results_file"] = filename

    session.modified = True

    return filename


def _load_ml_results():

    """
    Load saved results using the filename stored in session.
    Supports old session data for backward compatibility.
    """

    filename = session.get("ml_results_file")

    if filename:

        if not isinstance(filename, str):
            return None

        if os.path.basename(filename) != filename:
            return None

        if not filename.endswith(".json"):
            return None

        results_path = os.path.join(
            _get_results_folder(),
            filename
        )

        if not os.path.isfile(results_path):
            return None

        with open(
            results_path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    # Legacy compatibility with previous versions.

    legacy_results = session.get("ml_results")

    if isinstance(legacy_results, str):

        return json.loads(legacy_results)

    if isinstance(legacy_results, dict):

        return legacy_results

    return None


# ============================================================
# AVAILABLE MODELS
# ============================================================

def _get_available_models():

    classification_models = {

        "logistic_regression": {
            "name": "Logistic Regression",
            "description": "Linear classification model",
            "icon": "fa-chart-line",
        },

        "decision_tree_clf": {
            "name": "Decision Tree",
            "description": "Tree-based classifier",
            "icon": "fa-sitemap",
        },

        "random_forest_clf": {
            "name": "Random Forest",
            "description": "Ensemble classifier",
            "icon": "fa-tree",
        },

        "gradient_boosting_clf": {
            "name": "Gradient Boosting",
            "description": "Boosting classifier",
            "icon": "fa-bolt",
        },

        "svm_clf": {
            "name": "Support Vector Machine",
            "description": "Support vector classifier",
            "icon": "fa-vector-square",
        },

        "knn_clf": {
            "name": "K-Nearest Neighbors",
            "description": "Nearest neighbors classifier",
            "icon": "fa-users",
        },

        "naive_bayes": {
            "name": "Naive Bayes",
            "description": "Probabilistic classifier",
            "icon": "fa-calculator",
        },

        "mlp_clf": {
            "name": "Neural Network (MLP)",
            "description": "Multi-layer perceptron",
            "icon": "fa-brain",
        },
    }

    regression_models = {

        "linear_regression": {
            "name": "Linear Regression",
            "description": "Linear regression model",
            "icon": "fa-chart-line",
        },

        "decision_tree_reg": {
            "name": "Decision Tree",
            "description": "Tree-based regression",
            "icon": "fa-sitemap",
        },

        "random_forest_reg": {
            "name": "Random Forest",
            "description": "Ensemble regression",
            "icon": "fa-tree",
        },

        "gradient_boosting_reg": {
            "name": "Gradient Boosting",
            "description": "Boosting regression",
            "icon": "fa-bolt",
        },

        "svr": {
            "name": "Support Vector Regression",
            "description": "Support vector regression",
            "icon": "fa-vector-square",
        },

        "knn_reg": {
            "name": "K-Nearest Neighbors",
            "description": "Nearest neighbors regression",
            "icon": "fa-users",
        },

        "mlp_reg": {
            "name": "Neural Network (MLP)",
            "description": "Multi-layer perceptron",
            "icon": "fa-brain",
        },
    }

    return classification_models, regression_models


# ============================================================
# MODEL FACTORY
# ============================================================

def _build_model(model_key, random_state=42):

    from sklearn.linear_model import (
        LogisticRegression,
        LinearRegression,
    )

    from sklearn.tree import (
        DecisionTreeClassifier,
        DecisionTreeRegressor,
    )

    from sklearn.ensemble import (
        RandomForestClassifier,
        RandomForestRegressor,
        GradientBoostingClassifier,
        GradientBoostingRegressor,
    )

    from sklearn.svm import SVC, SVR

    from sklearn.neighbors import (
        KNeighborsClassifier,
        KNeighborsRegressor,
    )

    from sklearn.naive_bayes import GaussianNB

    from sklearn.neural_network import (
        MLPClassifier,
        MLPRegressor,
    )

    models = {

        "logistic_regression": lambda: LogisticRegression(
            max_iter=1000,
            random_state=random_state
        ),

        "decision_tree_clf": lambda: DecisionTreeClassifier(
            random_state=random_state
        ),

        "random_forest_clf": lambda: RandomForestClassifier(
            n_estimators=100,
            random_state=random_state
        ),

        "gradient_boosting_clf": lambda: GradientBoostingClassifier(
            n_estimators=100,
            random_state=random_state
        ),

        "svm_clf": lambda: SVC(
            probability=True,
            random_state=random_state
        ),

        "knn_clf": lambda: KNeighborsClassifier(),

        "naive_bayes": lambda: GaussianNB(),

        "mlp_clf": lambda: MLPClassifier(
            max_iter=500,
            random_state=random_state
        ),

        "linear_regression": lambda: LinearRegression(),

        "decision_tree_reg": lambda: DecisionTreeRegressor(
            random_state=random_state
        ),

        "random_forest_reg": lambda: RandomForestRegressor(
            n_estimators=100,
            random_state=random_state
        ),

        "gradient_boosting_reg": lambda: GradientBoostingRegressor(
            n_estimators=100,
            random_state=random_state
        ),

        "svr": lambda: SVR(),

        "knn_reg": lambda: KNeighborsRegressor(),

        "mlp_reg": lambda: MLPRegressor(
            max_iter=500,
            random_state=random_state
        ),
    }

    if model_key not in models:
        raise ValueError(f"Unknown model: {model_key}")

    return models[model_key]()


# ============================================================
# CHART DATA HELPER
# ============================================================

def _sample_chart_data(values, max_points=100):

    values = np.asarray(values)

    if len(values) <= max_points:
        return values.tolist()

    indices = np.linspace(
        0,
        len(values) - 1,
        max_points,
        dtype=int
    )

    return values[indices].tolist()


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

def _evaluate_classification(
    model,
    X_test,
    y_test,
    class_labels=None
):

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
        classification_report,
        roc_curve,
        auc,
    )

    from sklearn.preprocessing import label_binarize

    y_test = np.asarray(y_test)

    y_pred = model.predict(X_test)

    classes = np.asarray(model.classes_)

    n_classes = len(classes)

    if n_classes < 2:
        raise ValueError("The model must contain at least two classes.")

    average = "binary" if n_classes == 2 else "weighted"

    metrics = {

        "accuracy": round(
            float(accuracy_score(y_test, y_pred)), 4
        ),

        "precision": round(
            float(precision_score(
                y_test,
                y_pred,
                average=average,
                zero_division=0
            )), 4
        ),

        "recall": round(
            float(recall_score(
                y_test,
                y_pred,
                average=average,
                zero_division=0
            )), 4
        ),

        "f1_score": round(
            float(f1_score(
                y_test,
                y_pred,
                average=average,
                zero_division=0
            )), 4
        ),
    }

    # --------------------------------------------------------
    # CLASS LABELS
    # --------------------------------------------------------

    if class_labels is not None:

        if len(class_labels) != n_classes:
            raise ValueError(
                "Class labels do not match model classes."
            )

        display_labels = [
            str(v) for v in class_labels
        ]

    else:

        display_labels = [
            str(v) for v in classes
        ]

    metrics["class_labels"] = display_labels

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    metrics["confusion_matrix"] = confusion_matrix(
        y_test,
        y_pred,
        labels=classes
    ).tolist()

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        labels=classes,
        target_names=display_labels,
        output_dict=True,
        zero_division=0
    )

    metrics["classification_report"] = {}

    for key, value in report.items():

        if isinstance(value, dict):

            metrics["classification_report"][key] = {
                k: round(float(v), 4)
                for k, v in value.items()
            }

        else:

            metrics["classification_report"][key] = round(
                float(value), 4
            )

    # --------------------------------------------------------
    # ROC CURVE AND AUC
    # --------------------------------------------------------

    metrics["auc_roc"] = None
    metrics["roc_curve"] = None

    if not hasattr(model, "predict_proba"):
        return metrics

    try:

        probabilities = np.asarray(
            model.predict_proba(X_test)
        )

        # Ensure probability columns match model classes.

        if probabilities.ndim != 2:
            return metrics

        if probabilities.shape[1] != n_classes:
            return metrics

        if n_classes == 2:

            # ------------------------------------------------
            # BINARY CLASSIFICATION
            # ------------------------------------------------

            positive_class = classes[1]

            y_binary = (
                y_test == positive_class
            ).astype(int)

            positive_probabilities = probabilities[:, 1]

            if len(np.unique(y_binary)) < 2:
                return metrics

            fpr, tpr, thresholds = roc_curve(
                y_binary,
                positive_probabilities
            )

            binary_auc = roc_auc_score(
                y_binary,
                positive_probabilities
            )

            metrics["auc_roc"] = round(
                float(binary_auc), 4
            )

            metrics["roc_curve"] = {

                "fpr": _sample_chart_data(fpr),

                "tpr": _sample_chart_data(tpr),

                "thresholds": _sample_chart_data(thresholds),

                "auc": round(float(binary_auc), 4),
            }

        else:

            # ------------------------------------------------
            # MULTICLASS CLASSIFICATION - ONE VS REST
            # ------------------------------------------------

            y_binary = label_binarize(
                y_test,
                classes=classes
            )

            roc_data = {}

            valid_auc_scores = []
            valid_class_weights = []

            for i, class_value in enumerate(classes):

                class_name = display_labels[i]

                y_class = y_binary[:, i]

                # ROC requires positive and negative samples.

                if len(np.unique(y_class)) < 2:
                    continue

                class_probabilities = probabilities[:, i]

                fpr, tpr, thresholds = roc_curve(
                    y_class,
                    class_probabilities
                )

                class_auc = auc(
                    fpr,
                    tpr
                )

                roc_data[class_name] = {

                    "fpr": _sample_chart_data(fpr),

                    "tpr": _sample_chart_data(tpr),

                    "thresholds": _sample_chart_data(thresholds),

                    "auc": round(float(class_auc), 4),
                }

                valid_auc_scores.append(
                    float(class_auc)
                )

                valid_class_weights.append(
                    int(np.sum(y_class))
                )

            metrics["roc_curve"] = roc_data

            # Multiclass global weighted AUC.

            if len(valid_auc_scores) == n_classes:

                global_auc = roc_auc_score(
                    y_binary,
                    probabilities,
                    average="weighted"
                )

                metrics["auc_roc"] = round(
                    float(global_auc), 4
                )

            elif valid_auc_scores:

                # Fallback when some classes cannot be evaluated.

                weighted_auc = np.average(
                    valid_auc_scores,
                    weights=valid_class_weights
                )

                metrics["auc_roc"] = round(
                    float(weighted_auc), 4
                )

    except (
        ValueError,
        IndexError,
        AttributeError,
        TypeError
    ):

        traceback.print_exc()

        metrics["auc_roc"] = None
        metrics["roc_curve"] = None

    return metrics


# ============================================================
# REGRESSION METRICS
# ============================================================

def _evaluate_regression(model, X_test, y_test):

    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )

    y_pred = np.asarray(
        model.predict(X_test)
    )

    y_actual = np.asarray(y_test)

    residuals = y_actual - y_pred

    mse = mean_squared_error(
        y_actual,
        y_pred
    )

    r2 = (
        float(r2_score(y_actual, y_pred))
        if len(y_actual) >= 2
        else None
    )

    metrics = {

        "mae": round(
            float(mean_absolute_error(y_actual, y_pred)), 4
        ),

        "mse": round(float(mse), 4),

        "rmse": round(float(np.sqrt(mse)), 4),

        "r2_score": (
            round(r2, 4)
            if r2 is not None
            else None
        ),
    }

    sample_size = min(200, len(y_actual))

    rng = np.random.default_rng(42)

    indices = np.sort(
        rng.choice(
            len(y_actual),
            sample_size,
            replace=False
        )
    )

    actual_sample = y_actual[indices]
    predicted_sample = y_pred[indices]
    residual_sample = residuals[indices]

    metrics["actual_vs_predicted"] = {

        "actual": [
            round(float(v), 4)
            for v in actual_sample
        ],

        "predicted": [
            round(float(v), 4)
            for v in predicted_sample
        ],
    }

    metrics["residuals"] = {

        "predicted": [
            round(float(v), 4)
            for v in predicted_sample
        ],

        "residual": [
            round(float(v), 4)
            for v in residual_sample
        ],
    }

    return metrics


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def _get_feature_importance(
    model,
    feature_names,
    X_test=None,
    y_test=None,
    task_type="classification"
):

    importances = None

    try:

        if hasattr(model, "feature_importances_"):

            importances = model.feature_importances_

        elif hasattr(model, "coef_"):

            coef = np.asarray(model.coef_)

            if coef.ndim > 1:
                importances = np.abs(coef).mean(axis=0)
            else:
                importances = np.abs(coef)

        elif X_test is not None and y_test is not None:

            from sklearn.inspection import permutation_importance

            scoring = (
                "accuracy"
                if task_type == "classification"
                else "r2"
            )

            result = permutation_importance(
                model,
                X_test,
                y_test,
                n_repeats=5,
                random_state=42,
                scoring=scoring,
                n_jobs=-1
            )

            importances = np.maximum(
                result.importances_mean,
                0
            )

    except Exception:

        traceback.print_exc()
        return None

    if importances is None:
        return None

    importances = np.asarray(importances).flatten()

    if len(importances) != len(feature_names):
        return None

    pairs = sorted(
        zip(
            feature_names,
            [round(float(v), 4) for v in importances]
        ),
        key=lambda item: item[1],
        reverse=True
    )

    return {

        "features": [
            item[0] for item in pairs
        ],

        "importances": [
            item[1] for item in pairs
        ],
    }


# ============================================================
# STRATIFIED TRAIN / TEST SPLIT
# ============================================================

def _classification_train_test_split(
    X,
    y,
    test_size,
    random_state
):

    """
    Ensure every class is represented in both training
    and testing sets.

    Raises a clear error when this is mathematically impossible.
    """

    from sklearn.model_selection import train_test_split

    y = pd.Series(y)

    class_counts = y.value_counts()

    n_classes = len(class_counts)

    if n_classes < 2:
        raise ValueError(
            "Classification requires at least two classes."
        )

    # Every class must contain at least two observations.

    insufficient_classes = class_counts[
        class_counts < 2
    ]

    if not insufficient_classes.empty:

        raise ValueError(
            "Some classes contain fewer than 2 samples. "
            "Each class needs at least 2 samples to appear "
            "in both training and testing sets. "
            f"Problematic classes: "
            f"{insufficient_classes.index.tolist()}"
        )

    n_samples = len(y)

    n_test = int(np.ceil(n_samples * test_size))
    n_train = n_samples - n_test

    if n_test < n_classes:

        raise ValueError(
            f"Test set has only {n_test} samples, but there "
            f"are {n_classes} classes. Increase test size "
            "or add more data."
        )

    if n_train < n_classes:

        raise ValueError(
            f"Training set has only {n_train} samples, but "
            f"there are {n_classes} classes. Reduce test size "
            "or add more data."
        )

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=test_size,

        random_state=random_state,

        stratify=y
    )

    # Final validation.

    expected_classes = set(y.unique())

    if (
        set(y_train.unique()) != expected_classes
        or set(y_test.unique()) != expected_classes
    ):

        raise ValueError(
            "Unable to preserve all classes in both datasets."
        )

    return X_train, X_test, y_train, y_test


# ============================================================
# MAIN MACHINE LEARNING PAGE
# ============================================================

@ml_bp.route("/", methods=["GET"])
def ml_page():

    datasets = Dataset.query.order_by(
        Dataset.updated_at.desc()
    ).all()

    dataset_id = request.args.get(
        "dataset_id",
        type=int
    )

    dataset = None
    columns = []
    column_types = {}

    if dataset_id:

        dataset = Dataset.query.get_or_404(dataset_id)

    elif datasets:

        dataset = datasets[0]

    if dataset:

        try:

            df = _load_dataframe(dataset)

            columns = df.columns.tolist()

            column_types = {
                col: str(df[col].dtype)
                for col in columns
            }

        except Exception as exc:

            flash(
                f"Error loading dataset: {exc}",
                "error"
            )

    clf_models, reg_models = _get_available_models()

    ml_results = None

    try:

        ml_results = _load_ml_results()

    except (ValueError, TypeError, OSError, json.JSONDecodeError):

        flash(
            "Unable to load previous training results.",
            "warning"
        )

    return render_template(
        "ml_training.html",

        datasets=datasets,
        dataset=dataset,

        columns=columns,
        column_types=json.dumps(column_types),

        classification_models=clf_models,
        regression_models=reg_models,

        ml_results=ml_results,
    )


# ============================================================
# AUTOMATIC TASK DETECTION
# ============================================================

@ml_bp.route(
    "/detect-task/<int:dataset_id>",
    methods=["POST"]
)
def detect_task_type(dataset_id):

    dataset = Dataset.query.get_or_404(dataset_id)

    data = request.get_json(silent=True) or {}

    target = data.get("target_column", "")

    if not target:

        return jsonify({
            "error": "No target column specified"
        }), 400

    try:

        df = _load_dataframe(dataset)

        if target not in df.columns:

            return jsonify({
                "error": "Target column not found"
            }), 400

        col = df[target]

        nunique = col.nunique()

        if (
            pd.api.types.is_object_dtype(col)
            or isinstance(col.dtype, pd.CategoricalDtype)
            or nunique <= 20
        ):

            task_type = "classification"

        else:

            task_type = "regression"

        distribution = None

        if task_type == "classification":

            vc = col.value_counts()

            distribution = {

                "labels": vc.index.astype(str).tolist(),

                "values": vc.values.tolist(),
            }

        return jsonify({

            "task_type": task_type,

            "unique_values": int(nunique),

            "dtype": str(col.dtype),

            "distribution": distribution,
        })

    except Exception as exc:

        return jsonify({
            "error": str(exc)
        }), 500


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

@ml_bp.route(
    "/check-imbalance/<int:dataset_id>",
    methods=["POST"]
)
def check_imbalance(dataset_id):

    dataset = Dataset.query.get_or_404(dataset_id)

    data = request.get_json(silent=True) or {}

    target = data.get("target_column", "")

    if not target:

        return jsonify({
            "error": "No target column specified"
        }), 400

    try:

        df = _load_dataframe(dataset)

        if target not in df.columns:

            return jsonify({
                "error": "Target column not found"
            }), 400

        vc = df[target].value_counts()

        total = int(vc.sum())

        if total == 0:
            raise ValueError("No valid target values.")

        ratio = (
            round(float(vc.max() / vc.min()), 2)
            if vc.min() > 0
            else None
        )

        return jsonify({

            "distribution": {

                "labels": vc.index.astype(str).tolist(),

                "values": vc.values.tolist(),

                "percentages": [
                    round(v / total * 100, 1)
                    for v in vc.values.tolist()
                ],
            },

            "imbalance_ratio": ratio,

            "is_imbalanced": (
                ratio is not None and ratio > 1.5
            ),

            "total_samples": total,
        })

    except Exception as exc:

        return jsonify({
            "error": str(exc)
        }), 500


# ============================================================
# MODEL TRAINING
# ============================================================

@ml_bp.route(
    "/train/<int:dataset_id>",
    methods=["POST"]
)
def train_models(dataset_id):

    dataset = Dataset.query.get_or_404(dataset_id)

    try:

        # ----------------------------------------------------
        # FORM DATA
        # ----------------------------------------------------

        target_column = request.form.get(
            "target_column",
            ""
        )

        feature_columns = request.form.getlist(
            "feature_columns"
        )

        task_type = request.form.get(
            "task_type",
            "classification"
        )

        selected_models = request.form.getlist(
            "selected_models"
        )

        test_size = (
            float(request.form.get("test_size", 20)) / 100
        )

        random_state = int(
            request.form.get("random_state", 42)
        )

        balancing_method = request.form.get(
            "balancing_method",
            "none"
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if not target_column:
            raise ValueError("Please select a target column.")

        if not feature_columns:
            raise ValueError("Please select feature columns.")

        if not selected_models:
            raise ValueError("Please select at least one model.")

        if task_type not in (
            "classification",
            "regression"
        ):
            raise ValueError("Invalid task type.")

        if not 0.05 <= test_size <= 0.5:
            raise ValueError(
                "Test size must be between 5% and 50%."
            )

        if target_column in feature_columns:
            raise ValueError("Target cannot be a feature.")

        clf_models, reg_models = _get_available_models()

        allowed_models = (
            clf_models
            if task_type == "classification"
            else reg_models
        )

        if any(
            key not in allowed_models
            for key in selected_models
        ):

            raise ValueError("Invalid algorithm selection.")

        if task_type == "regression" and balancing_method != "none":

            raise ValueError(
                "Class balancing is only available for classification."
            )

        # ----------------------------------------------------
        # LOAD DATA
        # ----------------------------------------------------

        df = _load_dataframe(dataset)

        if target_column not in df.columns:
            raise ValueError("Target column does not exist.")

        if not set(feature_columns).issubset(df.columns):
            raise ValueError("Invalid feature columns.")

        df = df[
            feature_columns + [target_column]
        ].copy()

        df = df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        df = df.dropna(
            subset=[target_column]
        )

        if len(df) < 5:
            raise ValueError(
                "Dataset has too few valid rows."
            )

        X = df[feature_columns].copy()

        y = df[target_column].copy()

        # ----------------------------------------------------
        # TARGET PREPARATION
        # ----------------------------------------------------

        target_classes = None

        if task_type == "classification":

            if y.nunique() < 2:

                raise ValueError(
                    "Target must contain at least two classes."
                )

            from sklearn.preprocessing import LabelEncoder

            target_encoder = LabelEncoder()

            y = pd.Series(
                target_encoder.fit_transform(
                    y.astype(str)
                ),
                index=y.index,
                name=target_column
            )

            target_classes = [
                str(value)
                for value in target_encoder.classes_
            ]

        else:

            y = pd.to_numeric(
                y,
                errors="coerce"
            )

            valid = y.notna()

            X = X.loc[valid]
            y = y.loc[valid]

            if len(y) < 5:

                raise ValueError(
                    "Not enough valid target values."
                )

        # ----------------------------------------------------
        # PREPROCESSING
        # ----------------------------------------------------

        from sklearn.impute import SimpleImputer

        from sklearn.preprocessing import (
            StandardScaler,
            OneHotEncoder,
        )

        from sklearn.compose import ColumnTransformer

        from sklearn.pipeline import Pipeline

        from sklearn.model_selection import train_test_split

        numeric_columns = X.select_dtypes(
            include=np.number
        ).columns.tolist()

        categorical_columns = [
            col
            for col in X.columns
            if col not in numeric_columns
        ]

        transformers = []

        if numeric_columns:

            numeric_pipeline = Pipeline([

                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                ),

                (
                    "scaler",
                    StandardScaler()
                ),
            ])

            transformers.append((
                "numeric",
                numeric_pipeline,
                numeric_columns
            ))

        if categorical_columns:

            categorical_pipeline = Pipeline([

                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),

                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                ),
            ])

            transformers.append((
                "categorical",
                categorical_pipeline,
                categorical_columns
            ))

        if not transformers:
            raise ValueError("No usable feature columns.")

        preprocessor = ColumnTransformer(
            transformers=transformers
        )

        # ----------------------------------------------------
        # TRAIN / TEST SPLIT
        # ----------------------------------------------------

        if task_type == "classification":

            X_train, X_test, y_train, y_test = (
                _classification_train_test_split(
                    X,
                    y,
                    test_size,
                    random_state
                )
            )

        else:

            X_train, X_test, y_train, y_test = train_test_split(

                X,
                y,

                test_size=test_size,

                random_state=random_state
            )

        # ----------------------------------------------------
        # FIT PREPROCESSOR
        # ----------------------------------------------------

        X_train_processed = preprocessor.fit_transform(
            X_train
        )

        X_test_processed = preprocessor.transform(
            X_test
        )

        # ----------------------------------------------------
        # CLASS BALANCING
        # ----------------------------------------------------

        balancing_info = None

        if (
            task_type == "classification"
            and balancing_method != "none"
        ):

            before_dist = (
                pd.Series(y_train)
                .value_counts()
                .to_dict()
            )

            if balancing_method == "smote":

                from imblearn.over_sampling import SMOTE

                min_class_count = min(before_dist.values())

                if min_class_count < 2:
                    raise ValueError(
                        "SMOTE requires at least two samples "
                        "in each class in the training set."
                    )

                sampler = SMOTE(
                    random_state=random_state,
                    k_neighbors=min(
                        5,
                        min_class_count - 1
                    )
                )

            elif balancing_method == "smote_tomek":

                from imblearn.combine import SMOTETomek
                from imblearn.over_sampling import SMOTE

                min_class_count = min(before_dist.values())

                if min_class_count < 2:
                    raise ValueError(
                        "SMOTE-Tomek requires at least two "
                        "samples in each class."
                    )

                sampler = SMOTETomek(
                    random_state=random_state,
                    smote=SMOTE(
                        random_state=random_state,
                        k_neighbors=min(
                            5,
                            min_class_count - 1
                        )
                    )
                )

            elif balancing_method == "adasyn":

                from imblearn.over_sampling import ADASYN

                min_class_count = min(before_dist.values())

                if min_class_count < 2:
                    raise ValueError(
                        "ADASYN requires at least two "
                        "samples in each class."
                    )

                sampler = ADASYN(
                    random_state=random_state,
                    n_neighbors=min(
                        5,
                        min_class_count - 1
                    )
                )

            elif balancing_method == "random_over":

                from imblearn.over_sampling import RandomOverSampler

                sampler = RandomOverSampler(
                    random_state=random_state
                )

            elif balancing_method == "random_under":

                from imblearn.under_sampling import RandomUnderSampler

                sampler = RandomUnderSampler(
                    random_state=random_state
                )

            else:

                raise ValueError(
                    "Invalid balancing method."
                )

            X_train_processed, y_train = sampler.fit_resample(
                X_train_processed,
                y_train
            )

            after_dist = pd.Series(
                y_train
            ).value_counts()

            balancing_info = {

                "method": balancing_method,

                "before": {
                    str(k): int(v)
                    for k, v in before_dist.items()
                },

                "after": {
                    str(k): int(v)
                    for k, v in after_dist.items()
                },
            }

        # ----------------------------------------------------
        # FEATURE NAMES
        # ----------------------------------------------------

        try:

            feature_names = (
                preprocessor
                .get_feature_names_out()
                .tolist()
            )

        except Exception:

            feature_names = [
                f"Feature_{i}"
                for i in range(
                    X_train_processed.shape[1]
                )
            ]

        # ----------------------------------------------------
        # MODEL STORAGE
        # ----------------------------------------------------

        models_folder = current_app.config.get(
            "MODELS_FOLDER"
        )

        if not models_folder:

            raise ValueError(
                "MODELS_FOLDER is not configured."
            )

        os.makedirs(
            models_folder,
            exist_ok=True
        )

        # ----------------------------------------------------
        # TRAIN SELECTED MODELS
        # ----------------------------------------------------

        all_results = []

        for model_key in selected_models:

            try:

                model = _build_model(
                    model_key,
                    random_state
                )

                start_time = time.time()

                model.fit(
                    X_train_processed,
                    y_train
                )

                training_time = round(
                    time.time() - start_time,
                    3
                )

                # EVALUATION

                if task_type == "classification":

                    metrics = _evaluate_classification(
                        model,
                        X_test_processed,
                        y_test,
                        class_labels=target_classes
                    )

                else:

                    metrics = _evaluate_regression(
                        model,
                        X_test_processed,
                        y_test
                    )

                # FEATURE IMPORTANCE

                feature_importance = _get_feature_importance(
                    model,
                    feature_names,
                    X_test=X_test_processed,
                    y_test=y_test,
                    task_type=task_type
                )

                # SAVE COMPLETE PIPELINE

                complete_model = Pipeline([

                    (
                        "preprocessor",
                        copy.deepcopy(preprocessor)
                    ),

                    (
                        "model",
                        model
                    ),
                ])

                timestamp = datetime.now().strftime(
                    "%Y%m%d_%H%M%S_%f"
                )

                model_filename = (
                    f"{model_key}_{dataset_id}_{timestamp}.joblib"
                )

                model_path = os.path.join(
                    models_folder,
                    model_filename
                )

                joblib.dump(
                    complete_model,
                    model_path
                )

                result = {

                    "model_key": model_key,

                    "model_name": allowed_models[
                        model_key
                    ]["name"],

                    "task_type": task_type,

                    "metrics": metrics,

                    "feature_importance": feature_importance,

                    "training_time": training_time,

                    "model_filename": model_filename,
                }

                all_results.append(result)

            except Exception as exc:

                flash(
                    f"Error training {model_key}: {exc}",
                    "warning"
                )

                traceback.print_exc()

        if not all_results:

            raise ValueError(
                "No models were successfully trained."
            )

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        if task_type == "classification":

            best = max(
                all_results,

                key=lambda result: (
                    result["metrics"].get(
                        "f1_score",
                        0
                    )
                )
            )

            primary_metric = "f1_score"

        else:

            valid_regression_results = [
                result
                for result in all_results
                if result["metrics"].get("r2_score") is not None
            ]

            if valid_regression_results:

                best = max(
                    valid_regression_results,
                    key=lambda result: result["metrics"]["r2_score"]
                )

                primary_metric = "r2_score"

            else:

                best = min(
                    all_results,
                    key=lambda result: result["metrics"]["rmse"]
                )

                primary_metric = "rmse"

        # ----------------------------------------------------
        # SAVE RESULTS OUTSIDE SESSION
        # ----------------------------------------------------

        results_data = {

            "dataset_id": dataset_id,

            "dataset_name": dataset.name,

            "target_column": target_column,

            "feature_columns": feature_columns,

            "task_type": task_type,

            "test_size": test_size,

            "random_state": random_state,

            "balancing": balancing_info,

            "results": all_results,

            "best_model": best["model_key"],

            "best_model_name": best["model_name"],

            "primary_metric": primary_metric,

            "target_classes": target_classes,

            "trained_at": datetime.now().isoformat(),
        }

        _save_ml_results(results_data)

        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        flash(
            f'Successfully trained {len(all_results)} model(s)! '
            f'Best: {best["model_name"]} '
            f'({primary_metric}: '
            f'{best["metrics"][primary_metric]})',
            "success"
        )

        return redirect(
            url_for("evaluation.evaluation_page")
        )

    except Exception as exc:

        flash(
            f"Training error: {exc}",
            "error"
        )

        traceback.print_exc()

        return redirect(
            url_for(
                "ml.ml_page",
                dataset_id=dataset_id
            )
        )
