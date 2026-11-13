
"""
Model Evaluation Routes
ISRAA - Data Intelligence Studio

Loads and displays machine learning evaluation results.
"""

import os
import json

from flask import (
    Blueprint,
    current_app,
    render_template,
    session,
    flash,
    redirect,
    url_for,
)


evaluation_bp = Blueprint("evaluation", __name__)


# ============================================================
# LOAD TRAINING RESULTS
# ============================================================

def _load_evaluation_results():

    """
    Load training results from JSON storage.
    Session contains only the JSON filename.
    Supports legacy session results for compatibility.
    """

    filename = session.get("ml_results_file")

    if filename:

        if not isinstance(filename, str):
            raise ValueError("Invalid results filename.")

        if os.path.basename(filename) != filename:
            raise ValueError("Invalid results filename.")

        if not filename.endswith(".json"):
            raise ValueError("Invalid results file format.")

        models_folder = current_app.config.get(
            "MODELS_FOLDER"
        )

        if not models_folder:
            raise ValueError("MODELS_FOLDER is not configured.")

        results_folder = os.path.join(
            models_folder,
            "training_results"
        )

        results_path = os.path.join(
            results_folder,
            filename
        )

        if not os.path.isfile(results_path):
            raise FileNotFoundError(
                "Training results file not found."
            )

        with open(
            results_path,
            "r",
            encoding="utf-8"
        ) as file:

            ml_results = json.load(file)

    else:

        # Backward compatibility with old sessions.

        saved_results = session.get("ml_results")

        if not saved_results:
            return None

        if isinstance(saved_results, str):

            ml_results = json.loads(saved_results)

        elif isinstance(saved_results, dict):

            ml_results = saved_results

        else:

            raise ValueError(
                "Invalid evaluation data format."
            )

    if not isinstance(ml_results, dict):

        raise ValueError(
            "Evaluation results must be a dictionary."
        )

    return ml_results


# ============================================================
# MAIN EVALUATION PAGE
# ============================================================

@evaluation_bp.route("/", methods=["GET"])
def evaluation_page():

    """Display machine learning evaluation results."""

    # --------------------------------------------------------
    # LOAD RESULTS
    # --------------------------------------------------------

    try:

        ml_results = _load_evaluation_results()

        if not ml_results:

            flash(
                "No training results available. Train a model first.",
                "warning"
            )

            return redirect(
                url_for("ml.ml_page")
            )

    except (
        ValueError,
        TypeError,
        OSError,
        json.JSONDecodeError
    ):

        flash(
            "Unable to load evaluation results. "
            "Please train the models again.",
            "error"
        )

        return redirect(
            url_for("ml.ml_page")
        )

    # --------------------------------------------------------
    # EXTRACT MODEL RESULTS
    # --------------------------------------------------------

    results = ml_results.get(
        "results",
        []
    )

    if not isinstance(results, list):
        results = []

    # Keep only valid result dictionaries.

    results = [
        result
        for result in results
        if isinstance(result, dict)
    ]

    # --------------------------------------------------------
    # VALIDATE ADDITIONAL DATA
    # --------------------------------------------------------

    target_classes = ml_results.get(
        "target_classes",
        []
    )

    if not isinstance(target_classes, list):
        target_classes = []

    feature_columns = ml_results.get(
        "feature_columns",
        []
    )

    if not isinstance(feature_columns, list):
        feature_columns = []

    balancing = ml_results.get(
        "balancing"
    )

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    best_model = ml_results.get(
        "best_model"
    )

    best_model_name = ml_results.get(
        "best_model_name"
    )

    # --------------------------------------------------------
    # DISPLAY EVALUATION PAGE
    # --------------------------------------------------------

    return render_template(

        "evaluation.html",

        ml_results=ml_results,

        results=results,

        best_model=best_model,

        best_model_name=best_model_name,

        task_type=ml_results.get(
            "task_type"
        ),

        dataset_name=ml_results.get(
            "dataset_name"
        ),

        target_column=ml_results.get(
            "target_column"
        ),

        target_classes=target_classes,

        feature_columns=feature_columns,

        trained_at=ml_results.get(
            "trained_at"
        ),

        balancing=balancing,

        primary_metric=ml_results.get(
            "primary_metric"
        ),

    )


# ============================================================
# CLEAR EVALUATION RESULTS
# ============================================================

@evaluation_bp.route("/clear", methods=["GET"])
def clear_evaluation():

    """Clear saved model evaluation results."""

    filename = session.get(
        "ml_results_file"
    )

    try:

        if filename:

            # Security validation.

            if (
                isinstance(filename, str)
                and os.path.basename(filename) == filename
                and filename.endswith(".json")
            ):

                models_folder = current_app.config.get(
                    "MODELS_FOLDER"
                )

                if models_folder:

                    results_path = os.path.join(
                        models_folder,
                        "training_results",
                        filename
                    )

                    if os.path.isfile(results_path):
                        os.remove(results_path)

        # Remove both new and legacy session entries.

        session.pop(
            "ml_results_file",
            None
        )

        session.pop(
            "ml_results",
            None
        )

        session.modified = True

        flash(
            "Evaluation results cleared successfully.",
            "success"
        )

    except OSError:

        flash(
            "Unable to delete the saved evaluation results.",
            "error"
        )

    return redirect(
        url_for("ml.ml_page")
    )
