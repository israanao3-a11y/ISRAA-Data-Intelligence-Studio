
"""
ISRAA - Advanced Reports Module.
Displays ML reports and exports results to CSV.
"""

import csv
import io
import json

from flask import (
    Blueprint,
    render_template,
    session,
    redirect,
    url_for,
    flash,
    make_response
)

reports_bp = Blueprint('reports', __name__)


def get_report():
    saved_report = session.get('ml_results')

    if not saved_report:
        return None

    try:
        if isinstance(saved_report, str):
            report = json.loads(saved_report)
        elif isinstance(saved_report, dict):
            report = saved_report
        else:
            return None

        if not isinstance(report, dict):
            return None

        return report

    except (ValueError, TypeError):
        return None


@reports_bp.route('/')
def reports_page():
    report = get_report()

    if report:
        results = report.get('results', [])

        if not isinstance(results, list):
            results = []

        report['results'] = results
        report['successful_models'] = len(results)

        report['best_result'] = next(
            (
                result for result in results
                if result.get('model_key') == report.get('best_model')
            ),
            None
        )

    return render_template(
        'reports.html',
        report=report
    )


@reports_bp.route('/export/csv')
def export_csv():
    report = get_report()

    if not report or not report.get('results'):
        flash('No report available for export.', 'warning')
        return redirect(url_for('reports.reports_page'))

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        'Model',
        'Training Time',
        'Accuracy',
        'Precision',
        'Recall',
        'F1 Score',
        'AUC ROC',
        'MAE',
        'MSE',
        'RMSE',
        'R2 Score'
    ])

    for result in report.get('results', []):
        metrics = result.get('metrics', {})

        writer.writerow([
            result.get('model_name', ''),
            result.get('training_time', ''),
            metrics.get('accuracy', ''),
            metrics.get('precision', ''),
            metrics.get('recall', ''),
            metrics.get('f1_score', ''),
            metrics.get('auc_roc', ''),
            metrics.get('mae', ''),
            metrics.get('mse', ''),
            metrics.get('rmse', ''),
            metrics.get('r2_score', '')
        ])

    response = make_response('\ufeff' + output.getvalue())

    response.headers['Content-Type'] = 'text/csv; charset=utf-8'
    response.headers['Content-Disposition'] = (
        'attachment; filename=ISRAA_ML_Report.csv'
    )

    return response
