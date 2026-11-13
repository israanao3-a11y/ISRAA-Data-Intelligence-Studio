"""
Data quality analysis service.
Detects missing values, duplicates, outliers, inconsistencies,
and computes an overall quality score with recommendations.
"""
import pandas as pd
import numpy as np


def analyze_missing_values(df: pd.DataFrame) -> dict:
    """Analyze missing values across all columns.

    Args:
        df: Input DataFrame.

    Returns:
        Dict with per-column missing counts/percentages, overall stats,
        and visualization data for Plotly.
    """
    try:
        total_cells = int(df.shape[0] * df.shape[1])
        missing_per_col = df.isnull().sum()
        total_missing = int(missing_per_col.sum())

        columns_info = {}
        for col in df.columns:
            count = int(missing_per_col[col])
            columns_info[col] = {
                'count': count,
                'percentage': round(count / len(df) * 100, 2) if len(df) > 0 else 0.0,
                'dtype': str(df[col].dtype),
            }

        # Sort by missing count descending for visualization
        sorted_cols = sorted(columns_info.items(), key=lambda x: x[1]['count'], reverse=True)

        chart_data = {
            'labels': [item[0] for item in sorted_cols if item[1]['count'] > 0],
            'values': [item[1]['count'] for item in sorted_cols if item[1]['count'] > 0],
            'percentages': [item[1]['percentage'] for item in sorted_cols if item[1]['count'] > 0],
        }

        return {
            'total_missing': total_missing,
            'total_cells': total_cells,
            'overall_percentage': round(total_missing / total_cells * 100, 2) if total_cells > 0 else 0.0,
            'columns_with_missing': sum(1 for v in columns_info.values() if v['count'] > 0),
            'total_columns': len(df.columns),
            'columns': columns_info,
            'chart_data': chart_data,
        }

    except Exception as e:
        return {'error': str(e), 'total_missing': 0, 'columns': {}}


def detect_duplicates(df: pd.DataFrame) -> dict:
    """Detect duplicate rows in the DataFrame.

    Args:
        df: Input DataFrame.

    Returns:
        Dict with total duplicates, percentage, and first 20 duplicate rows.
    """
    try:
        duplicated_mask = df.duplicated(keep='first')
        n_duplicates = int(duplicated_mask.sum())

        # Get first 20 duplicate rows for preview
        duplicate_rows = df[duplicated_mask].head(20)
        duplicate_preview = duplicate_rows.to_dict(orient='records')

        # Convert numpy types in preview
        for row in duplicate_preview:
            for key, val in row.items():
                if isinstance(val, (np.integer,)):
                    row[key] = int(val)
                elif isinstance(val, (np.floating,)):
                    row[key] = float(val) if not np.isnan(val) else None
                elif isinstance(val, (np.bool_,)):
                    row[key] = bool(val)
                elif pd.isna(val):
                    row[key] = None

        return {
            'total_duplicates': n_duplicates,
            'total_rows': int(len(df)),
            'percentage': round(n_duplicates / len(df) * 100, 2) if len(df) > 0 else 0.0,
            'has_duplicates': n_duplicates > 0,
            'duplicate_indices': [int(i) for i in duplicate_rows.index.tolist()],
            'duplicate_preview': duplicate_preview,
        }

    except Exception as e:
        return {'error': str(e), 'total_duplicates': 0, 'has_duplicates': False}


def detect_outliers(df: pd.DataFrame, method: str = 'iqr') -> dict:
    """Detect outliers in numeric columns.

    Args:
        df: Input DataFrame.
        method: Detection method — 'iqr' (Interquartile Range) or 'zscore'.

    Returns:
        Dict with per-column outlier counts, percentages, bounds, and indices.
    """
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        results = {}
        total_outliers = 0

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            if method == 'iqr':
                q1 = float(series.quantile(0.25))
                q3 = float(series.quantile(0.75))
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outlier_mask = (series < lower) | (series > upper)
            elif method == 'zscore':
                mean = float(series.mean())
                std = float(series.std())
                if std == 0:
                    results[col] = {
                        'count': 0, 'percentage': 0.0,
                        'lower_bound': mean, 'upper_bound': mean,
                        'indices': [],
                    }
                    continue
                z_scores = (series - mean) / std
                lower = mean - 3 * std
                upper = mean + 3 * std
                outlier_mask = z_scores.abs() > 3
            else:
                raise ValueError(f"Unsupported method: {method}. Use 'iqr' or 'zscore'.")

            count = int(outlier_mask.sum())
            total_outliers += count

            results[col] = {
                'count': count,
                'percentage': round(count / len(series) * 100, 2) if len(series) > 0 else 0.0,
                'lower_bound': round(lower, 4),
                'upper_bound': round(upper, 4),
                'indices': [int(i) for i in series[outlier_mask].index.tolist()[:50]],
            }

        return {
            'method': method,
            'total_outliers': total_outliers,
            'columns_analyzed': len(numeric_cols),
            'columns': results,
        }

    except Exception as e:
        return {'error': str(e), 'total_outliers': 0, 'columns': {}}


def detect_inconsistencies(df: pd.DataFrame) -> dict:
    """Detect data inconsistencies: mixed types, whitespace issues, empty strings.

    Args:
        df: Input DataFrame.

    Returns:
        Dict with per-column inconsistency details.
    """
    try:
        results = {}

        for col in df.columns:
            issues = []
            series = df[col].dropna()

            if len(series) == 0:
                continue

            # Check mixed types (only for object columns)
            if series.dtype == 'object':
                type_set = set(type(v).__name__ for v in series)
                if len(type_set) > 1:
                    issues.append({
                        'type': 'mixed_types',
                        'detail': f"Mixed types found: {', '.join(sorted(type_set))}",
                        'count': len(type_set),
                    })

                # Check leading/trailing whitespace
                str_series = series.astype(str)
                whitespace_mask = str_series != str_series.str.strip()
                ws_count = int(whitespace_mask.sum())
                if ws_count > 0:
                    issues.append({
                        'type': 'whitespace',
                        'detail': f"{ws_count} values with leading/trailing whitespace",
                        'count': ws_count,
                    })

                # Check empty strings
                empty_count = int((str_series.str.strip() == '').sum())
                if empty_count > 0:
                    issues.append({
                        'type': 'empty_strings',
                        'detail': f"{empty_count} empty string values",
                        'count': empty_count,
                    })

            if issues:
                results[col] = issues

        total_issues = sum(len(v) for v in results.values())

        return {
            'total_issues': total_issues,
            'columns_affected': len(results),
            'total_columns': len(df.columns),
            'columns': results,
        }

    except Exception as e:
        return {'error': str(e), 'total_issues': 0, 'columns': {}}


def calculate_quality_score(df: pd.DataFrame) -> dict:
    """Calculate an overall data quality score from 0 to 100.

    Scoring breakdown (weights):
        - Missing values: 40%
        - Duplicates: 20%
        - Outliers: 20%
        - Inconsistencies: 20%

    Args:
        df: Input DataFrame.

    Returns:
        Dict with overall score, component scores, grade, and breakdown.
    """
    try:
        # --- Missing values component (40%) ---
        total_cells = df.shape[0] * df.shape[1]
        total_missing = int(df.isnull().sum().sum())
        missing_pct = total_missing / total_cells * 100 if total_cells > 0 else 0
        # Score: 100 if 0% missing, 0 if >=50% missing
        missing_score = max(0, 100 - (missing_pct * 2))

        # --- Duplicates component (20%) ---
        n_duplicates = int(df.duplicated().sum())
        dup_pct = n_duplicates / len(df) * 100 if len(df) > 0 else 0
        dup_score = max(0, 100 - (dup_pct * 2))

        # --- Outliers component (20%) ---
        outlier_info = detect_outliers(df, method='iqr')
        total_numeric_values = sum(df[col].count() for col in df.select_dtypes(include=[np.number]).columns)
        total_outliers = outlier_info.get('total_outliers', 0)
        outlier_pct = total_outliers / total_numeric_values * 100 if total_numeric_values > 0 else 0
        outlier_score = max(0, 100 - (outlier_pct * 4))

        # --- Inconsistencies component (20%) ---
        inconsistency_info = detect_inconsistencies(df)
        total_issues = inconsistency_info.get('total_issues', 0)
        incon_pct = total_issues / len(df.columns) * 100 if len(df.columns) > 0 else 0
        incon_score = max(0, 100 - (incon_pct * 2))

        # Weighted overall score
        overall = (
            missing_score * 0.4 +
            dup_score * 0.2 +
            outlier_score * 0.2 +
            incon_score * 0.2
        )
        overall = round(min(100, max(0, overall)), 1)

        # Grade
        if overall >= 90:
            grade = 'Excellent'
        elif overall >= 70:
            grade = 'Good'
        elif overall >= 50:
            grade = 'Fair'
        else:
            grade = 'Poor'

        breakdown = {
            'missing': {
                'score': round(missing_score, 1),
                'weight': 40,
                'detail': f"{missing_pct:.1f}% missing values",
            },
            'duplicates': {
                'score': round(dup_score, 1),
                'weight': 20,
                'detail': f"{dup_pct:.1f}% duplicate rows",
            },
            'outliers': {
                'score': round(outlier_score, 1),
                'weight': 20,
                'detail': f"{outlier_pct:.1f}% outlier values",
            },
            'inconsistencies': {
                'score': round(incon_score, 1),
                'weight': 20,
                'detail': f"{total_issues} inconsistency issues",
            },
        }

        return {
            'score': overall,
            'grade': grade,
            'breakdown': breakdown,
            'total_missing': total_missing,
            'total_duplicates': n_duplicates,
            'total_outliers': total_outliers,
            'total_inconsistencies': total_issues,
        }

    except Exception as e:
        return {'score': 0, 'grade': 'Poor', 'error': str(e), 'breakdown': {}}


def get_recommendations(quality_report: dict) -> list:
    """Generate actionable recommendations based on quality analysis results.

    Args:
        quality_report: Dict from calculate_quality_score().

    Returns:
        List of recommendation strings.
    """
    recommendations = []

    try:
        breakdown = quality_report.get('breakdown', {})

        # Missing values recommendations
        missing = breakdown.get('missing', {})
        missing_score = missing.get('score', 100)
        if missing_score < 100:
            if missing_score < 50:
                recommendations.append(
                    "⚠️ High missing data detected. Consider dropping columns with >50% missing values, "
                    "then impute remaining missing values using mean/median (numeric) or mode (categorical)."
                )
            elif missing_score < 80:
                recommendations.append(
                    "📋 Moderate missing data detected. Use mean/median imputation for numeric columns "
                    "and mode imputation for categorical columns."
                )
            else:
                recommendations.append(
                    "✅ Low missing data. Consider imputing with appropriate methods or dropping affected rows."
                )

        # Duplicate recommendations
        dup = breakdown.get('duplicates', {})
        dup_score = dup.get('score', 100)
        if dup_score < 100:
            total_dups = quality_report.get('total_duplicates', 0)
            if dup_score < 70:
                recommendations.append(
                    f"⚠️ {total_dups} duplicate rows found. Review and remove duplicates to avoid model bias."
                )
            else:
                recommendations.append(
                    f"📋 {total_dups} duplicate rows found. Consider removing duplicates."
                )

        # Outlier recommendations
        outlier = breakdown.get('outliers', {})
        outlier_score = outlier.get('score', 100)
        if outlier_score < 100:
            total_out = quality_report.get('total_outliers', 0)
            if outlier_score < 60:
                recommendations.append(
                    f"⚠️ {total_out} outliers detected. Consider clipping (winsorizing) outliers "
                    "to IQR bounds or applying log transformation to reduce impact."
                )
            else:
                recommendations.append(
                    f"📋 {total_out} outliers detected. Inspect whether they are genuine or data errors."
                )

        # Inconsistency recommendations
        incon = breakdown.get('inconsistencies', {})
        incon_score = incon.get('score', 100)
        if incon_score < 100:
            total_incon = quality_report.get('total_inconsistencies', 0)
            recommendations.append(
                f"📋 {total_incon} data inconsistencies found (whitespace, mixed types, empty strings). "
                "Use text standardization to clean string columns."
            )

        # General recommendations based on overall score
        score = quality_report.get('score', 0)
        if score >= 90:
            recommendations.append(
                "🎉 Excellent data quality! Your dataset is ready for analysis and modeling."
            )
        elif score >= 70:
            recommendations.append(
                "👍 Good data quality. Address the issues above before building models for best results."
            )
        elif score < 50:
            recommendations.append(
                "🔧 Data quality needs significant improvement. Apply cleaning steps before proceeding."
            )

        if not recommendations:
            recommendations.append("✅ No quality issues detected. Your data looks clean!")

    except Exception:
        recommendations.append("Unable to generate recommendations. Please review data manually.")

    return recommendations
