import pandas as pd
import numpy as np
from scipy import stats
import regex as re
import altair as alt
from typing import List, Dict, Any, Optional
from utils.data_utils import dtype_split
from utils.stats_utils import compute_basic_stats
from utils.viz_utils import alt_histogram
import logging

logger = logging.getLogger(__name__)

class PreprocessingRecommendations:
    def __init__(self):
        """Initialize with PII patterns aligned with steps.py."""
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,4}|\d{3}[-.\s]\d{3}[-.\s]\d{4})\b',
            'credit_card': r'\b(?:\d[ -]*?){13,16}\b'
        }

    def analyze_dataset(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analyze the DataFrame and return a list of preprocessing recommendations.
        Returns recommendations sorted by priority.
        """
        try:
            if not isinstance(df, pd.DataFrame):
                logger.error("Input must be a pandas DataFrame")
                return []

            # Cache statistics to avoid redundant computation
            stats = compute_basic_stats(df)
            recommendations = []
            num_cols, cat_cols = dtype_split(df)

            # 1. Missing Data Analysis
            missing_by_col = stats.get("missing_by_col", {})
            if missing_by_col:
                max_missing_ratio = max(v / stats["shape"][0] for v in missing_by_col.values())
                if max_missing_ratio > 0.05:  # Dynamic threshold: 5% or higher
                    for col, missing_count in missing_by_col.items():
                        missing_ratio = missing_count / stats["shape"][0]
                        if missing_ratio > 0.05:
                            strategy = "mean" if col in num_cols else "mode"
                            recommendations.append({
                                'type': 'missing_data',
                                'severity': 'high' if missing_ratio > 0.3 else 'medium',
                                'suggestion': f"Impute missing values in '{col}' using {strategy} or drop rows/columns.",
                                'column': col,
                                'missing_count': missing_count,
                                'missing_ratio': round(missing_ratio, 2),
                                'priority': min(0.9, 0.7 + missing_ratio)
                            })

            # 2. Outlier Detection (IQR and Z-score)
            for col in num_cols:
                clean_series = df[col].dropna()
                if len(clean_series) < 2:
                    continue
                # IQR-based detection
                q1, q3 = clean_series.quantile([0.25, 0.75])
                iqr = q3 - q1
                iqr_outliers = clean_series[(clean_series < q1 - 1.5 * iqr) | (clean_series > q3 + 1.5 * iqr)]
                # Z-score-based detection
                z_scores = np.abs(stats.zscore(clean_series))
                z_outliers = clean_series[z_scores > 3]
                if len(iqr_outliers) > len(df) * 0.01 or len(z_outliers) > len(df) * 0.01:
                    recommendations.append({
                        'type': 'outliers',
                        'column': col,
                        'count': max(len(iqr_outliers), len(z_outliers)),
                        'suggestion': 'Handle outliers using IQR or z-score capping/removal, or apply log transformation.',
                        'priority': 0.7
                    })

            # 3. Bias Check (Entropy-based)
            for col in cat_cols:
                value_counts = df[col].value_counts(normalize=True, dropna=False)
                if len(value_counts) > 1:
                    entropy = -np.sum(value_counts * np.log2(value_counts + 1e-10))
                    max_entropy = np.log2(len(value_counts))
                    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 1.0
                    if normalized_entropy < 0.5:  # Low entropy indicates potential bias
                        recommendations.append({
                            'type': 'bias_risk',
                            'column': col,
                            'suggestion': 'Mitigate potential bias by oversampling minority classes or using robust algorithms.',
                            'priority': 0.8
                        })

            # 4. Duplicate Detection
            duplicate_rows = stats.get("duplicate_rows", 0)
            if duplicate_rows > max(1, len(df) * 0.005):  # Dynamic threshold
                recommendations.append({
                    'type': 'duplicates',
                    'severity': 'medium',
                    'suggestion': 'Remove duplicate rows to improve data quality.',
                    'count': duplicate_rows,
                    'priority': 0.6
                })

            # 5. Data Type Mismatch
            for col in df.columns:
                inferred_type = pd.api.types.infer_dtype(df[col], skipna=True)
                if inferred_type in ('mixed', 'string'):
                    if df[col].str.isnumeric().any():
                        recommendations.append({
                            'type': 'data_type_mismatch',
                            'column': col,
                            'suggestion': 'Convert to numeric type to correct mixed or string-based numeric data.',
                            'priority': 0.5
                        })
                    elif df[col].str.match(r'\d{4}-\d{2}-\d{2}').any():
                        recommendations.append({
                            'type': 'data_type_mismatch',
                            'column': col,
                            'suggestion': 'Standardize to datetime format to correct string-based dates.',
                            'priority': 0.5
                        })

            # 6. Skewness Analysis
            for col in num_cols:
                skewness = df[col].skew()
                if abs(skewness) > 1:
                    transform = 'log' if df[col].min() > 0 else 'square_root'
                    if transform == 'log' and 'scipy' in globals():
                        transform = 'boxcox'
                    recommendations.append({
                        'type': 'skewness',
                        'column': col,
                        'suggestion': f"Apply {transform} transformation to reduce skewness ({skewness:.2f}).",
                        'priority': 0.4
                    })

            # 7. Sensitive Data Detection
            for col in cat_cols:
                for pii_type, pattern in self.pii_patterns.items():
                    matches = df[col].astype(str).str.contains(pattern, regex=True, na=False).sum()
                    if matches > 0:
                        recommendations.append({
                            'type': 'sensitive_data',
                            'column': col,
                            'suggestion': f"Mask potential {pii_type} PII in '{col}' to protect sensitive information.",
                            'count': matches,
                            'priority': 0.95
                        })

            # 8. Auto-Pipeline Suggestion
            pipeline_suggestion = []
            # Step 1: Handle missing data
            for rec in [r for r in recommendations if r['type'] == 'missing_data']:
                strategy = 'mean' if rec['column'] in num_cols else 'mode'
                pipeline_suggestion.append({
                    'kind': 'impute',
                    'params': {'columns': [rec['column']], 'strategy': strategy}
                })
            # Step 2: Handle type mismatches
            for rec in [r for r in recommendations if r['type'] == 'data_type_mismatch']:
                if 'datetime' in rec['suggestion']:
                    pipeline_suggestion.append({
                        'kind': 'standardize_dates',
                        'params': {'columns': [rec['column']]}
                    })
                else:
                    pipeline_suggestion.append({
                        'kind': 'type_convert',
                        'params': {'column': rec['column'], 'type': 'numeric'}
                    })
            # Step 3: Remove duplicates
            if any(r['type'] == 'duplicates' for r in recommendations):
                pipeline_suggestion.append({
                    'kind': 'duplicates',
                    'params': {}
                })
            # Step 4: Handle outliers
            for rec in [r for r in recommendations if r['type'] == 'outliers']:
                pipeline_suggestion.append({
                    'kind': 'outliers',
                    'params': {'columns': [rec['column']], 'method': 'iqr'}
                })
            # Step 5: Handle skewness
            for rec in [r for r in recommendations if r['type'] == 'skewness']:
                pipeline_suggestion.append({
                    'kind': 'skewness_transform',
                    'params': {'column': rec['column'], 'transform': 'log' if 'log' in rec['suggestion'] else 'square_root'}
                })
            # Step 6: Handle bias
            for rec in [r for r in recommendations if r['type'] == 'bias_risk']:
                pipeline_suggestion.append({
                    'kind': 'rebalance',
                    'params': {'target': rec['column'], 'method': 'oversample'}
                })
            # Step 7: Handle PII
            for rec in [r for r in recommendations if r['type'] == 'sensitive_data']:
                pipeline_suggestion.append({
                    'kind': 'mask_pii',
                    'params': {'column': rec['column'], 'pii_types': ['email', 'phone', 'credit_card']}
                })
            if pipeline_suggestion:
                recommendations.append({
                    'type': 'auto_pipeline',
                    'suggestion': 'Apply the following preprocessing pipeline: ' + '; '.join(f"{step['kind']} ({step['params']})" for step in pipeline_suggestion),
                    'pipeline': pipeline_suggestion,
                    'priority': 0.85
                })

            # Sort by priority
            recommendations.sort(key=lambda x: x.get('priority', 0.5), reverse=True)
            return recommendations

        except Exception as e:
            logger.error(f"Error analyzing dataset: {e}")
            return []

    def visualize_recommendation(self, df: pd.DataFrame, recommendation: Dict[str, Any]) -> Optional[alt.Chart]:
        """
        Generate an Altair chart for a recommendation.
        Returns None if visualization is not applicable.
        """
        try:
            if recommendation['type'] == 'missing_data':
                data = pd.DataFrame({
                    'Column': [recommendation['column']],
                    'Missing_Ratio': [recommendation['missing_ratio']]
                })
                chart = alt.Chart(data).mark_bar().encode(
                    x=alt.X('Column:N', title='Column'),
                    y=alt.Y('Missing_Ratio:Q', title='Missing Ratio', scale=alt.Scale(domain=[0, 1])),
                    tooltip=['Column', alt.Tooltip('Missing_Ratio:Q', format='.2f')]
                ).properties(
                    title=f"Missing Data in {recommendation['column']}",
                    width=400,
                    height=300
                )
                return chart

            elif recommendation['type'] == 'outliers':
                col = recommendation['column']
                clean_series = df[col].dropna()
                if len(clean_series) < 2:
                    return None
                chart = alt_histogram(df, col, title=f"Outliers in {col} (Count: {recommendation['count']})")
                return chart

            elif recommendation['type'] == 'bias_risk':
                col = recommendation['column']
                value_counts = df[col].value_counts(normalize=True, dropna=False)
                data = pd.DataFrame({
                    'Category': value_counts.index,
                    'Proportion': value_counts.values
                })
                chart = alt.Chart(data).mark_bar().encode(
                    x=alt.X('Category:N', title=col),
                    y=alt.Y('Proportion:Q', title='Proportion', scale=alt.Scale(domain=[0, 1])),
                    color=alt.condition(
                        alt.datum.Proportion > 0.8,
                        alt.value('red'),
                        alt.value('steelblue')
                    ),
                    tooltip=['Category', alt.Tooltip('Proportion:Q', format='.2f')]
                ).properties(
                    title=f"Bias Risk in {col}",
                    width=400,
                    height=300
                )
                return chart

            elif recommendation['type'] == 'duplicates':
                data = pd.DataFrame({
                    'Status': ['Duplicates', 'Unique'],
                    'Count': [recommendation['count'], len(df) - recommendation['count']]
                })
                chart = alt.Chart(data).mark_bar().encode(
                    x=alt.X('Status:N', title='Status'),
                    y=alt.Y('Count:Q', title='Count'),
                    color=alt.condition(
                        alt.datum.Status == 'Duplicates',
                        alt.value('red'),
                        alt.value('steelblue')
                    ),
                    tooltip=['Status', 'Count']
                ).properties(
                    title=f"Duplicate Rows (Count: {recommendation['count']})",
                    width=400,
                    height=300
                )
                return chart

            elif recommendation['type'] == 'skewness':
                col = recommendation['column']
                chart = alt_histogram(df, col, title=f"Distribution of {col} (Skewness: {df[col].skew():.2f})")
                return chart

            elif recommendation['type'] == 'sensitive_data':
                col = recommendation['column']
                data = pd.DataFrame({
                    'PII_Type': ['Potential PII', 'Non-PII'],
                    'Count': [recommendation['count'], len(df[col].dropna()) - recommendation['count']]
                })
                chart = alt.Chart(data).mark_bar().encode(
                    x=alt.X('PII_Type:N', title='Type'),
                    y=alt.Y('Count:Q', title='Count'),
                    color=alt.condition(
                        alt.datum.PII_Type == 'Potential PII',
                        alt.value('red'),
                        alt.value('steelblue')
                    ),
                    tooltip=['PII_Type', 'Count']
                ).properties(
                    title=f"Sensitive Data in {col} (Count: {recommendation['count']})",
                    width=400,
                    height=300
                )
                return chart

            return None

        except Exception as e:
            logger.error(f"Error visualizing recommendation: {e}")
            return None

    def preview_pipeline(self, df: pd.DataFrame, pipeline: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, List[str]]:
        """
        Preview the effect of an auto-generated pipeline without modifying the input DataFrame.
        Returns (preview_df, messages).
        """
        try:
            from preprocessing.pipeline import run_pipeline
            preview_df, messages = run_pipeline(df, pipeline, preview=True)
            return preview_df, messages
        except Exception as e:
            logger.error(f"Error previewing pipeline: {e}")
            return df, [f"Error previewing pipeline: {e}"]
