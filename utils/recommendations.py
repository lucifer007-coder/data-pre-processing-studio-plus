import pandas as pd
import numpy as np
from scipy import stats
import regex as re
import altair as alt

class PreprocessingRecommendations:
    def analyze_dataset(self, df):
        recommendations = []
        
        # Missing Data Analysis
        missing_ratio = df.isnull().sum() / len(df)
        if missing_ratio.max() > 0.1:
            recommendations.append({
                'type': 'missing_data',
                'severity': 'high',
                'suggestion': 'Consider imputation (mean/median for numeric, mode for categorical) or removal of rows/columns.',
                'columns': missing_ratio[missing_ratio > 0.1].index.tolist(),
                'priority': 0.9
            })
        
        # Outlier Detection with Z-score
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            clean_series = df[col].dropna()
            if len(clean_series) < 2:
                continue
            z_scores = np.abs(stats.zscore(clean_series))
            z_scores_series = pd.Series(z_scores, index=clean_series.index).reindex(df.index, fill_value=0)
            outliers = df[col][z_scores_series > 3]
            if len(outliers.dropna()) > len(df) * 0.05:
                recommendations.append({
                    'type': 'outliers',
                    'column': col,
                    'count': len(outliers.dropna()),
                    'suggestion': 'Handle outliers using capping, removal, or log transformation.',
                    'priority': 0.7
                })
        
        # Bias Check for Categorical Data
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            value_counts = df[col].value_counts(normalize=True)
            if value_counts.max() > 0.8:
                recommendations.append({
                    'type': 'bias_risk',
                    'column': col,
                    'suggestion': 'Mitigate potential bias by rebalancing or using robust ML algorithms.',
                    'priority': 0.8
                })
        
        # Duplicate Detection
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > len(df) * 0.01:
            recommendations.append({
                'type': 'duplicates',
                'severity': 'medium',
                'suggestion': 'Remove duplicate rows to improve data quality.',
                'count': int(duplicate_rows),
                'priority': 0.6
            })
        
        # Data Type Mismatch
        for col in df.columns:
            inferred_type = pd.api.types.infer_dtype(df[col], skipna=True)
            if inferred_type in ('mixed', 'string') and df[col].str.isnumeric().any():
                recommendations.append({
                    'type': 'data_type_mismatch',
                    'column': col,
                    'suggestion': 'Convert to numeric type to correct mixed or string-based numeric data.',
                    'priority': 0.5
                })
        
        # Skewness Analysis
        for col in numeric_cols:
            skewness = df[col].skew()
            if abs(skewness) > 1:
                transform = 'log' if df[col].min() > 0 else 'square root'
                recommendations.append({
                    'type': 'skewness',
                    'column': col,
                    'suggestion': f"Apply {transform} transformation to reduce skewness ({skewness:.2f}).",
                    'priority': 0.4
                })
        
        # Sensitive Data Detection
        pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(\+\d{1,3}[-.\s]?\d{10}|\d{3}[-.\s]\d{3}[-.\s]\d{4})\b',
            'credit_card': r'\b(?:\d[ -]*?){13,16}\b'
        }
        for col in cat_cols:
            for pii_type, pattern in pii_patterns.items():
                if df[col].astype(str).str.contains(pattern, regex=True, na=False).any():
                    recommendations.append({
                        'type': 'sensitive_data',
                        'column': col,
                        'suggestion': f"Potential {pii_type} detected; consider masking or anonymization.",
                        'priority': 0.95
                    })
        
        # Auto-Preprocessing Pipeline
        pipeline_suggestion = []
        if any(r['type'] == 'missing_data' for r in recommendations):
            pipeline_suggestion.append({'kind': 'impute', 'params': {'columns': missing_ratio[missing_ratio > 0.1].index.tolist(), 'strategy': 'mean'}})
        if any(r['type'] == 'outliers' for r in recommendations):
            pipeline_suggestion.append({'kind': 'outliers', 'params': {'columns': [r['column'] for r in recommendations if r['type'] == 'outliers'], 'method': 'cap', 'detect_method': 'Z-score'}})
        if any(r['type'] == 'bias_risk' for r in recommendations):
            pipeline_suggestion.append({'kind': 'rebalance', 'params': {'target': [r['column'] for r in recommendations if r['type'] == 'bias_risk'][0], 'method': 'oversample', 'ratio': 1.0}})
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

    def visualize_recommendation(self, df, recommendation):
        """Generate Altair chart for a recommendation."""
        if recommendation['type'] == 'missing_data':
            data = pd.DataFrame({
                'Column': recommendation['columns'],
                'Missing_Ratio': [df[col].isnull().mean() for col in recommendation['columns']]
            })
            chart = alt.Chart(data).mark_bar().encode(
                x=alt.X('Column:N', title='Column'),
                y=alt.Y('Missing_Ratio:Q', title='Missing Ratio', scale=alt.Scale(domain=[0, 1])),
                tooltip=['Column', 'Missing_Ratio']
            ).properties(
                title="Missing Data by Column",
                width=400,
                height=300
            )
            return chart
        elif recommendation['type'] == 'outliers':
            col = recommendation['column']
            clean_series = df[col].dropna()
            if len(clean_series) < 2:
                return None
            z_scores = np.abs(stats.zscore(clean_series))
            data = pd.DataFrame({
                'Value': clean_series,
                'Z_Score': z_scores
            }).reset_index()
            chart = alt.Chart(data).mark_circle().encode(
                x=alt.X('index:O', title='Index'),
                y=alt.Y('Value:Q', title=col),
                color=alt.condition(
                    alt.datum.Z_Score > 3,
                    alt.value('red'),
                    alt.value('steelblue')
                ),
                tooltip=['Value', 'Z_Score']
            ).properties(
                title=f"Outliers in {col}",
                width=400,
                height=300
            )
            return chart
        elif recommendation['type'] == 'bias_risk':
            col = recommendation['column']
            value_counts = df[col].value_counts(normalize=True)
            data = pd.DataFrame({
                'Category': value_counts.index,
                'Proportion': value_counts.values
            })
            chart = alt.Chart(data).mark_bar().encode(
                x=alt.X('Category:N', title=col),
                y=alt.Y('Proportion:Q', title='Proportion'),
                color=alt.condition(
                    alt.datum.Proportion > 0.8,
                    alt.value('red'),
                    alt.value('steelblue')
                ),
                tooltip=['Category', 'Proportion']
            ).properties(
                title=f"Bias Risk in {col}",
                width=400,
                height=300
            )
            return chart
        elif recommendation['type'] == 'skewness':
            col = recommendation['column']
            data = df[[col]].dropna().reset_index()
            chart = alt.Chart(data).mark_bar().encode(
                x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=30), title=col),
                y=alt.Y('count():Q', title='Count'),
                tooltip=['count()']
            ).properties(
                title=f"Distribution of {col} (Skewness: {df[col].skew():.2f})",
                width=400,
                height=300
            )
            return chart
        return None