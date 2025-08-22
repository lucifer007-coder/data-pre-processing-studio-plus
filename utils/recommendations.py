import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import regex as re
import altair as alt
import logging

logger = logging.getLogger(__name__)

class PreprocessingRecommendations:
    def analyze_dataset(self, df):
        recommendations = []
        
        # Missing Data Analysis
        missing_ratio = df.isnull().sum() / len(df)
        if missing_ratio.max() > 0.1:
            missing_cols = missing_ratio[missing_ratio > 0.1].index.tolist()
            for col in missing_cols:
                recommendations.append({
                    'type': 'missing_data',
                    'severity': 'high',
                    'suggestion': 'Consider imputation (mean/median for numeric, mode for categorical) or removal of rows/columns.',
                    'column': col,
                    'missing_count': int(df[col].isnull().sum()),
                    'missing_ratio': missing_ratio[col],
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
                transform = 'log' if df[col].min() > 0 else 'square_root'
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
            pipeline_suggestion.append({
                'kind': 'impute',
                'params': {'columns': [r['column'] for r in recommendations if r['type'] == 'missing_data'], 'strategy': 'mean'}
            })
        if any(r['type'] == 'outliers' for r in recommendations):
            pipeline_suggestion.append({
                'kind': 'outliers',
                'params': {'columns': [r['column'] for r in recommendations if r['type'] == 'outliers'], 'method': 'cap', 'detect_method': 'Z-score'}
            })
        if any(r['type'] == 'bias_risk' for r in recommendations):
            pipeline_suggestion.append({
                'kind': 'rebalance',
                'params': {'target': [r['column'] for r in recommendations if r['type'] == 'bias_risk'][0], 'method': 'oversample', 'ratio': 1.0}
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

    def visualize_recommendation(self, df, recommendation):
        """Generate Altair chart for a recommendation."""
        if recommendation['type'] == 'missing_data':
            data = pd.DataFrame({
                'Column': [recommendation['column']],
                'Missing_Ratio': [recommendation['missing_ratio']]
            })
            chart = alt.Chart(data).mark_bar().encode(
                x=alt.X('Column:N', title='Column'),
                y=alt.Y('Missing_Ratio:Q', title='Missing Ratio', scale=alt.Scale(domain=[0, 1])),
                tooltip=['Column', 'Missing_Ratio']
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

    def preview_pipeline(self, df, pipeline):
        """Preview the effects of applying the pipeline steps."""
        df_preview = df.copy()
        messages = []
        
        for step in pipeline:
            try:
                if step['kind'] == 'impute':
                    for col in step['params']['columns']:
                        if step['params']['strategy'] == 'mean':
                            df_preview[col] = df_preview[col].fillna(df_preview[col].mean())
                            messages.append(f"Imputed missing values in {col} with mean.")
                        elif step['params']['strategy'] == 'mode':
                            df_preview[col] = df_preview[col].fillna(df_preview[col].mode()[0])
                            messages.append(f"Imputed missing values in {col} with mode.")
                elif step['kind'] == 'outliers':
                    for col in step['params']['columns']:
                        if step['params']['method'] == 'cap':
                            q1 = df_preview[col].quantile(0.25)
                            q3 = df_preview[col].quantile(0.75)
                            iqr = q3 - q1
                            lower_bound = q1 - 1.5 * iqr
                            upper_bound = q3 + 1.5 * iqr
                            df_preview[col] = df_preview[col].clip(lower=lower_bound, upper=upper_bound)
                            messages.append(f"Capped outliers in {col} using IQR method.")
                elif step['kind'] == 'rebalance':
                    messages.append(f"Rebalancing for {step['params']['target']} not implemented in preview.")
                elif step['kind'] == 'duplicates':
                    before_count = len(df_preview)
                    df_preview = df_preview.drop_duplicates(subset=step['params']['subset'], keep=step['params']['keep'])
                    messages.append(f"Removed {before_count - len(df_preview)} duplicate rows.")
                elif step['kind'] == 'type_convert':
                    col = step['params']['column']
                    if step['params']['type'] == 'numeric':
                        df_preview[col] = pd.to_numeric(df_preview[col], errors='coerce')
                        messages.append(f"Converted {col} to numeric type.")
                elif step['kind'] == 'skewness_transform':
                    col = step['params']['column']
                    if step['params']['transform'] == 'log':
                        df_preview[col] = np.log1p(df_preview[col])
                        messages.append(f"Applied log transformation to {col}.")
                    elif step['params']['transform'] == 'square_root':
                        df_preview[col] = np.sqrt(df_preview[col])
                        messages.append(f"Applied square root transformation to {col}.")
                elif step['kind'] == 'mask_pii':
                    col = step['params']['column']
                    df_preview[col] = df_preview[col].astype(str).replace(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '***', regex=True)
                    messages.append(f"Masked PII in {col}.")
            except Exception as e:
                messages.append(f"Error applying {step['kind']} to {col}: {str(e)}")
        
        return df_preview, messages

def dtype_split(df):
    """Split columns into numeric and categorical."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    return numeric_cols, categorical_cols

def section_recommendations():
    st.header("🔍 Data Quality Recommendations")
    df = st.session_state.get('df', None)
    if df is None:
        st.warning("Upload a dataset first.")
        return

    try:
        recommender = PreprocessingRecommendations()
        recommendations = recommender.analyze_dataset(df)
        
        if not recommendations:
            st.info("No significant issues detected in the dataset.")
            return

        st.subheader("Summary of Recommendations")
        summary_data = []
        for i, rec in enumerate(recommendations, 1):
            summary_data.append({
                "ID": i,
                "Type": rec['type'].replace('_', ' ').title(),
                "Priority": f"{rec.get('priority', 0.5):.2f}",
                "Severity": rec.get('severity', 'medium').title(),
                "Column": rec.get('column', rec.get('columns', ['N/A'])[0]),
                "Details": rec.get('suggestion', '')
            })
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

        st.subheader("Detailed Recommendations")
        for i, rec in enumerate(recommendations, 1):
            with st.expander(f"{i}. {rec['type'].replace('_', ' ').title()} (Priority: {rec.get('priority', 0.5):.2f})"):
                st.write(f"**Suggestion**: {rec['suggestion']}")
                if 'column' in rec:
                    st.write(f"**Column**: {rec['column']}")
                if 'columns' in rec:
                    st.write(f"**Affected Columns**: {', '.join(rec['columns'])}")
                if 'missing_count' in rec:
                    st.write(f"**Missing Count**: {rec['missing_count']} ({rec['missing_ratio']*100:.1f}%)")
                if 'count' in rec:
                    st.write(f"**Count**: {rec['count']}")

                # Visualization
                chart = recommender.visualize_recommendation(df, rec)
                if chart:
                    st.altair_chart(chart, use_container_width=True)

                # Actionable buttons
                col1, col2 = st.columns(2)
                if rec['type'] == 'missing_data':
                    with col1:
                        if st.button(f"📦 Add Imputation", key=f"impute_{i}", help="Add imputation step to pipeline"):
                            strategy = 'mean' if rec['column'] in dtype_split(df)[0] else 'mode'
                            step = {
                                "kind": "impute",
                                "params": {"columns": [rec['column']], "strategy": strategy}
                            }
                            st.session_state.pipeline.append(step)
                            st.success("Added imputation step to pipeline.")
                    with col2:
                        if st.button(f"📦 Add Drop", key=f"drop_{i}", help="Add drop step to pipeline"):
                            step = {
                                "kind": "drop_missing",
                                "params": {"axis": "rows", "columns": [rec['column']]}
                            }
                            st.session_state.pipeline.append(step)
                            st.success("Added drop step to pipeline.")
                elif rec['type'] == 'outliers':
                    with col1:
                        if st.button(f"📦 Add Outlier Handling", key=f"outliers_{i}", help="Add outlier handling step"):
                            step = {
                                "kind": "outliers",
                                "params": {"columns": [rec['column']], "method": "iqr", "factor": 1.5}
                            }
                            st.session_state.pipeline.append(step)
                            st.success("Added outlier handling step to pipeline.")
                elif rec['type'] == 'bias_risk':
                    with col1:
                        if st.button(f"📦 Add Rebalancing", key=f"bias_{i}", help="Add rebalancing step"):
                            step = {
                                "kind": "rebalance",
                                "params": {"target": rec['column'], "method": "oversample", "ratio": 1.0}
                            }
                            st.session_state.pipeline.append(step)
                            st.success("Added rebalancing step to pipeline.")
                elif rec['type'] == 'duplicates':
                    with col1:
                        if st.button(f"📦 Add Duplicate Removal", key=f"duplicates_{i}", help="Add duplicate removal step"):
                            step = {
                                "kind": "duplicates",
                                "params": {"subset": None, "keep": "first"}
                            }
                            st.session_state.pipeline.append(step)
                            st.success("Added duplicate removal step to pipeline.")
                elif rec['type'] == 'data_type_mismatch':
                    with col1:
                        if st.button(f"📦 Add Type Conversion", key=f"type_{i}", help="Add type conversion step"):
                            type_val = "numeric" if "numeric" in rec['suggestion'] else "datetime" if "datetime" in rec['suggestion'] else "boolean"
                            step = {
                                "kind": "type_convert" if type_val != "datetime" else "standardize_dates",
                                "params": {"column": rec['column'], "type": type_val} if type_val != "datetime" else {"columns": [rec['column']]}
                            }
                            st.session_state.pipeline.append(step)
                            st.success("Added type conversion step to pipeline.")
                elif rec['type'] == 'skewness':
                    with col1:
                        transform = 'log' if 'log' in rec['suggestion'] else 'square_root'
                        if st.button(f"📦 Add {transform.title()} Transformation", key=f"skew_{i}", help=f"Add {transform} transformation"):
                            step = {
                                "kind": "skewness_transform",
                                "params": {"column": rec['column'], "transform": transform}
                            }
                            st.session_state.pipeline.append(step)
                            st.success(f"Added {transform} transformation step to pipeline.")
                elif rec['type'] == 'sensitive_data':
                    with col1:
                        if st.button(f"📦 Add PII Masking", key=f"pii_{i}", help="Add PII masking step"):
                            step = {
                                "kind": "mask_pii",
                                "params": {"column": rec['column'], "pii_types": ["email", "phone", "credit_card"]}
                            }
                            st.session_state.pipeline.append(step)
                            st.success("Added PII masking step to pipeline.")
                elif rec['type'] == 'auto_pipeline':
                    with col1:
                        if st.button(f"🔍 Preview Auto Pipeline", key=f"preview_auto_{i}", help="Preview the auto pipeline"):
                            preview_df, messages = recommender.preview_pipeline(df, rec['pipeline'])
                            st.session_state.last_preview = (preview_df, "\n".join(messages))
                            st.write("**Preview Results**:")
                            for msg in messages:
                                st.write(msg)
                            st.dataframe(preview_df.head(10))
                            # Display before/after stats
                            before_stats = compute_basic_stats(df)
                            after_stats = compute_basic_stats(preview_df)
                            st.write("**Preview Statistics**")
                            col_stats1, col_stats2 = st.columns(2)
                            with col_stats1:
                                st.write("Before")
                                st.write(f"Shape: {before_stats['shape']}")
                                st.write(f"Missing Values: {before_stats['missing_total']}")
                            with col_stats2:
                                st.write("After Preview")
                                st.write(f"Shape: {after_stats['shape']}")
                                st.write(f"Missing Values: {after_stats['missing_total']}")
                    with col2:
                        if st.button(f"📦 Add Auto Pipeline", key=f"auto_{i}", help="Add all auto pipeline steps"):
                            for step in rec['pipeline']:
                                st.session_state.pipeline.append(step)
                            st.success("Added auto pipeline steps to pipeline.")

        if st.button("🔄 Clear Recommendations", help="Reset recommendations and clear preview"):
            st.session_state.last_preview = None
            st.rerun()

    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        st.error(f"Error generating recommendations: {e}")

def compute_basic_stats(df):
    """Compute basic statistics for the dataset."""
    return {
        'shape': df.shape,
        'missing_total': df.isnull().sum().sum()
    }
