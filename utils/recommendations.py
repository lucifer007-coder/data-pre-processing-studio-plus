import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import regex as re
import altair as alt
import logging
from utils.data_utils import dtype_split, _arrowize
from utils.stats_utils import compute_basic_stats
from utils.viz_utils import alt_histogram, alt_line_plot, word_cloud
from preprocessing.pipeline import run_pipeline

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
                'count': duplicate_rows,
                'suggestion': 'Remove duplicate rows to improve data quality.',
                'priority': 0.6
            })

        # PII Detection
        for col in cat_cols:
            if df[col].astype(str).str.contains(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b|\b(\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,4}|\d{3}[-.\s]\d{3}[-.\s]\d{4})\b|\b(?:\d[ -]*?){13,16}\b', regex=True, na=False).any():
                recommendations.append({
                    'type': 'pii',
                    'column': col,
                    'suggestion': 'Mask PII (e.g., emails, phone numbers) to protect sensitive information.',
                    'priority': 0.9
                })

        # Time-Series Detection
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
        date_like_cols = [
            col for col in df.columns
            if col not in datetime_cols and df[col].astype(str).str.match(r'\d{4}-\d{2}-\d{2}', na=False).any()
        ]
        datetime_cols.extend(date_like_cols)
        if datetime_cols and numeric_cols:
            for col in datetime_cols:
                recommendations.append({
                    'type': 'time_series',
                    'severity': 'medium',
                    'suggestion': f'Consider smoothing with a window of {min(5, len(df)//10)} or resampling to a uniform frequency (e.g., daily).',
                    'column': col,
                    'priority': 0.6
                })

        # Text Detection
        for col in cat_cols:
            if df[col].nunique() > 10 and not df[col].astype(str).str.strip().eq('').all():
                recommendations.append({
                    'type': 'text',
                    'severity': 'medium',
                    'suggestion': 'Clean text (e.g., remove stopwords) and extract TF-IDF features (e.g., max 100 features).',
                    'column': col,
                    'priority': 0.5
                })

        # Image Detection
        for col in df.columns:
            if df[col].astype(str).str.contains(r'\.(png|jpg|jpeg)$|data:image', regex=True, na=False).any():
                recommendations.append({
                    'type': 'image',
                    'severity': 'medium',
                    'suggestion': 'Resize images to a standard size (e.g., 224x224) or normalize pixel values.',
                    'column': col,
                    'priority': 0.5
                })

        return sorted(recommendations, key=lambda x: x.get('priority', 0.5), reverse=True)

    def visualize_recommendation(self, df, rec):
        if rec['type'] == 'missing_data':
            return alt_histogram(df, rec['column'], f"Distribution of {rec['column']} (Missing: {rec['missing_count']})")
        elif rec['type'] == 'outliers':
            return alt_histogram(df, rec['column'], f"Outliers in {rec['column']} ({rec['count']} detected)")
        elif rec['type'] == 'time_series':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if numeric_cols:
                return alt_line_plot(df, rec['column'], numeric_cols[0], f"Time-Series: {rec['column']} vs {numeric_cols[0]}")
        elif rec['type'] == 'text':
            word_cloud(df, rec['column'], f"Word Cloud for {rec['column']}")
            return None
        return None

    def preview_pipeline(self, df, pipeline):
        return run_pipeline(df, pipeline, preview=True)

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
                        if st.button(f"📦 Drop Column", key=f"drop_{i}", help="Add drop column step"):
                            step = {
                                "kind": "drop_missing",
                                "params": {"axis": "columns", "columns": [rec['column']]}
                            }
                            st.session_state.pipeline.append(step)
                            st.success("Added drop column step to pipeline.")
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
                        if st.button(f"📦 Add Rebalancing", key=f"rebalance_{i}", help="Add rebalancing step"):
                            step = {
                                "kind": "rebalance",
                                "params": {"target": rec['column'], "method": "oversample", "ratio": 1.0}
                            }
                            st.session_state.pipeline.append(step)
                            st.success("Added rebalancing step to pipeline.")
                elif rec['type'] == 'duplicates':
                    with col1:
                        if st.button(f"📦 Remove Duplicates", key=f"duplicates_{i}", help="Add duplicate removal step"):
                            step = {
                                "kind": "duplicates",
                                "params": {"subset": None, "keep": "first"}
                            }
                            st.session_state.pipeline.append(step)
                            st.success("Added duplicate removal step to pipeline.")
                elif rec['type'] == 'pii':
                    with col1:
                        if st.button(f"📦 Add PII Masking", key=f"pii_{i}", help="Add PII masking step"):
                            step = {
                                "kind": "mask_pii",
                                "params": {"column": rec['column'], "pii_types": ["email", "phone", "credit_card"]}
                            }
                            st.session_state.pipeline.append(step)
                            st.success("Added PII masking step to pipeline.")
                elif rec['type'] == 'time_series':
                    with col1:
                        if st.button(f"📦 Add Time-Series Smoothing", key=f"time_series_{i}", help="Add smoothing step"):
                            step = {
                                "kind": "smooth_time_series",
                                "params": {"column": rec['column'], "window": min(5, len(df)//10), "method": "moving_average", "interpolate": "linear"}
                            }
                            st.session_state.pipeline.append(step)
                            st.success("Added time-series smoothing step to pipeline.")
                    with col2:
                        if st.button(f"📦 Add Time-Series Resampling", key=f"resample_{i}", help="Add resampling step"):
                            step = {
                                "kind": "resample_time_series",
                                "params": {"time_column": rec['column'], "freq": "1D", "agg_func": "mean"}
                            }
                            st.session_state.pipeline.append(step)
                            st.success("Added time-series resampling step to pipeline.")
                elif rec['type'] == 'text':
                    with col1:
                        if st.button(f"📦 Add Text Cleaning", key=f"text_{i}", help="Add text cleaning step"):
                            step = {
                                "kind": "clean_text",
                                "params": {"column": rec['column'], "remove_stopwords": True, "lemmatize": True}
                            }
                            st.session_state.pipeline.append(step)
                            st.success("Added text cleaning step to pipeline.")
                    with col2:
                        if st.button(f"📦 Add TF-IDF Extraction", key=f"tfidf_{i}", help="Add TF-IDF extraction step"):
                            step = {
                                "kind": "extract_tfidf",
                                "params": {"column": rec['column'], "max_features": 100}
                            }
                            st.session_state.pipeline.append(step)
                            st.success("Added TF-IDF extraction step to pipeline.")
                elif rec['type'] == 'image':
                    with col1:
                        if st.button(f"📦 Add Image Resizing", key=f"image_{i}", help="Add image resizing step"):
                            step = {
                                "kind": "resize_image",
                                "params": {"column": rec['column'], "width": 224, "height": 224}
                            }
                            st.session_state.pipeline.append(step)
                            st.success("Added image resizing step to pipeline.")
                    with col2:
                        if st.button(f"📦 Add Image Normalization", key=f"image_norm_{i}", help="Add image normalization step"):
                            step = {
                                "kind": "normalize_image",
                                "params": {"column": rec['column']}
                            }
                            st.session_state.pipeline.append(step)
                            st.success("Added image normalization step to pipeline.")

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
