import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import regex as re
import altair as alt
import logging
from typing import Optional
from utils.data_utils import dtype_split, _arrowize
from utils.stats_utils import compute_basic_stats
from utils.viz_utils import alt_histogram, alt_line_plot, word_cloud
from preprocessing.pipeline import run_pipeline
from sklearn.feature_selection import mutual_info_regression

logger = logging.getLogger(__name__)

class PreprocessingRecommendations:
    def analyze_dataset(self, df: pd.DataFrame) -> list:
        """
        Analyze the dataset and generate preprocessing recommendations.
        Returns a list of recommendation dictionaries.
        """
        recommendations = []
        
        if df is None or df.empty:
            logger.warning("Empty or None DataFrame provided for analysis.")
            return recommendations

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
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
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
                    'suggestion': 'Handle outliers using capping (IQR, factor=1.5) or log transformation.',
                    'priority': 0.7
                })
        
        # Bias Check for Categorical Data
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        for col in cat_cols:
            value_counts = df[col].value_counts(normalize=True)
            if value_counts.max() > 0.8:
                recommendations.append({
                    'type': 'bias_risk',
                    'column': col,
                    'suggestion': 'Mitigate potential bias by oversampling minority classes (ratio=1.0) or using robust ML algorithms.',
                    'priority': 0.8
                })
        
        # Duplicate Detection
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > len(df) * 0.01:
            recommendations.append({
                'type': 'duplicates',
                'count': duplicate_rows,
                'suggestion': 'Remove duplicate rows to improve data quality (keep=first).',
                'priority': 0.6
            })

        # PII Detection
        for col in cat_cols:
            if df[col].astype(str).str.contains(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b|\b(\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,4}|\d{3}[-.\s]\d{3}[-.\s]\d{4})\b|\b(?:\d[ -]*?){13,16}\b', regex=True, na=False).any():
                recommendations.append({
                    'type': 'pii',
                    'column': col,
                    'suggestion': 'Mask PII (e.g., emails, phone numbers, credit cards) to protect sensitive information.',
                    'priority': 0.9
                })

        # Time-Series Detection
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
        date_like_cols = [
            col for col in df.columns
            if col not in datetime_cols and df[col].astype(str).str.match(r'\d{4}-\d{2}-\d{2}', na=False).any()
        ]
        datetime_cols.extend(date_like_cols)
        if len(datetime_cols) > 0 and len(numeric_cols) > 0:
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
                    'suggestion': 'Clean text (e.g., remove stopwords, lemmatize) and extract TF-IDF features (max 100 features).',
                    'column': col,
                    'priority': 0.5
                })

        # Image Detection
        for col in df.columns:
            if df[col].astype(str).str.contains(r'\.(png|jpg|jpeg)$|data:image', regex=True, na=False).any():
                recommendations.append({
                    'type': 'image',
                    'severity': 'medium',
                    'suggestion': 'Resize images to a standard size (e.g., 224x224) or normalize pixel values for ML compatibility.',
                    'column': col,
                    'priority': 0.5
                })

        # 1. High-Cardinality Categorical
        for col in cat_cols:
            n_unique = df[col].nunique()
            n_rows = len(df)
            if n_unique > 50 or (n_unique / n_rows > 0.3):
                recommendations.append({
                    'type': 'high_cardinality',
                    'column': col,
                    'n_unique': n_unique,
                    'suggestion': 'High-cardinality categorical column detected. Consider target encoding, frequency encoding, or hashing.',
                    'priority': 0.7
                })

        # 2. Near-Zero-Variance Numeric
        for col in numeric_cols:
            variance = df[col].var()
            unique_count = df[col].nunique()
            if variance < 1e-8 or unique_count == 1:
                recommendations.append({
                    'type': 'near_zero_variance',
                    'column': col,
                    'variance': variance,
                    'suggestion': 'Near-zero-variance numeric column detected. Consider dropping this column.',
                    'priority': 0.6
                })

        # 3. Highly Collinear Numeric Pairs
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr().abs()
            np.fill_diagonal(corr_matrix.values, 0)
            high_corr = (corr_matrix > 0.95).stack()
            for (col1, col2), value in high_corr[high_corr].items():
                recommendations.append({
                    'type': 'collinearity',
                    'columns': [col1, col2],
                    'correlation': value,
                    'suggestion': f'High correlation ({value:.2f}) between {col1} and {col2}. Consider dropping one of the columns.',
                    'priority': 0.7
                })

        # 4. Constant / Quasi-Constant Categories
        for col in cat_cols:
            value_counts = df[col].value_counts(normalize=True)
            if value_counts.max() > 0.99:
                recommendations.append({
                    'type': 'constant_category',
                    'column': col,
                    'max_freq_ratio': value_counts.max(),
                    'suggestion': 'Constant or quasi-constant categorical column detected. Consider dropping or grouping rare categories as "Other".',
                    'priority': 0.6
                })

        # 5. Leakage via ID-like Columns
        for col in cat_cols:
            if re.search(r'^(id|key|uuid|guid)$', col.lower()) and df[col].nunique() == len(df):
                recommendations.append({
                    'type': 'id_leakage',
                    'column': col,
                    'suggestion': 'Potential ID-like column detected. Drop before train/test split to avoid data leakage.',
                    'priority': 0.9
                })

        # 6. Rare Category Tail
        for col in cat_cols:
            value_counts = df[col].value_counts()
            rare_cats = value_counts[value_counts < len(df) * 0.01]
            if not rare_cats.empty:
                recommendations.append({
                    'type': 'rare_categories',
                    'column': col,
                    'rare_count': len(rare_cats),
                    'suggestion': f'{len(rare_cats)} rare categories detected (<1% of rows). Consider grouping into "Rare" bucket.',
                    'priority': 0.5
                })

        # 7. Categorical High-Cardinality with Numeric Target (Potential Leakage)
        for col in cat_cols:
            if df[col].nunique() > 50:
                for target_col in numeric_cols:
                    X = pd.get_dummies(df[col].astype(str))
                    y = df[target_col].fillna(df[target_col].mean())
                    mi = mutual_info_regression(X, y)
                    if np.mean(mi) > 0.5:
                        recommendations.append({
                            'type': 'target_leakage',
                            'column': col,
                            'target': target_col,
                            'suggestion': f'High mutual information with {target_col}. Review for potential target leakage.',
                            'priority': 0.8
                        })

        # 8. Numeric Columns That Should Be Categorical
        for col in numeric_cols:
            if pd.api.types.is_integer_dtype(df[col]) and df[col].nunique() <= 20:
                recommendations.append({
                    'type': 'numeric_to_categorical',
                    'column': col,
                    'n_unique': df[col].nunique(),
                    'suggestion': 'Integer column with few unique values. Consider casting to categorical.',
                    'priority': 0.5
                })

        # 9. High Skew Numeric
        for col in numeric_cols:
            skew = stats.skew(df[col].dropna())
            if abs(skew) > 2:
                recommendations.append({
                    'type': 'high_skew',
                    'column': col,
                    'skew': skew,
                    'suggestion': 'High skewness detected. Consider Box-Cox or Yeo-Johnson transformation.',
                    'priority': 0.6
                })

        # 10. Heavy Class Imbalance (Classification Target)
        for col in cat_cols:
            value_counts = df[col].value_counts(normalize=True)
            if value_counts.min() < 0.05:
                recommendations.append({
                    'type': 'class_imbalance',
                    'column': col,
                    'min_ratio': value_counts.min(),
                    'suggestion': 'Heavy class imbalance detected. Consider SMOTE or class-weighting.',
                    'priority': 0.8
                })

        # 11. Date Columns with No Parseable Format
        for col in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                parsed = pd.to_datetime(df[col], errors='coerce')
                if parsed.isna().mean() > 0.5 and not df[col].astype(str).str.match(r'^\d{4}-\d{2}-\d{2}', na=False).all():
                    recommendations.append({
                        'type': 'unparseable_dates',
                        'column': col,
                        'suggestion': 'Date column with unparseable format. Consider manual parsing or dropping.',
                        'priority': 0.6
                    })

        # 12. Currency Symbols in Numeric Columns
        for col in cat_cols:
            if df[col].astype(str).str.contains(r'[\$€£]', regex=True, na=False).any():
                try:
                    numeric_vals = df[col].str.replace(r'[\$€£]', '', regex=True).astype(float)
                    recommendations.append({
                        'type': 'currency_symbols',
                        'column': col,
                        'suggestion': 'Currency symbols detected. Apply unit conversion to extract numeric values.',
                        'priority': 0.7
                    })
                except ValueError:
                    continue

        # 13. URL / File-Path Columns
        for col in df.columns:
            if df[col].astype(str).str.contains(r'http[s]?://|\.(csv|txt|pdf|png|jpg|jpeg)', regex=True, na=False).any():
                recommendations.append({
                    'type': 'url_filepath',
                    'column': col,
                    'suggestion': 'URL or file-path column detected. Consider extracting domain, file name, or dropping.',
                    'priority': 0.6
                })

        # 14. Boolean Disguised as 0/1 Integer
        for col in numeric_cols:
            if pd.api.types.is_integer_dtype(df[col]) and set(df[col].dropna().unique()) <= {0, 1}:
                recommendations.append({
                    'type': 'boolean_disguised',
                    'column': col,
                    'suggestion': 'Integer column with only 0/1 values. Consider casting to boolean.',
                    'priority': 0.5
                })

        # 15. Wide Numeric Ranges (Potential Normalization Need)
        for col in numeric_cols:
            if (df[col].max() - df[col].min()) > 1e6:
                recommendations.append({
                    'type': 'wide_numeric_range',
                    'column': col,
                    'range': df[col].max() - df[col].min(),
                    'suggestion': 'Wide numeric range detected. Consider standard or minmax scaling.',
                    'priority': 0.6
                })

        return sorted(recommendations, key=lambda x: x.get('priority', 0.5), reverse=True)

    @st.cache_data
    def preview_pipeline(self, df: pd.DataFrame, pipeline: list) -> tuple:
        """Preview the pipeline on a sampled dataset."""
        try:
            return run_pipeline(df, pipeline, preview=True)
        except Exception as e:
            logger.error(f"Error previewing pipeline: {e}")
            return df, [f"Error previewing pipeline: {e}"]

def section_recommendations():
    st.header("🔍 Data Quality Recommendations")
    df = st.session_state.get('df', None)
    if df is None:
        st.warning("Upload a dataset first.")
        return

    try:
        if df.empty:
            st.error("The dataset is empty. Please upload a non-empty dataset.")
            return

        # Warning for large datasets
        if len(df) > 100_000:
            st.warning(
                "Large dataset detected (>100,000 rows). Some analyses (e.g., collinearity) may be slow. "
                "Consider enabling 'Fast Mode' to skip expensive computations."
            )
            fast_mode = st.checkbox("Enable Fast Mode (skip collinearity and target leakage)", help="Skips computationally expensive checks for faster processing.")

        recommender = PreprocessingRecommendations()
        recommendations = recommender.analyze_dataset(df)
        
        if not recommendations:
            st.info("No significant issues detected in the dataset.")
            return

        # Summary Dashboard
        st.subheader("📊 Recommendation Summary")
        severity_counts = {
            'High': sum(1 for r in recommendations if r.get('severity', 'medium').lower() == 'high'),
            'Medium': sum(1 for r in recommendations if r.get('severity', 'medium').lower() == 'medium'),
            'Low': sum(1 for r in recommendations if r.get('severity', 'medium').lower() == 'low')
        }
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Issues", len(recommendations))
        with col2:
            st.metric("High Severity", severity_counts['High'])
        with col3:
            st.metric("Medium Severity", severity_counts['Medium'])
        
        # Tabbed Layout for Recommendations
        tabs = st.tabs(["All", "High Priority", "Medium Priority", "Low Priority"])
        priority_filters = {
            "All": (0.0, 1.0),
            "High Priority": (0.8, 1.0),
            "Medium Priority": (0.6, 0.8),
            "Low Priority": (0.0, 0.6)
        }

        # Global counter for unique keys
        key_counter = 0

        # Apply All Button
        if st.button("🚀 Apply All Recommended Actions", help="Add all suggested actions to the pipeline", key=f"apply_all_{key_counter}"):
            key_counter += 1
            pipeline_steps = []
            for rec in recommendations:
                if rec['type'] == 'missing_data':
                    strategy = 'mean' if rec['column'] in dtype_split(df)[0] else 'mode'
                    pipeline_steps.append({"kind": "impute", "params": {"columns": [rec['column']], "strategy": strategy}})
                elif rec['type'] == 'outliers':
                    pipeline_steps.append({"kind": "outliers", "params": {"columns": [rec['column']], "method": "iqr", "factor": 1.5}})
                elif rec['type'] == 'bias_risk':
                    pipeline_steps.append({"kind": "rebalance", "params": {"target": rec['column'], "method": "oversample", "ratio": 1.0}})
                elif rec['type'] == 'duplicates':
                    pipeline_steps.append({"kind": "duplicates", "params": {"subset": None, "keep": "first"}})
                elif rec['type'] == 'pii':
                    pipeline_steps.append({"kind": "mask_pii", "params": {"column": rec['column'], "pii_types": ["email", "phone", "credit_card"]}})
                elif rec['type'] == 'time_series':
                    pipeline_steps.append({"kind": "smooth_time_series", "params": {"column": rec['column'], "window": min(5, len(df)//10), "method": "moving_average", "interpolate": "linear"}})
                elif rec['type'] == 'text':
                    pipeline_steps.append({"kind": "clean_text", "params": {"column": rec['column'], "remove_stopwords": True, "lemmatize": True}})
                elif rec['type'] == 'image':
                    pipeline_steps.append({"kind": "resize_image", "params": {"column": rec['column'], "width": 224, "height": 224}})
                elif rec['type'] == 'high_cardinality':
                    pipeline_steps.append({"kind": "encode", "params": {"columns": [rec['column']], "method": "label"}})
                elif rec['type'] in ['near_zero_variance', 'constant_category', 'id_leakage', 'unparseable_dates', 'url_filepath']:
                    pipeline_steps.append({"kind": "drop_missing", "params": {"axis": "columns", "columns": [rec['column']]}})
                elif rec['type'] == 'collinearity':
                    pipeline_steps.append({"kind": "drop_missing", "params": {"axis": "columns", "columns": [rec['columns'][1]]}})
                elif rec['type'] == 'rare_categories':
                    pipeline_steps.append({"kind": "encode", "params": {"columns": [rec['column']], "method": "onehot", "max_categories": 10}})
                elif rec['type'] == 'numeric_to_categorical':
                    pipeline_steps.append({"kind": "type_convert", "params": {"column": rec['column'], "type": "str"}})
                elif rec['type'] == 'high_skew':
                    pipeline_steps.append({"kind": "skewness_transform", "params": {"column": rec['column'], "transform": "log"}})
                elif rec['type'] == 'class_imbalance':
                    pipeline_steps.append({"kind": "rebalance", "params": {"target": rec['column'], "method": "oversample", "ratio": 1.0}})
                elif rec['type'] == 'currency_symbols':
                    pipeline_steps.append({"kind": "unit_convert", "params": {"column": rec['column'], "factor": 1.0}})
                elif rec['type'] == 'boolean_disguised':
                    pipeline_steps.append({"kind": "type_convert", "params": {"column": rec['column'], "type": "bool"}})
                elif rec['type'] == 'wide_numeric_range':
                    pipeline_steps.append({"kind": "scale", "params": {"columns": [rec['column']], "method": "standard"}})
            st.session_state.pipeline.extend(pipeline_steps)
            st.success(f"Added {len(pipeline_steps)} recommended actions to the pipeline.")

        for tab_name, (min_priority, max_priority) in priority_filters.items():
            with tabs[list(priority_filters.keys()).index(tab_name)]:
                filtered_recs = [r for r in recommendations if min_priority <= r.get('priority', 0.5) <= max_priority]
                if not filtered_recs:
                    st.info(f"No {tab_name.lower()} recommendations.")
                    continue
                st.markdown(f"**{tab_name} Recommendations ({len(filtered_recs)})**")
                for rec in filtered_recs:
                    # Use global counter for unique keys
                    key_counter += 1
                    key_suffix = f"{tab_name.replace(' ', '_')}_{rec['type']}_{rec.get('column', '') or '_'.join(rec.get('columns', ['']))}_{key_counter}"
                    with st.expander(f"{rec['type'].replace('_', ' ').title()} (Priority: {rec.get('priority', 0.5):.2f})"):
                        st.markdown(f"**Issue**: {rec['type'].replace('_', ' ').title()}")
                        st.markdown(f"**Suggestion**: {rec['suggestion']}")
                        if 'column' in rec:
                            st.markdown(f"**Column**: `{rec['column']}`")
                        if 'columns' in rec:
                            st.markdown(f"**Columns**: `{', '.join(rec['columns'])}`")
                        if 'missing_count' in rec:
                            st.markdown(f"**Missing Count**: {rec['missing_count']} ({rec['missing_ratio']*100:.1f}%)")
                        if 'count' in rec:
                            st.markdown(f"**Count**: {rec['count']}")
                        if 'n_unique' in rec:
                            st.markdown(f"**Unique Values**: {rec['n_unique']}")
                        if 'variance' in rec:
                            st.markdown(f"**Variance**: {rec['variance']:.2e}")
                        if 'correlation' in rec:
                            st.markdown(f"**Correlation**: {rec['correlation']:.2f}")
                        if 'max_freq_ratio' in rec:
                            st.markdown(f"**Max Frequency Ratio**: {rec['max_freq_ratio']:.2f}")
                        if 'rare_count' in rec:
                            st.markdown(f"**Rare Categories**: {rec['rare_count']}")
                        if 'skew' in rec:
                            st.markdown(f"**Skewness**: {rec['skew']:.2f}")
                        if 'min_ratio' in rec:
                            st.markdown(f"**Minority Class Ratio**: {rec['min_ratio']:.2f}")
                        if 'range' in rec:
                            st.markdown(f"**Range**: {rec['range']:.2e}")

                        # Actionable Buttons
                        col1, col2 = st.columns(2)
                        if rec['type'] == 'missing_data':
                            with col1:
                                if st.button(f"📦 Impute ({'Mean' if rec['column'] in dtype_split(df)[0] else 'Mode'})", key=f"impute_{key_suffix}", help="Fill missing values with mean (numeric) or mode (categorical)."):
                                    strategy = 'mean' if rec['column'] in dtype_split(df)[0] else 'mode'
                                    step = {"kind": "impute", "params": {"columns": [rec['column']], "strategy": strategy}}
                                    st.session_state.pipeline.append(step)
                                    st.success("Added imputation step to pipeline.")
                            with col2:
                                if st.button(f"📦 Drop Column", key=f"drop_{key_suffix}", help="Remove the column with missing values."):
                                    step = {"kind": "drop_missing", "params": {"axis": "columns", "columns": [rec['column']]}}
                                    st.session_state.pipeline.append(step)
                                    st.success("Added drop column step to pipeline.")
                        elif rec['type'] == 'outliers':
                            with col1:
                                if st.button(f"📦 Handle Outliers", key=f"outliers_{key_suffix}", help="Cap outliers using IQR method."):
                                    step = {"kind": "outliers", "params": {"columns": [rec['column']], "method": "iqr", "factor": 1.5}}
                                    st.session_state.pipeline.append(step)
                                    st.success("Added outlier handling step to pipeline.")
                        elif rec['type'] == 'bias_risk':
                            with col1:
                                if st.button(f"📦 Rebalance", key=f"rebalance_{key_suffix}", help="Oversample minority classes to balance the dataset."):
                                    step = {"kind": "rebalance", "params": {"target": rec['column'], "method": "oversample", "ratio": 1.0}}
                                    st.session_state.pipeline.append(step)
                                    st.success("Added rebalancing step to pipeline.")
                        elif rec['type'] == 'duplicates':
                            with col1:
                                if st.button(f"📦 Remove Duplicates", key=f"duplicates_{key_suffix}", help="Remove duplicate rows, keeping the first occurrence."):
                                    step = {"kind": "duplicates", "params": {"subset": None, "keep": "first"}}
                                    st.session_state.pipeline.append(step)
                                    st.success("Added duplicate removal step to pipeline.")
                        elif rec['type'] == 'pii':
                            with col1:
                                if st.button(f"📦 Mask PII", key=f"pii_{key_suffix}", help="Mask sensitive information like emails and phone numbers."):
                                    step = {"kind": "mask_pii", "params": {"column": rec['column'], "pii_types": ["email", "phone", "credit_card"]}}
                                    st.session_state.pipeline.append(step)
                                    st.success("Added PII masking step to pipeline.")
                        elif rec['type'] == 'time_series':
                            with col1:
                                if st.button(f"📦 Smooth Time-Series", key=f"time_series_{key_suffix}", help="Apply moving average smoothing to the time-series."):
                                    step = {"kind": "smooth_time_series", "params": {"column": rec['column'], "window": min(5, len(df)//10), "method": "moving_average", "interpolate": "linear"}}
                                    st.session_state.pipeline.append(step)
                                    st.success("Added time-series smoothing step to pipeline.")
                            with col2:
                                if st.button(f"📦 Resample Time-Series", key=f"resample_{key_suffix}", help="Resample to a uniform frequency (daily)."):
                                    step = {"kind": "resample_time_series", "params": {"time_column": rec['column'], "freq": "1D", "agg_func": "mean"}}
                                    st.session_state.pipeline.append(step)
                                    st.success("Added time-series resampling step to pipeline.")
                        elif rec['type'] == 'text':
                            with col1:
                                if st.button(f"📦 Clean Text", key=f"text_{key_suffix}", help="Clean text by removing stopwords and lemmatizing."):
                                    step = {"kind": "clean_text", "params": {"column": rec['column'], "remove_stopwords": True, "lemmatize": True}}
                                    st.session_state.pipeline.append(step)
                                    st.success("Added text cleaning step to pipeline.")
                            with col2:
                                if st.button(f"📦 Extract TF-IDF", key=f"tfidf_{key_suffix}", help="Extract TF-IDF features from text."):
                                    step = {"kind": "extract_tfidf", "params": {"column": rec['column'], "max_features": 100}}
                                    st.session_state.pipeline.append(step)
                                    st.success("Added TF-IDF extraction step to pipeline.")
                        elif rec['type'] == 'image':
                            with col1:
                                if st.button(f"📦 Resize Images", key=f"image_{key_suffix}", help="Resize images to a standard size (224x224)."):
                                    step = {"kind": "resize_image", "params": {"column": rec['column'], "width": 224, "height": 224}}
                                    st.session_state.pipeline.append(step)
                                    st.success("Added image resizing step to pipeline.")
                            with col2:
                                if st.button(f"📦 Normalize Images", key=f"image_norm_{key_suffix}", help="Normalize image pixel values to [0,1]."):
                                    step = {"kind": "normalize_image", "params": {"column": rec['column']}}
                                    st.session_state.pipeline.append(step)
                                    st.success("Added image normalization step to pipeline.")
                        elif rec['type'] == 'high_cardinality':
                            with col1:
                                if st.button(f"📦 Encode Column", key=f"encode_{key_suffix}", help="Apply label encoding to high-cardinality column."):
                                    step = {"kind": "encode", "params": {"columns": [rec['column']], "method": "label"}}
                                    st.session_state.pipeline.append(step)
                                    st.success("Added encoding step to pipeline.")
                        elif rec['type'] == 'near_zero_variance':
                            with col1:
                                if st.button(f"📦 Drop Column", key=f"drop_var_{key_suffix}", help="Remove near-zero-variance column."):
                                    step = {"kind": "drop_missing", "params": {"axis": "columns", "columns": [rec['column']]}}
                                    st.session_state.pipeline.append(step)
                                    st.success("Added drop column step to pipeline.")
                        elif rec['type'] == 'collinearity':
                            with col1:
                                if st.button(f"📦 Drop Column {rec['columns'][1]}", key=f"drop_collinear_{key_suffix}", help="Remove one of the collinear columns."):
                                    step = {"kind": "drop_missing", "params": {"axis": "columns", "columns": [rec['columns'][1]]}}
                                    st.session_state.pipeline.append(step)
                                    st.success(f"Added drop column {rec['columns'][1]} step to pipeline.")
                        elif rec['type'] == 'constant_category':
                            with col1:
                                if st.button(f"📦 Drop Column", key=f"drop_const_{key_suffix}", help="Remove constant or quasi-constant column."):
                                    step = {"kind": "drop_missing", "params": {"axis": "columns", "columns": [rec['column']]}}
                                    st.session_state.pipeline.append(step)
                                    st.success("Added drop column step to pipeline.")
                        elif rec['type'] == 'id_leakage':
                            with col1:
                                if st.button(f"📦 Drop Column", key=f"drop_id_{key_suffix}", help="Remove ID-like column to avoid leakage."):
                                    step = {"kind": "drop_missing", "params": {"axis": "columns", "columns": [rec['column']]}}
                                    st.session_state.pipeline.append(step)
                                    st.success("Added drop column step to pipeline.")
                        elif rec['type'] == 'rare_categories':
                            with col1:
                                if st.button(f"📦 Group Rare Categories", key=f"encode_rare_{key_suffix}", help="Group rare categories into 'Rare' bucket."):
                                    step = {"kind": "encode", "params": {"columns": [rec['column']], "method": "onehot", "max_categories": 10}}
                                    st.session_state.pipeline.append(step)
                                    st.success("Added encoding step with rare category grouping to pipeline.")
                        elif rec['type'] == 'numeric_to_categorical':
                            with col1:
                                if st.button(f"📦 Cast to Categorical", key=f"cast_cat_{key_suffix}", help="Convert numeric to categorical type."):
                                    step = {"kind": "type_convert", "params": {"column": rec['column'], "type": "str"}}
                                    st.session_state.pipeline.append(step)
                                    st.success("Added type conversion to categorical step to pipeline.")
                        elif rec['type'] == 'high_skew':
                            with col1:
                                if st.button(f"📦 Transform Skewness", key=f"skew_{key_suffix}", help="Apply log transformation to reduce skewness."):
                                    step = {"kind": "skewness_transform", "params": {"column": rec['column'], "transform": "log"}}
                                    st.session_state.pipeline.append(step)
                                    st.success("Added skewness transformation step to pipeline.")
                        elif rec['type'] == 'class_imbalance':
                            with col1:
                                if st.button(f"📦 Rebalance Classes", key=f"rebalance_class_{key_suffix}", help="Oversample minority classes to balance dataset."):
                                    step = {"kind": "rebalance", "params": {"target": rec['column'], "method": "oversample", "ratio": 1.0}}
                                    st.session_state.pipeline.append(step)
                                    st.success("Added rebalancing step to pipeline.")
                        elif rec['type'] == 'unparseable_dates':
                            with col1:
                                if st.button(f"📦 Drop Column", key=f"drop_date_{key_suffix}", help="Remove unparseable date column."):
                                    step = {"kind": "drop_missing", "params": {"axis": "columns", "columns": [rec['column']]}}
                                    st.session_state.pipeline.append(step)
                                    st.success("Added drop column step to pipeline.")
                        elif rec['type'] == 'currency_symbols':
                            with col1:
                                if st.button(f"📦 Convert Units", key=f"unit_convert_{key_suffix}", help="Extract numeric values from currency column."):
                                    step = {"kind": "unit_convert", "params": {"column": rec['column'], "factor": 1.0}}
                                    st.session_state.pipeline.append(step)
                                    st.success("Added unit conversion step to pipeline.")
                        elif rec['type'] == 'url_filepath':
                            with col1:
                                if st.button(f"📦 Drop Column", key=f"drop_url_{key_suffix}", help="Remove URL or file-path column."):
                                    step = {"kind": "drop_missing", "params": {"axis": "columns", "columns": [rec['column']]}}
                                    st.session_state.pipeline.append(step)
                                    st.success("Added drop column step to pipeline.")
                        elif rec['type'] == 'boolean_disguised':
                            with col1:
                                if st.button(f"📦 Cast to Boolean", key=f"cast_bool_{key_suffix}", help="Convert 0/1 integer to boolean."):
                                    step = {"kind": "type_convert", "params": {"column": rec['column'], "type": "bool"}}
                                    st.session_state.pipeline.append(step)
                                    st.success("Added type conversion to boolean step to pipeline.")
                        elif rec['type'] == 'wide_numeric_range':
                            with col1:
                                if st.button(f"📦 Scale Column", key=f"scale_{key_suffix}", help="Apply standard scaling to normalize range."):
                                    step = {"kind": "scale", "params": {"columns": [rec['column']], "method": "standard"}}
                                    st.session_state.pipeline.append(step)
                                    st.success("Added scaling step to pipeline.")

        if st.button("🔄 Clear Recommendations", help="Reset recommendations and clear preview", key=f"clear_recommendations_{key_counter}"):
            key_counter += 1
            st.session_state.last_preview = None
            st.rerun()

    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        st.error(f"Error generating recommendations: {e}")
