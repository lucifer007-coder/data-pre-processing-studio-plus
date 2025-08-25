import streamlit as st
import pandas as pd
import numpy as np
import dask.dataframe as dd
from scipy import stats
import logging
import warnings
from typing import Optional, List, Dict, Tuple
from utils.data_utils import dtype_split
from utils.stats_utils import compute_basic_stats
from preprocessing.pipeline import run_pipeline
import threading

logger = logging.getLogger(__name__)

# Thread lock for session state updates
session_lock = threading.Lock()

# Suppress regex warnings for str.contains
warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression, and has match groups")

class PreprocessingRecommendations:
    def analyze_dataset(self, df: pd.DataFrame | dd.DataFrame) -> List[Dict]:
        """
        Analyze the dataset and generate preprocessing recommendations.
        Returns a list of recommendation dictionaries.
        """
        recommendations = []
        
        if df is None or (isinstance(df, pd.DataFrame) and df.empty) or (isinstance(df, dd.DataFrame) and df.compute().empty):
            logger.warning("Empty or None DataFrame provided for analysis.")
            return recommendations

        # Missing Data Analysis
        if isinstance(df, dd.DataFrame):
            missing_ratio = df.isnull().mean().compute()
            missing_counts = df.isnull().sum().compute()
            total_rows = df.shape[0].compute()
            columns = df.columns.compute().tolist()
        else:
            missing_ratio = df.isnull().mean()
            missing_counts = df.isnull().sum()
            total_rows = len(df)
            columns = df.columns.tolist()
        if missing_ratio.max() > 0.1:
            missing_cols = missing_ratio[missing_ratio > 0.1].index.tolist()
            for col in missing_cols:
                recommendations.append({
                    'type': 'missing_data',
                    'severity': 'high',
                    'suggestion': 'Consider imputation (mean/median for numeric, mode for categorical) or removal of rows/columns.',
                    'column': col,
                    'missing_count': int(missing_counts[col]),
                    'missing_ratio': missing_ratio[col],
                    'priority': 0.9
                })
        
        # Outlier Detection with Z-score
        numeric_cols = dtype_split(df)[0]
        for col in numeric_cols:
            try:
                if isinstance(df, dd.DataFrame):
                    clean_series = df[col].dropna()
                    if clean_series.shape[0].compute() < 2:
                        continue
                    sample_df = clean_series.head(10000).compute()
                    z_scores = np.abs(stats.zscore(sample_df))
                    outlier_count = np.sum(z_scores > 3)
                else:
                    clean_series = df[col].dropna()
                    if len(clean_series) < 2:
                        continue
                    z_scores = np.abs(stats.zscore(clean_series))
                    outlier_count = np.sum(z_scores > 3)
                if outlier_count > total_rows * 0.05:
                    recommendations.append({
                        'type': 'outliers',
                        'severity': 'medium',
                        'column': col,
                        'count': outlier_count,
                        'suggestion': 'Handle outliers using capping (IQR, factor=1.5) or log transformation.',
                        'priority': 0.7
                    })
            except Exception as e:
                logger.warning(f"Error analyzing outliers in {col}: {e}")
        
        # Biased Categorical Data (bias_risk/class_imbalance)
        categorical_cols = dtype_split(df)[1]
        for col in categorical_cols:
            try:
                if isinstance(df, dd.DataFrame):
                    value_counts = df[col].value_counts().compute()
                    total = value_counts.sum()
                    unique_count = df[col].nunique().compute()
                else:
                    value_counts = df[col].value_counts()
                    total = value_counts.sum()
                    unique_count = df[col].nunique()
                # Bias Risk
                if value_counts.max() / total > 0.8:
                    recommendations.append({
                        'type': 'bias_risk',
                        'severity': 'medium',
                        'column': col,
                        'suggestion': 'Mitigate potential bias by oversampling minority classes or using robust ML algorithms.',
                        'priority': 0.8
                    })
                # Class Imbalance
                if value_counts.min() / total < 0.05:
                    recommendations.append({
                        'type': 'class_imbalance',
                        'severity': 'high',
                        'column': col,
                        'min_ratio': value_counts.min() / total,
                        'suggestion': 'Heavy class imbalance detected. Consider SMOTE or class-weighting.',
                        'priority': 0.8
                    })
                # Constant/Quasi-Constant Categories
                if value_counts.max() / total > 0.99:
                    recommendations.append({
                        'type': 'constant_category',
                        'severity': 'medium',
                        'column': col,
                        'max_freq_ratio': value_counts.max() / total,
                        'suggestion': 'Constant or quasi-constant categorical column detected. Consider dropping or grouping rare categories as "Other".',
                        'priority': 0.6
                    })
                # Rare Categories
                rare_cats = value_counts[value_counts / total < 0.01]
                if not rare_cats.empty:
                    recommendations.append({
                        'type': 'rare_categories',
                        'severity': 'low',
                        'column': col,
                        'rare_count': len(rare_cats),
                        'suggestion': f'{len(rare_cats)} rare categories detected (<1% of rows). Consider grouping into "Rare" bucket.',
                        'priority': 0.5
                    })
                # High Cardinality
                if unique_count > total_rows * 0.1 or unique_count > 50:
                    recommendations.append({
                        'type': 'high_cardinality',
                        'severity': 'medium',
                        'column': col,
                        'unique_count': unique_count,
                        'suggestion': 'High-cardinality categorical column detected. Consider target encoding, frequency encoding, or hashing.',
                        'priority': 0.7
                    })
            except Exception as e:
                logger.warning(f"Error analyzing categorical column {col}: {e}")
        
        # Duplicate Detection
        try:
            duplicate_count = df.duplicated().sum().compute() if isinstance(df, dd.DataFrame) else df.duplicated().sum()
            if duplicate_count > total_rows * 0.01:
                recommendations.append({
                    'type': 'duplicates',
                    'severity': 'medium',
                    'count': int(duplicate_count),
                    'suggestion': 'Remove duplicate rows to improve data quality (keep=first).',
                    'priority': 0.6
                })
        except Exception as e:
            logger.warning(f"Error analyzing duplicates: {e}")
        
        # Leakage via ID-like Columns
        import re
        for col in categorical_cols:
            try:
                if isinstance(df, dd.DataFrame):
                    unique_count = df[col].nunique().compute()
                else:
                    unique_count = df[col].nunique()
                if re.search(r'^(id|key|uuid|guid)$', col.lower()) and unique_count == total_rows:
                    recommendations.append({
                        'type': 'id_leakage',
                        'severity': 'high',
                        'column': col,
                        'suggestion': 'Potential ID-like column detected. Drop before train/test split to avoid data leakage.',
                        'priority': 0.9
                    })
            except Exception as e:
                logger.warning(f"Error analyzing ID-like column {col}: {e}")

        # Numeric Columns That Should Be Categorical
        for col in numeric_cols:
            try:
                if isinstance(df, dd.DataFrame):
                    unique_count = df[col].nunique().compute()
                    is_integer = pd.api.types.is_integer_dtype(df[col].dtype)
                else:
                    unique_count = df[col].nunique()
                    is_integer = pd.api.types.is_integer_dtype(df[col])
                if is_integer and unique_count <= 20:
                    recommendations.append({
                        'type': 'numeric_to_categorical',
                        'severity': 'low',
                        'column': col,
                        'n_unique': unique_count,
                        'suggestion': 'Integer column with few unique values. Consider casting to categorical.',
                        'priority': 0.5
                    })
            except Exception as e:
                logger.warning(f"Error analyzing numeric-to-categorical in {col}: {e}")

        # High Skew Numeric
        for col in numeric_cols:
            try:
                if isinstance(df, dd.DataFrame):
                    skew = stats.skew(df[col].dropna().head(10000).compute())
                else:
                    skew = stats.skew(df[col].dropna())
                if abs(skew) > 2:
                    recommendations.append({
                        'type': 'high_skew',
                        'severity': 'medium',
                        'column': col,
                        'skew': skew,
                        'suggestion': 'High skewness detected. Consider applying Box-Cox or Yeo-Johnson transformation.',
                        'priority': 0.7
                    })
            except Exception as e:
                logger.warning(f"Error analyzing skewness in {col}: {e}")

        # Boolean Disguised as 0/1
        for col in numeric_cols:
            try:
                if isinstance(df, dd.DataFrame):
                    unique_vals = df[col].unique().compute()
                else:
                    unique_vals = df[col].unique()
                if set(unique_vals.dropna()) <= {0, 1}:
                    recommendations.append({
                        'type': 'boolean_disguised',
                        'severity': 'low',
                        'column': col,
                        'suggestion': 'Integer column with only 0/1 values. Consider casting to boolean.',
                        'priority': 0.5
                    })
            except Exception as e:
                logger.warning(f"Error analyzing boolean-disguised in {col}: {e}")

        # Wide Numeric Range
        for col in numeric_cols:
            try:
                if isinstance(df, dd.DataFrame):
                    col_range = df[col].max().compute() - df[col].min().compute()
                else:
                    col_range = df[col].max() - df[col].min()
                if col_range > 1e6:
                    recommendations.append({
                        'type': 'wide_numeric_range',
                        'severity': 'medium',
                        'column': col,
                        'range': col_range,
                        'suggestion': 'Wide numeric range detected. Consider scaling (e.g., standard or minmax).',
                        'priority': 0.6
                    })
            except Exception as e:
                logger.warning(f"Error analyzing numeric range in {col}: {e}")

        return sorted(recommendations, key=lambda x: x.get('priority', 0.5), reverse=True)

    def preview_pipeline(self, df: pd.DataFrame | dd.DataFrame, pipeline: List[Dict]) -> Tuple[pd.DataFrame, List[str]]:
        """
        Preview the effect of applying a preprocessing pipeline.
        """
        try:
            return run_pipeline(df, pipeline, preview=True)
        except Exception as e:
            logger.error(f"Error previewing pipeline: {e}")
            return df, [f"Error previewing pipeline: {e}"]

def section_recommendations():
    """
    Streamlit section for displaying data quality recommendations and pipeline actions.
    """
    st.header("🔍 Data Quality Recommendations")
    df = st.session_state.get('df', None)
    if df is None:
        st.warning("Upload a dataset first.")
        return

    try:
        total_rows = df.shape[0].compute() if isinstance(df, dd.DataFrame) else len(df)
        fast_mode = total_rows > 100_000
        if fast_mode:
            st.warning("Large dataset detected (>100,000 rows). Fast mode enabled to optimize performance.")
            fast_mode = not st.checkbox("Disable Fast Mode", value=False)

        recommender = PreprocessingRecommendations()
        recommendations = recommender.analyze_dataset(df)

        if not recommendations:
            st.info("No significant issues detected in the dataset.")
            return

        # Summary Dashboard
        st.subheader("Summary Dashboard")
        high_severity = sum(1 for r in recommendations if r.get('severity', 'medium') == 'high')
        medium_severity = sum(1 for r in recommendations if r.get('severity', 'medium') == 'medium')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Issues", len(recommendations))
        with col2:
            st.metric("High Severity", high_severity)
        with col3:
            st.metric("Medium Severity", medium_severity)

        # Apply All Recommended Actions
        if st.button("🚀 Apply All Recommended Actions", help="Add all recommended actions to the pipeline"):
            pipeline_steps = []
            for rec in recommendations:
                if rec['type'] == 'missing_data':
                    strategy = 'mean' if rec['column'] in dtype_split(df)[0] else 'mode'
                    pipeline_steps.append({"kind": "impute", "params": {"columns": [rec['column']], "strategy": strategy}})
                elif rec['type'] == 'outliers':
                    pipeline_steps.append({"kind": "outliers", "params": {"columns": [rec['column']], "method": "iqr", "factor": 1.5}})
                elif rec['type'] in ['bias_risk', 'class_imbalance']:
                    pipeline_steps.append({"kind": "rebalance", "params": {"target": col1, "method": "oversample", "ratio": 1.0}})
                elif rec['type'] == 'duplicates':
                    pipeline_steps.append({"kind": "duplicates", "params": {"subset": None, "keep": "first"}})
                elif rec['type'] == 'high_cardinality':
                    pipeline_steps.append({"kind": "encode", "params": {"columns": [rec['column']], "method": "frequency_encode"}})
                elif rec['type'] == 'constant_category':
                    pipeline_steps.append({"kind": "drop_missing", "params": {"axis": "columns", "columns": [rec['column']]}})
                elif rec['type'] == 'id_leakage':
                    pipeline_steps.append({"kind": "drop_missing", "params": {"axis": "columns", "columns": [rec['column']]}})
                elif rec['type'] == 'rare_categories':
                    pipeline_steps.append({"kind": "encode", "params": {"columns": [rec['column']], "method": "onehot", "max_categories": 10}})
                elif rec['type'] == 'numeric_to_categorical':
                    pipeline_steps.append({"kind": "type_convert", "params": {"column": rec['column'], "type": "category"}})
                elif rec['type'] == 'high_skew':
                    pipeline_steps.append({"kind": "outliers", "params": {"columns": [rec['column']], "method": "iqr", "factor": 1.5}})
                elif rec['type'] == 'boolean_disguised':
                    pipeline_steps.append({"kind": "type_convert", "params": {"column": rec['column'], "type": "bool"}})
                elif rec['type'] == 'wide_numeric_range':
                    pipeline_steps.append({"kind": "scale", "params": {"columns": [rec['column']], "method": "standard"}})
            with session_lock:
                st.session_state.pipeline.extend(pipeline_steps)
            st.success(f"Added {len(pipeline_steps)} recommended actions to the pipeline.")

        # Tabbed Layout for Recommendations
        tabs = st.tabs(["All", "High Priority", "Medium Priority", "Low Priority"])
        priority_filters = {
            "All": (0.0, 1.0),
            "High Priority": (0.8, 1.0),
            "Medium Priority": (0.6, 0.8),
            "Low Priority": (0.0, 0.6)
        }

        key_counter = st.session_state.get("recommendation_key_counter", 0)
        for tab_name, (min_priority, max_priority) in priority_filters.items():
            with tabs[list(priority_filters.keys()).index(tab_name)]:
                filtered_recs = [r for r in recommendations if min_priority <= r.get('priority', 0.5) <= max_priority]
                if not filtered_recs:
                    st.info(f"No {tab_name.lower()} recommendations.")
                    continue
                st.markdown(f"**{tab_name} Recommendations ({len(filtered_recs)})**")
                for rec in filtered_recs:
                    key_suffix = f"{tab_name.replace(' ', '_')}_{rec['type']}_{rec.get('column', '')}_{key_counter}"
                    with st.expander(f"{rec['type'].replace('_', ' ').title()} (Priority: {rec.get('priority', 0.5):.2f})"):
                        st.markdown(f"**Issue**: {rec['type'].replace('_', ' ').title()}")
                        st.markdown(f"**Suggestion**: {rec['suggestion']}")
                        if 'column' in rec:
                            st.markdown(f"**Column**: `{rec['column']}`")
                        if 'missing_count' in rec:
                            st.markdown(f"**Missing Count**: {rec['missing_count']} ({rec['missing_ratio']*100:.1f}%)")
                        if 'count' in rec:
                            st.markdown(f"**Count**: {rec['count']}")
                        if 'unique_count' in rec:
                            st.markdown(f"**Unique Values**: {rec['unique_count']}")
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
                                    with session_lock:
                                        st.session_state.pipeline.append(step)
                                    st.success("Added imputation step to pipeline.")
                            with col2:
                                if st.button(f"📦 Drop Column", key=f"drop_{key_suffix}", help="Remove the column with missing values."):
                                    step = {"kind": "drop_missing", "params": {"axis": "columns", "columns": [rec['column']]}}
                                    with session_lock:
                                        st.session_state.pipeline.append(step)
                                    st.success("Added drop column step to pipeline.")
                        elif rec['type'] in ['outliers', 'high_skew']:
                            with col1:
                                if st.button(f"📦 Handle Outliers/Skew", key=f"outliers_{key_suffix}", help="Cap outliers or apply log transformation."):
                                    step = {"kind": "outliers", "params": {"columns": [rec['column']], "method": "iqr", "factor": 1.5}}
                                    with session_lock:
                                        st.session_state.pipeline.append(step)
                                    st.success("Added outlier/skew handling step to pipeline.")
                        elif rec['type'] in ['bias_risk', 'class_imbalance']:
                            with col1:
                                if st.button(f"📦 Rebalance", key=f"rebalance_{key_suffix}", help="Oversample minority classes to balance the dataset."):
                                    step = {"kind": "rebalance", "params": {"target": rec['column'], "method": "oversample", "ratio": 1.0}}
                                    with session_lock:
                                        st.session_state.pipeline.append(step)
                                    st.success("Added rebalancing step to pipeline.")
                        elif rec['type'] == 'duplicates':
                            with col1:
                                if st.button(f"📦 Remove Duplicates", key=f"duplicates_{key_suffix}", help="Remove duplicate rows, keeping the first occurrence."):
                                    step = {"kind": "duplicates", "params": {"subset": None, "keep": "first"}}
                                    with session_lock:
                                        st.session_state.pipeline.append(step)
                                    st.success("Added duplicate removal step to pipeline.")
                        elif rec['type'] == 'high_cardinality':
                            with col1:
                                if st.button(f"📦 Frequency Encode", key=f"freq_encode_{key_suffix}", help="Apply frequency encoding to high-cardinality column."):
                                    step = {"kind": "encode", "params": {"columns": [rec['column']], "method": "frequency_encode"}}
                                    with session_lock:
                                        st.session_state.pipeline.append(step)
                                    st.success("Added frequency encoding step to pipeline.")
                            with col2:
                                if st.button(f"📦 Hashing Encode", key=f"hash_encode_{key_suffix}", help="Apply hashing encoding to high-cardinality column."):
                                    step = {"kind": "encode", "params": {"columns": [rec['column']], "method": "hashing_encode", "n_components": 8}}
                                    with session_lock:
                                        st.session_state.pipeline.append(step)
                                    st.success("Added hashing encoding step to pipeline.")
                        elif rec['type'] == 'constant_category':
                            with col1:
                                if st.button(f"📦 Drop Column", key=f"drop_const_{key_suffix}", help="Remove constant or quasi-constant column."):
                                    step = {"kind": "drop_missing", "params": {"axis": "columns", "columns": [rec['column']]}}
                                    with session_lock:
                                        st.session_state.pipeline.append(step)
                                    st.success("Added drop column step to pipeline.")
                        elif rec['type'] == 'id_leakage':
                            with col1:
                                if st.button(f"📦 Drop Column", key=f"drop_id_{key_suffix}", help="Remove ID-like column to avoid leakage."):
                                    step = {"kind": "drop_missing", "params": {"axis": "columns", "columns": [rec['column']]}}
                                    with session_lock:
                                        st.session_state.pipeline.append(step)
                                    st.success("Added drop column step to pipeline.")
                        elif rec['type'] == 'rare_categories':
                            with col1:
                                if st.button(f"📦 Group Rare Categories", key=f"encode_rare_{key_suffix}", help="Group rare categories into 'Rare' bucket."):
                                    step = {"kind": "encode", "params": {"columns": [rec['column']], "method": "onehot", "max_categories": 10}}
                                    with session_lock:
                                        st.session_state.pipeline.append(step)
                                    st.success("Added encoding step with rare category grouping to pipeline.")
                        elif rec['type'] == 'numeric_to_categorical':
                            with col1:
                                if st.button(f"📦 Cast to Categorical", key=f"cast_cat_{key_suffix}", help="Convert numeric to categorical type."):
                                    step = {"kind": "type_convert", "params": {"column": rec['column'], "type": "category"}}
                                    with session_lock:
                                        st.session_state.pipeline.append(step)
                                    st.success("Added type conversion to categorical step to pipeline.")
                        elif rec['type'] == 'boolean_disguised':
                            with col1:
                                if st.button(f"📦 Cast to Boolean", key=f"cast_bool_{key_suffix}", help="Convert 0/1 integer to boolean."):
                                    step = {"kind": "type_convert", "params": {"column": rec['column'], "type": "bool"}}
                                    with session_lock:
                                        st.session_state.pipeline.append(step)
                                    st.success("Added type conversion to boolean step to pipeline.")
                        elif rec['type'] == 'wide_numeric_range':
                            with col1:
                                if st.button(f"📦 Scale Column", key=f"scale_{key_suffix}", help="Apply standard scaling to normalize range."):
                                    step = {"kind": "scale", "params": {"columns": [rec['column']], "method": "standard"}}
                                    with session_lock:
                                        st.session_state.pipeline.append(step)
                                    st.success("Added scaling step to pipeline.")
                    key_counter += 1
                with session_lock:
                    st.session_state["recommendation_key_counter"] = key_counter

        if st.button("🔄 Clear Recommendations", help="Reset recommendations and clear preview"):
            with session_lock:
                st.session_state.last_preview = None
                st.session_state["recommendation_key_counter"] = key_counter + 1
            st.rerun()

        st.markdown("---")
        st.caption("Recommendations are generated based on automatic data analysis. Each recommendation includes actionable steps to improve data quality.")

    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        st.error(f"Error generating recommendations: {e}")
