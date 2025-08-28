import streamlit as st
import pandas as pd
import dask.dataframe as dd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from dask_ml.preprocessing import PolynomialFeatures as DaskPolynomialFeatures
from sklearn.feature_selection import SelectKBest, mutual_info_regression, chi2
from sklearn.decomposition import PCA
import featuretools as ft
import altair as alt
import logging
from typing import List, Tuple, Dict, Any
from datetime import datetime
import threading
import uuid
import numexpr as ne
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

# Placeholder implementations for external utilities
def dtype_split(df: pd.DataFrame | dd.DataFrame) -> Tuple[List[str], List[str]]:
    """Split columns into numeric and non-numeric."""
    num_cols = [col for col in df.columns if is_numeric_dtype(df[col])]
    cat_cols = [col for col in df.columns if col not in num_cols]
    return num_cols, cat_cols

def sample_for_preview(df: pd.DataFrame | dd.DataFrame, n: int = 1000) -> pd.DataFrame:
    """Sample DataFrame for preview."""
    if isinstance(df, dd.DataFrame):
        return df.sample(frac=n / df.shape[0].compute()).compute()
    return df.sample(n=min(n, len(df)))

def alt_histogram(series: pd.Series, title: str) -> alt.Chart:
    """Create a histogram using Altair."""
    return alt.Chart(pd.DataFrame({title: series})).mark_bar().encode(
        x=alt.X(f"{title}:Q", bin=True),
        y='count()'
    )

def compute_basic_stats(df: pd.DataFrame | dd.DataFrame) -> Dict[str, Any]:
    """Compute basic statistics."""
    if isinstance(df, dd.DataFrame):
        df = df.compute()
    return {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'columns': list(df.columns)
    }

def compare_stats(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """Compare stats before and after."""
    return {
        'n_columns_before': before['n_columns'],
        'n_columns_after': after['n_columns'],
        'added_columns': [c for c in after['columns'] if c not in before['columns']],
        'removed_columns': [c for c in before['columns'] if c not in after['columns']]
    }

def push_history(message: str):
    """Push message to session state history."""
    with threading.Lock():
        if 'history' not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append({'message': message, 'timestamp': datetime.now().isoformat()})

def validate_step_function(func):
    """Validate a step function (placeholder)."""
    return func

logger = logging.getLogger(__name__)

# Thread lock for session state updates
session_lock = threading.Lock()

# Local feature engineering step registry
FEATURE_STEP_REGISTRY = {
    "create_polynomial_features": {
        "func": validate_step_function(lambda df, **kwargs: create_polynomial_features(df, **kwargs)),
        "depends_on": []
    },
    "extract_datetime_features": {
        "func": validate_step_function(lambda df, **kwargs: extract_datetime_features(df, **kwargs)),
        "depends_on": []
    },
    "bin_features": {
        "func": validate_step_function(lambda df, **kwargs: bin_features(df, **kwargs)),
        "depends_on": []
    },
    "select_features_correlation": {
        "func": validate_step_function(lambda df, **kwargs: select_features_correlation(df, **kwargs)),
        "depends_on": []
    },
    "automated_feature_engineering": {
        "func": validate_step_function(lambda df, **kwargs: automated_feature_engineering(df, **kwargs)),
        "depends_on": []
    }
}

def create_polynomial_features(df: pd.DataFrame | dd.DataFrame, columns: List[str], degree: int = 2, preview: bool = False) -> Tuple[pd.DataFrame | dd.DataFrame, str]:
    """Create polynomial and interaction features for specified numeric columns."""
    try:
        if not isinstance(df, (pd.DataFrame, dd.DataFrame)):
            return df, "Invalid DataFrame input"
        
        # Validate columns
        columns = [c for c in columns if c in df.columns and is_numeric_dtype(df[c])]
        if not columns:
            return df, "No valid numeric columns selected for polynomial features"
        
        # Check for duplicate column names
        if len(set(df.columns)) != len(df.columns):
            return df, "Duplicate column names detected; please resolve before processing"
        
        # Sanitize degree
        degree = max(1, min(5, int(degree)))
        
        df_out = df if preview else df.copy()
        
        with st.spinner(f"Generating polynomial features (degree={degree})..."):
            if isinstance(df_out, dd.DataFrame):
                poly = DaskPolynomialFeatures(degree=degree, include_bias=False)
                poly_features = poly.fit_transform(df_out[columns])
                feature_names = poly.get_feature_names_out(columns)
                poly_df = dd.from_array(poly_features, columns=feature_names, index=df_out.index)
                df_out = df_out.join(poly_df)
            else:
                poly = PolynomialFeatures(degree=degree, include_bias=False)
                poly_features = poly.fit_transform(df_out[columns])
                feature_names = poly.get_feature_names_out(columns)
                poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df_out.index)
                df_out = df_out.join(poly_df)
        
        msg = f"Created polynomial features (degree={degree}) for columns: {', '.join(columns)}"
        logger.info(msg)
        if not preview:
            with session_lock:
                st.session_state.pipeline.append({"kind": "create_polynomial_features", "params": {"columns": columns, "degree": degree}})
        return df_out, msg
    except ValueError as e:
        logger.error(f"ValueError in create_polynomial_features: {str(e)}")
        return df, f"Error creating polynomial features: {str(e)}"
    except MemoryError:
        logger.error("MemoryError in create_polynomial_features: Dataset too large")
        return df, "Error: Dataset too large for polynomial feature creation"
    except Exception as e:
        logger.error(f"Unexpected error in create_polynomial_features: {str(e)}")
        return df, f"Unexpected error: {str(e)}"

def extract_datetime_features(df: pd.DataFrame | dd.DataFrame, columns: List[str], features: List[str], preview: bool = False) -> Tuple[pd.DataFrame | dd.DataFrame, str]:
    """Extract datetime features (e.g., year, month, day) from specified columns."""
    try:
        if not isinstance(df, (pd.DataFrame, dd.DataFrame)):
            return df, "Invalid DataFrame input"
        
        # Validate columns
        columns = [c for c in columns if c in df.columns]
        if not columns:
            return df, "No valid columns selected for datetime features"
        
        # Validate datetime compatibility
        for col in columns:
            if not is_datetime64_any_dtype(df[col]):
                sample = df[col].head(5) if isinstance(df, dd.DataFrame) else df[col][:5]
                try:
                    pd.to_datetime(sample, errors='raise')
                except ValueError:
                    return df, f"Column {col} contains invalid datetime values"
        
        valid_features = ["year", "month", "day", "hour", "minute", "second", "dayofweek", "quarter"]
        features = [f for f in features if f in valid_features]
        if not features:
            return df, "No valid datetime features selected"
        
        df_out = df if preview else df.copy()
        invalid_counts = {}
        
        with st.spinner(f"Extracting datetime features ({', '.join(features)})"):
            for col in columns:
                if not is_datetime64_any_dtype(df_out[col]):
                    if isinstance(df_out, dd.DataFrame):
                        df_out[col] = dd.to_datetime(df_out[col], errors='coerce')
                    else:
                        df_out[col] = pd.to_datetime(df_out[col], errors='coerce')
                
                # Count invalid datetimes
                invalid_count = df_out[col].isna().sum()
                if isinstance(df_out, dd.DataFrame):
                    invalid_count = invalid_count.compute()
                if invalid_count > 0:
                    invalid_counts[col] = invalid_count
                
                for feature in features:
                    new_col = f"{col}_{feature}"
                    if isinstance(df_out, dd.DataFrame):
                        df_out[new_col] = getattr(df_out[col].dt, feature)
                    else:
                        df_out[new_col] = getattr(df_out[col].dt, feature)
        
        msg = f"Extracted datetime features ({', '.join(features)}) for columns: {', '.join(columns)}"
        if invalid_counts:
            msg += f". Warning: Invalid datetimes found in {', '.join(f'{col} ({count} invalid)' for col, count in invalid_counts.items())}"
        logger.info(msg)
        if not preview:
            with session_lock:
                st.session_state.pipeline.append({"kind": "extract_datetime_features", "params": {"columns": columns, "features": features}})
        return df_out, msg
    except ValueError as e:
        logger.error(f"ValueError in extract_datetime_features: {str(e)}")
        return df, f"Error extracting datetime features: {str(e)}"
    except MemoryError:
        logger.error("MemoryError in extract_datetime_features: Dataset too large")
        return df, "Error: Dataset too large for datetime extraction"
    except Exception as e:
        logger.error(f"Unexpected error in extract_datetime_features: {str(e)}")
        return df, f"Unexpected error: {str(e)}"

def bin_features(df: pd.DataFrame | dd.DataFrame, columns: List[str], bins: int = 10, preview: bool = False) -> Tuple[pd.DataFrame | dd.DataFrame, str]:
    """Bin numeric features into discrete intervals."""
    try:
        if not isinstance(df, (pd.DataFrame, dd.DataFrame)):
            return df, "Invalid DataFrame input"
        
        # Validate columns
        columns = [c for c in columns if c in df.columns and is_numeric_dtype(df[c])]
        if not columns:
            return df, "No valid numeric columns selected for binning"
        
        # Check for sufficient unique values
        for col in columns:
            unique_count = df[col].nunique()
            if isinstance(df, dd.DataFrame):
                unique_count = unique_count.compute()
            if unique_count < 2:
                return df, f"Column {col} has insufficient unique values for binning"
        
        bins = max(2, min(50, int(bins)))  # Sanitize bins
        
        df_out = df if preview else df.copy()
        
        with st.spinner(f"Binning columns into {bins} bins"):
            for col in columns:
                new_col = f"{col}_binned"
                try:
                    if isinstance(df_out, dd.DataFrame):
                        # Use Dask's qcut equivalent
                        bins_edges = df_out[col].quantile(np.linspace(0, 1, bins + 1)).compute()
                        df_out[new_col] = df_out[col].map_partitions(
                            lambda s: pd.cut(s, bins=bins_edges, labels=False, include_lowest=True),
                            meta=(new_col, 'int64')
                        )
                    else:
                        df_out[new_col] = pd.qcut(df_out[col], q=bins, duplicates='drop')
                except ValueError as e:
                    logger.warning(f"Skipping binning for {col}: {str(e)}")
                    msg = f"Error binning column {col}: {str(e)}"
                    return df_out, msg
        
        msg = f"Binned columns ({bins} bins): {', '.join(columns)}"
        logger.info(msg)
        if not preview:
            with session_lock:
                st.session_state.pipeline.append({"kind": "bin_features", "params": {"columns": columns, "bins": bins}})
        return df_out, msg
    except ValueError as e:
        logger.error(f"ValueError in bin_features: {str(e)}")
        return df, f"Error binning features: {str(e)}"
    except MemoryError:
        logger.error("MemoryError in bin_features: Dataset too large")
        return df, "Error: Dataset too large for binning"
    except Exception as e:
        logger.error(f"Unexpected error in bin_features: {str(e)}")
        return df, f"Unexpected error: {str(e)}"

def select_features_correlation(df: pd.DataFrame | dd.DataFrame, threshold: float = 0.8, preview: bool = False) -> Tuple[pd.DataFrame | dd.DataFrame, str]:
    """Select features based on correlation analysis."""
    try:
        if not isinstance(df, (pd.DataFrame, dd.DataFrame)):
            return df, "Invalid DataFrame input"
        
        threshold = max(0.1, min(0.9, float(threshold)))  # Sanitize threshold
        
        num_cols, _ = dtype_split(df)
        if not num_cols:
            return df, "No numeric columns available for correlation analysis"
        
        # Check for constant columns
        for col in num_cols:
            unique_count = df[col].nunique()
            if isinstance(df, dd.DataFrame):
                unique_count = unique_count.compute()
            if unique_count <= 1:
                num_cols.remove(col)
                logger.warning(f"Removed constant column {col} from correlation analysis")
        
        if not num_cols:
            return df, "No valid numeric columns after removing constants"
        
        with st.spinner("Computing correlation matrix..."):
            if isinstance(df, dd.DataFrame):
                corr_matrix = df[num_cols].corr()
            else:
                corr_matrix = df[num_cols].corr()
        
        # Handle NaN values in correlation matrix
        corr_matrix = corr_matrix.fillna(0)
        
        # Find highly correlated pairs
        corr_matrix = corr_matrix.abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        df_out = df if preview else df.copy()
        if to_drop:
            df_out = df_out.drop(columns=to_drop)
            msg = f"Dropped highly correlated columns (threshold={threshold}): {', '.join(to_drop)}"
        else:
            msg = f"No columns dropped (correlation threshold={threshold})"
        
        logger.info(msg)
        if not preview:
            with session_lock:
                st.session_state.pipeline.append({"kind": "select_features_correlation", "params": {"threshold": threshold}})
        return df_out, msg
    except ValueError as e:
        logger.error(f"ValueError in select_features_correlation: {str(e)}")
        return df, f"Error in correlation-based feature selection: {str(e)}"
    except MemoryError:
        logger.error("MemoryError in select_features_correlation: Dataset too large")
        return df, "Error: Dataset too large for correlation analysis"
    except Exception as e:
        logger.error(f"Unexpected error in select_features_correlation: {str(e)}")
        return df, f"Unexpected error: {str(e)}"

def automated_feature_engineering(df: pd.DataFrame | dd.DataFrame, max_features: int = 50, preview: bool = False) -> Tuple[pd.DataFrame | dd.DataFrame, str]:
    """Generate automated features using featuretools."""
    try:
        if not isinstance(df, (pd.DataFrame, dd.DataFrame)):
            return df, "Invalid DataFrame input"
        
        max_features = max(10, min(200, int(max_features)))  # Sanitize max_features
        
        # Validate index and column names
        if len(set(df.columns)) != len(df.columns):
            return df, "Duplicate column names detected; please resolve before processing"
        if 'index' not in df.index.name and not df.index.is_unique:
            return df, "DataFrame must have a unique index for featuretools"
        
        with st.spinner(f"Generating up to {max_features} automated features..."):
            if isinstance(df, dd.DataFrame):
                # Sample for large datasets
                df_sample = sample_for_preview(df, n=10000)
            else:
                df_sample = df
            
            es = ft.EntitySet(id="dataset")
            es = es.add_dataframe(dataframe_name="data", dataframe=df_sample, index="index")
            feature_matrix, feature_defs = ft.dfs(
                entityset=es,
                target_dataframe_name="data",
                max_depth=2,
                max_features=max_features,
                agg_primitives=['sum', 'mean', 'count'],
                trans_primitives=['add_numeric', 'multiply_numeric']
            )
        
        df_out = feature_matrix if not preview else df
        msg = f"Generated {len(feature_defs)} automated features using featuretools"
        logger.info(msg)
        if not preview:
            with session_lock:
                st.session_state.pipeline.append({"kind": "automated_feature_engineering", "params": {"max_features": max_features}})
        return df_out, msg
    except ValueError as e:
        logger.error(f"ValueError in automated_feature_engineering: {str(e)}")
        return df, f"Error in automated feature engineering: {str(e)}"
    except MemoryError:
        logger.error("MemoryError in automated_feature_engineering: Dataset too large")
        return df, "Error: Dataset too large for automated feature engineering"
    except Exception as e:
        logger.error(f"Unexpected error in automated_feature_engineering: {str(e)}")
        return df, f"Unexpected error: {str(e)}"

def plot_correlation_heatmap(df: pd.DataFrame | dd.DataFrame):
    """Plot a correlation heatmap for numeric columns."""
    try:
        num_cols, _ = dtype_split(df)
        if not num_cols:
            st.warning("No numeric columns available for correlation heatmap")
            return
        
        # Limit to top 20 columns for performance
        num_cols = num_cols[:20]
        
        with st.spinner("Computing correlation heatmap..."):
            if isinstance(df, dd.DataFrame):
                corr_matrix = df[num_cols].corr().compute()
            else:
                corr_matrix = df[num_cols].corr()
            
            # Handle NaN values
            corr_matrix = corr_matrix.fillna(0)
            
            chart = alt.Chart(corr_matrix.reset_index().melt(id_vars=['index'])).mark_rect().encode(
                x=alt.X('index:O', title=''),
                y=alt.Y('variable:O', title=''),
                color=alt.Color('value:Q', scale=alt.Scale(scheme='redblue', domain=[-1, 1])),
                tooltip=['index', 'variable', 'value']
            ).properties(
                title='Correlation Heatmap',
                width=400,
                height=400
            )
            st.altair_chart(chart, use_container_width=True)
    except ValueError as e:
        logger.error(f"ValueError in plot_correlation_heatmap: {str(e)}")
        st.error(f"Error plotting correlation heatmap: {str(e)}")
    except MemoryError:
        logger.error("MemoryError in plot_correlation_heatmap: Dataset too large")
        st.error("Error: Dataset too large for correlation heatmap")
    except Exception as e:
        logger.error(f"Unexpected error in plot_correlation_heatmap: {str(e)}")
        st.error(f"Unexpected error: {str(e)}")

def safe_eval_expression(df: pd.DataFrame | dd.DataFrame, expression: str, new_col: str) -> Tuple[pd.Series, str]:
    """Safely evaluate a custom expression using numexpr."""
    try:
        # Validate column references
        valid_cols = [col for col in df.columns if col in expression]
        if not valid_cols:
            return None, "Expression must reference valid column names"
        
        # Restrict to basic operations
        allowed_ops = {'+', '-', '*', '/', 'log', 'exp', 'sin', 'cos', 'sqrt'}
        if not any(op in expression for op in allowed_ops):
            return None, "Expression must include at least one valid operation (+, -, *, /, log, exp, sin, cos, sqrt)"
        
        if isinstance(df, dd.DataFrame):
            result = df.compute().eval(expression, engine='numexpr')
        else:
            result = df.eval(expression, engine='numexpr')
        return result, "Expression evaluated successfully"
    except Exception as e:
        return None, f"Error evaluating expression: {str(e)}"

def section_feature_engineering():
    """Feature Engineering section with sub-tabs for different operations."""
    if st.session_state.get('df') is None:
        st.warning("Please upload a dataset in the Upload section first.")
        return
    
    # Initialize session state
    with session_lock:
        if 'pipeline' not in st.session_state:
            st.session_state.pipeline = []
        if 'history' not in st.session_state:
            st.session_state.history = []
    
    st.header("🎨 Feature Engineering Studio")
    st.markdown("Create, transform, and select features to uncover hidden patterns in your data.")
    
    tabs = st.tabs([
        "🖌️ Feature Creation",
        "🔮 Feature Transformation",
        "✂️ Feature Selection",
        "🤖 Automated Feature Engineering",
        "🧭 Feature Evaluation"
    ])
    
    df = st.session_state.df
    before_stats = compute_basic_stats(df)
    
    with tabs[0]:  # Feature Creation
        st.subheader("🖌️ Feature Creation")
        st.markdown("Generate new features from existing data.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Polynomial Features**")
            poly_cols = st.multiselect("Select numeric columns for polynomial features", 
                                     [c for c in df.columns if is_numeric_dtype(df[c])], 
                                     key="poly_cols")
            poly_degree = st.slider("Polynomial degree", 1, 5, 2, key="poly_degree")
            if st.button("Apply Polynomial Features"):
                with st.spinner("Applying polynomial features..."):
                    preview_df, preview_msg = create_polynomial_features(df, poly_cols, poly_degree, preview=True)
                    if "Error" not in preview_msg:
                        st.write("Preview of transformed DataFrame:")
                        st.write(sample_for_preview(preview_df))
                        if st.button("Confirm Polynomial Features"):
                            df, msg = create_polynomial_features(df, poly_cols, poly_degree)
                            if "Error" not in msg:
                                with session_lock:
                                    st.session_state.df = df
                                    push_history(f"Applied polynomial features (degree={poly_degree})")
                                st.success(msg)
                            else:
                                st.error(msg)
                    else:
                        st.error(preview_msg)
        
        with col2:
            st.markdown("**Datetime Features**")
            dt_cols = st.multiselect("Select datetime columns", df.columns, key="dt_cols")
            dt_features = st.multiselect("Select features to extract", 
                                      ["year", "month", "day", "hour", "minute", "second", "dayofweek", "quarter"], 
                                      key="dt_features")
            if st.button("Extract Datetime Features"):
                with st.spinner("Extracting datetime features..."):
                    preview_df, preview_msg = extract_datetime_features(df, dt_cols, dt_features, preview=True)
                    if "Error" not in preview_msg:
                        st.write("Preview of transformed DataFrame:")
                        st.write(sample_for_preview(preview_df))
                        if st.button("Confirm Datetime Features"):
                            df, msg = extract_datetime_features(df, dt_cols, dt_features)
                            if "Error" not in msg:
                                with session_lock:
                                    st.session_state.df = df
                                    push_history(f"Extracted datetime features: {', '.join(dt_features)}")
                                st.success(msg)
                            else:
                                st.error(msg)
                    else:
                        st.error(preview_msg)
    
    with tabs[1]:  # Feature Transformation
        st.subheader("🔮 Feature Transformation")
        st.markdown("Transform features to enhance model performance.")
        
        st.markdown("**Binning**")
        bin_cols = st.multiselect("Select numeric columns to bin", 
                                [c for c in df.columns if is_numeric_dtype(df[c])], 
                                key="bin_cols")
        bins = st.slider("Number of bins", 2, 50, 10, key="bins")
        if st.button("Apply Binning"):
            with st.spinner("Applying binning..."):
                preview_df, preview_msg = bin_features(df, bin_cols, bins, preview=True)
                if "Error" not in preview_msg:
                    st.write("Preview of transformed DataFrame:")
                    st.write(sample_for_preview(preview_df))
                    if st.button("Confirm Binning"):
                        df, msg = bin_features(df, bin_cols, bins)
                        if "Error" not in msg:
                            with session_lock:
                                st.session_state.df = df
                                push_history(f"Binned columns into {bins} bins")
                            st.success(msg)
                        else:
                            st.error(msg)
                else:
                    st.error(preview_msg)
    
    with tabs[2]:  # Feature Selection
        st.subheader("✂️ Feature Selection")
        st.markdown("Select the most relevant features to reduce dimensionality.")
        
        st.markdown("**Correlation-based Selection**")
        corr_threshold = st.slider("Correlation threshold", 0.1, 0.9, 0.8, step=0.05, key="corr_threshold")
        if st.button("Apply Correlation-based Selection"):
            with st.spinner("Applying correlation-based selection..."):
                preview_df, preview_msg = select_features_correlation(df, corr_threshold, preview=True)
                if "Error" not in preview_msg:
                    st.write("Preview of transformed DataFrame:")
                    st.write(sample_for_preview(preview_df))
                    if st.button("Confirm Correlation Selection"):
                        df, msg = select_features_correlation(df, corr_threshold)
                        if "Error" not in msg:
                            with session_lock:
                                st.session_state.df = df
                                push_history(f"Selected features based on correlation (threshold={corr_threshold})")
                            st.success(msg)
                        else:
                            st.error(msg)
                else:
                    st.error(preview_msg)
    
    with tabs[3]:  # Automated Feature Engineering
        st.subheader("🤖 Automated Feature Engineering")
        st.markdown("Generate candidate features automatically using featuretools.")
        
        max_features = st.slider("Maximum number of features to generate", 10, 200, 50, key="max_features")
        if st.button("Generate Automated Features"):
            with st.spinner("Generating automated features..."):
                preview_df, preview_msg = automated_feature_engineering(df, max_features, preview=True)
                if "Error" not in preview_msg:
                    st.write("Preview of transformed DataFrame:")
                    st.write(sample_for_preview(preview_df))
                    if st.button("Confirm Automated Features"):
                        df, msg = automated_feature_engineering(df, max_features)
                        if "Error" not in msg:
                            with session_lock:
                                st.session_state.df = df
                                push_history(f"Generated {max_features} automated features")
                            st.success(msg)
                        else:
                            st.error(msg)
                else:
                    st.error(preview_msg)
    
    with tabs[4]:  # Feature Evaluation
        st.subheader("🧭 Feature Evaluation")
        st.markdown("Evaluate feature importance and relationships.")
        
        st.markdown("**Feature Palette Dashboard**")
        after_stats = compute_basic_stats(st.session_state.df)
        comparison = compare_stats(before_stats, after_stats)
        
        st.write(f"**Features Before**: {comparison['n_columns_before']}")
        st.write(f"**Features After**: {comparison['n_columns_after']}")
        st.write(f"**Added Features**: {', '.join(comparison['added_columns']) if comparison['added_columns'] else 'None'}")
        st.write(f"**Removed Features**: {', '.join(comparison['removed_columns']) if comparison['removed_columns'] else 'None'}")
        
        st.markdown("**Correlation Heatmap**")
        plot_correlation_heatmap(st.session_state.df)
        
        st.markdown("**Feature Sketchbook**")
        expression = st.text_input("Enter custom feature expression (e.g., 'col1 / log(col2 + 1)')", key="feature_sketch")
        if st.button("Test Feature Expression"):
            with st.spinner("Evaluating expression..."):
                new_col = f"custom_feature_{uuid.uuid4().hex[:8]}"
                result, msg = safe_eval_expression(df, expression, new_col)
                if result is not None:
                    st.write("Result preview:")
                    st.write(result.head())
                    if st.button("Add to Dataset"):
                        with session_lock:
                            st.session_state.df[new_col] = result
                            push_history(f"Added custom feature: {expression}")
                            st.success(f"Added custom feature as {new_col}")
                else:
                    st.error(msg)