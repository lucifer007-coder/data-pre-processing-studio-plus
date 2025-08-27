import streamlit as st
import pandas as pd
import dask.dataframe as dd
import altair as alt
import logging
import re
import sqlparse
import io
import numpy as np
from typing import List, Dict, Union
import scipy.stats as scipy_stats
from utils.stats_utils import compute_basic_stats
from utils.viz_utils import alt_histogram
from utils.data_utils import dtype_split, sample_for_preview
from components.sections.recommendations import PreprocessingRecommendations

# Optional dependencies with explicit checks
try:
    from st_aggrid import AgGrid
except ImportError:
    AgGrid = None
try:
    import duckdb
except ImportError:
    duckdb = None
try:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
except ImportError:
    sklearn = None

# Constants
CONFIG = {
    "BIAS_THRESHOLD": 0.8,
    "SAMPLE_SIZE": 1000,
    "CHART_WIDTH": 400,
    "CHART_HEIGHT": 300,
    "CORR_SAMPLE_SIZE": 1000,
    "PII_SAMPLE_SIZE": 1000,
    "MAX_COLS": 20,
    "MIN_VALID_ROWS": 3  # Minimum rows for reliable correlation
}

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_session_state():
    """Initialize session state with defaults if not set."""
    if "df" not in st.session_state:
        st.session_state.df = None
    if "raw_df" not in st.session_state:
        st.session_state.raw_df = None
    if "changelog" not in st.session_state:
        st.session_state.changelog = []

def detect_pii(df: Union[pd.DataFrame, dd.DataFrame]) -> List[tuple]:
    """
    Detect PII in DataFrame columns using regex patterns.
    
    Args:
        df: pandas or Dask DataFrame
    Returns:
        List of tuples containing (column, pii_type)
    """
    try:
        pii_patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b(\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b",
            "ssn": r"\b\d{3}-?\d{2}-?\d{4}\b",
            "credit_card": r"\b(?:\d{4}[ -]?){3}\d{4}\b"
        }
        pii_columns = []
        object_cols = df.select_dtypes(include=["object"]).columns
        total_rows = df.shape[0].compute() if isinstance(df, dd.DataFrame) else len(df)
        if not object_cols.empty and total_rows > 0:
            for col in object_cols:
                sample_size = min(CONFIG["PII_SAMPLE_SIZE"], total_rows)
                sample_df = (df[col].sample(n=sample_size, random_state=42).compute() 
                            if isinstance(df, dd.DataFrame) 
                            else df[col].head(sample_size))
                sample_df = sample_df.fillna('').astype(str)
                for pii_type, pattern in pii_patterns.items():
                    try:
                        if sample_df.str.contains(pattern, regex=True, na=False).any():
                            pii_columns.append((col, pii_type))
                    except re.error as e:
                        logger.error(f"Invalid regex pattern for {pii_type} in column {col}: {e}")
        return pii_columns
    except (ValueError, re.error) as e:
        logger.error(f"Error in PII detection: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in PII detection: {e}", exc_info=True)
        return []

@st.cache_data(hash_funcs={pd.DataFrame: lambda df: pd.util.hash_pandas_object(df.head(100)).sum(),
                          dd.DataFrame: lambda df: hash(tuple(df.columns) + tuple(df.head(10).values.flatten()))})
def cached_compute_basic_stats(df: Union[pd.DataFrame, dd.DataFrame]) -> Dict:
    """
    Compute basic statistics for the DataFrame with caching.
    
    Args:
        df: pandas or Dask DataFrame
    Returns:
        Dictionary containing statistics
    """
    try:
        return compute_basic_stats(df)
    except (ValueError, KeyError) as e:
        logger.error(f"Error computing basic stats: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error in compute_basic_stats: {e}", exc_info=True)
        return {}

def validate_sql_query(query: str) -> str:
    """
    Validate that the SQL query is a SELECT statement.
    
    Args:
        query: SQL query string
    Returns:
        Validated query or raises ValueError
    """
    try:
        parsed = sqlparse.parse(query.strip())
        if not parsed or parsed[0].get_type().upper() != "SELECT":
            raise ValueError("Only SELECT queries are allowed.")
        return query
    except Exception as e:
        logger.error(f"Invalid SQL query: {e}")
        raise ValueError(f"Invalid SQL query: {e}")

def fallback_sample(df: Union[pd.DataFrame, dd.DataFrame], sample_size: int = 1000) -> pd.DataFrame:
    """
    Fallback function to sample a DataFrame if sample_for_preview fails.
    
    Args:
        df: pandas or Dask DataFrame
        sample_size: Number of rows to sample
    Returns:
        pandas DataFrame with sampled rows
    """
    try:
        total_rows = df.shape[0].compute() if isinstance(df, dd.DataFrame) else len(df)
        sample_size = min(sample_size, total_rows)
        if isinstance(df, dd.DataFrame):
            return df.sample(frac=sample_size / total_rows, random_state=42).compute()
        return df.sample(n=sample_size, random_state=42) if total_rows > sample_size else df.copy()
    except Exception as e:
        logger.error(f"Error in fallback sampling: {e}", exc_info=True)
        raise

def section_dashboard():
    """Render the data dashboard with data exploration and visualization."""
    st.header("📊 Data Dashboard")
    initialize_session_state()
    logger.debug("Initializing dashboard session state")

    # Check if DataFrame exists
    if "df" not in st.session_state or st.session_state.df is None:
        st.error("No dataset uploaded. Please upload a valid CSV file.")
        logger.error("No DataFrame found in session state")
        return

    df = st.session_state.df
    logger.debug(f"DataFrame type: {type(df)}")

    # Validate DataFrame
    try:
        total_rows = df.shape[0].compute() if isinstance(df, dd.DataFrame) else len(df)
        total_cols = len(df.columns.compute()) if isinstance(df, dd.DataFrame) else len(df.columns)
        if total_rows == 0:
            st.error("The uploaded dataset has no rows. Please upload a non-empty CSV file.")
            logger.error("DataFrame has zero rows")
            return
        if total_cols == 0:
            st.error("The uploaded dataset has no columns. Please upload a valid CSV file.")
            logger.error("DataFrame has zero columns")
            return
        logger.info(f"Dashboard loaded with DataFrame: {total_rows} rows, {total_cols} columns")
    except (ValueError, AttributeError) as e:
        st.error(f"Error validating dataset: {e}. Please ensure the uploaded CSV is valid.")
        logger.error(f"Error validating DataFrame: {e}", exc_info=True)
        return

    # Cache sampled DataFrame
    try:
        try:
            # Try calling sample_for_preview without max_rows
            sampled_df = sample_for_preview(df)
            logger.debug(f"Sampled DataFrame shape (using sample_for_preview): {sampled_df.shape}")
        except TypeError as e:
            if "max_rows" in str(e):
                logger.warning("sample_for_preview does not support max_rows; using fallback sampling")
                sampled_df = fallback_sample(df, sample_size=CONFIG["SAMPLE_SIZE"])
            else:
                raise
        # Validate sampled DataFrame
        if sampled_df.empty or len(sampled_df.columns) == 0:
            st.error("Sampled DataFrame is empty or invalid. Please check the dataset.")
            logger.error("Sampled DataFrame is empty or invalid")
            return
    except Exception as e:
        st.error(f"Error sampling dataset: {e}. Using fallback sampling method.")
        logger.error(f"Error sampling DataFrame: {e}", exc_info=True)
        try:
            sampled_df = fallback_sample(df, sample_size=CONFIG["SAMPLE_SIZE"])
            logger.debug(f"Sampled DataFrame shape (fallback): {sampled_df.shape}")
            if sampled_df.empty or len(sampled_df.columns) == 0:
                st.error("Fallback sampled DataFrame is empty or invalid. Please check the dataset.")
                logger.error("Fallback sampled DataFrame is empty or invalid")
                return
        except Exception as e:
            st.error(f"Fallback sampling failed: {e}. Please check the dataset format.")
            logger.error(f"Fallback sampling failed: {e}", exc_info=True)
            return

    # Debug mode toggle
    debug_mode = st.checkbox("Enable debug mode", key="dashboard_debug")
    if debug_mode:
        st.subheader("Debug Information")
        st.write(f"DataFrame shape: {total_rows} rows, {total_cols} columns")
        st.write(f"Columns: {df.columns.tolist()}")
        st.write(f"Sampled DataFrame head:\n{sampled_df.head().to_markdown()}")

    try:
        # Compute statistics
        start_time = pd.Timestamp.now()
        df_stats = cached_compute_basic_stats(df)
        if not df_stats or df_stats.get("shape", (0, 0)) == (0, 0):
            st.error("Failed to compute dataset statistics. The dataset may be invalid or empty.")
            logger.error("Empty or invalid stats returned from compute_basic_stats")
            return
        logger.info(f"Stats computation took {(pd.Timestamp.now() - start_time).total_seconds():.2f} seconds")
        if debug_mode:
            st.json(df_stats)

        # PII Warning
        try:
            pii_columns = detect_pii(df)
            if pii_columns:
                st.warning(
                    f"Potential PII detected in columns: {', '.join([col for col, pii_type in pii_columns])}. "
                    "Consider masking sensitive data in the Inconsistency section."
                )
        except Exception as e:
            st.error(f"Error in PII detection: {e}")
            logger.error(f"Error in PII detection section: {e}", exc_info=True)

        # Data Quality Metrics
        try:
            st.subheader("Data Quality Metrics")
            completeness = (1 - df_stats['missing_total'] / (df_stats['shape'][0] * df_stats['shape'][1]) 
                          if df_stats['shape'][0] > 0 and df_stats['shape'][1] > 0 else 0)
            uniqueness = (1 - df_stats['duplicate_rows'] / df_stats['shape'][0] 
                         if df_stats['shape'][0] > 0 else 1)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Completeness", f"{completeness:.2%}")
            with c2:
                st.metric("Uniqueness", f"{uniqueness:.2%}")
            with c3:
                st.metric("Memory (MB)", f"{df_stats.get('memory_usage_mb', 0):.2f}")
        except (KeyError, ZeroDivisionError) as e:
            st.error(f"Error displaying data quality metrics: {e}")
            logger.error(f"Error in data quality metrics: {e}", exc_info=True)

        # Interactive Data Explorer
        try:
            st.subheader("Interactive Data Explorer")
            if AgGrid is None:
                st.warning("Interactive data explorer requires streamlit-aggrid. Install it to enable this feature.")
            else:
                AgGrid(sampled_df, height=400, fit_columns_on_grid_load=True, key="data_explorer")
            st.caption(f"Showing up to {CONFIG['SAMPLE_SIZE']} rows for performance.")
        except Exception as e:
            st.error(f"Error in interactive data explorer: {e}")
            logger.error(f"Error in interactive data explorer: {e}", exc_info=True)

        # Correlation Analysis
        st.subheader("Correlation Analysis")
        if "df" not in st.session_state or st.session_state.df is None:
            st.error("No valid dataset available. Please upload a dataset.")
        else:
            # Use sampled_df for Dask compatibility
            df = sampled_df if isinstance(st.session_state.df, dd.DataFrame) else st.session_state.df
            num_cols = df_stats.get("numeric_cols", [])
   
            if num_cols:
                try:
                    # Validate numeric columns
                    num_cols = [col for col in num_cols if df[col].dtype in ["float64", "int64"]]
                    if not num_cols:
                        st.info("No valid numeric columns available for correlation analysis.")
                    else:
                        # Limit columns to prevent memory issues
                        if len(num_cols) > CONFIG["MAX_COLS"]:
                            st.warning(f"Too many numeric columns ({len(num_cols)}). Please select up to {CONFIG['MAX_COLS']} columns to avoid performance issues.")
                            num_cols = num_cols[:CONFIG["MAX_COLS"]]
                        # Allow users to select columns
                        selected_cols = st.multiselect(
                            "Select numeric columns for correlation analysis",
                            num_cols,
                            default=num_cols[:min(len(num_cols), 5)],
                            help="Choose at least two numeric columns to compute correlations."
                        )
                        if len(selected_cols) < 2:
                            st.warning("Please select at least two numeric columns for correlation analysis.")
                        else:
                            # Select correlation method
                            corr_method = st.selectbox(
                                "Correlation method",
                                ["Pearson", "Spearman", "Kendall"],
                                help="Pearson measures linear relationships, Spearman measures monotonic relationships, and Kendall measures ordinal associations."
                            )
                            allowed_methods = ["pearson", "spearman", "kendall"]
                            if corr_method.lower() not in allowed_methods:
                                st.error(f"Invalid correlation method: {corr_method}")
                            else:
                                # Check for sufficient valid data
                                valid_cols = []
                                for col in selected_cols:
                                    valid_count = len(df[col].dropna())
                                    if valid_count >= CONFIG["MIN_VALID_ROWS"]:
                                        valid_cols.append(col)
                                    else:
                                        st.warning(f"Column '{col}' has too few valid values ({valid_count}) for correlation analysis.")
                                if len(valid_cols) < 2:
                                    st.error("Not enough columns with sufficient valid data for correlation analysis.")
                                else:
                                    # Cache correlation matrix computation
                                    @st.cache_data(hash_funcs={pd.DataFrame: lambda df: pd.util.hash_pandas_object(df).sum()})
                                    def compute_correlation_matrix(df, cols, method):
                                        return df[cols].corr(method=method)
                                    with st.spinner("Computing correlation matrix..."):
                                        corr_matrix = compute_correlation_matrix(df, valid_cols, corr_method.lower())
                                        # Replace NaN/None with 0 for visualization
                                        corr_matrix = corr_matrix.fillna(0)
                                    # Compute p-values efficiently
                                    p_matrix = pd.DataFrame(index=valid_cols, columns=valid_cols, dtype=float)
                                    sample_sizes = pd.Series(index=valid_cols, dtype=int)
                                    with st.spinner("Computing statistical significance..."):
                                        for col in valid_cols:
                                            sample_sizes[col] = len(df[col].dropna())
                                        for col1 in valid_cols:
                                            for col2 in valid_cols:
                                                if col1 != col2:
                                                    clean_data = df[[col1, col2]].dropna()
                                                    if len(clean_data) < CONFIG["MIN_VALID_ROWS"]:
                                                        p_matrix.loc[col1, col2] = 1.0
                                                        corr_matrix.loc[col1, col2] = 0
                                                        continue
                                                    if corr_method == "Pearson":
                                                        _, p_val = scipy_stats.pearsonr(clean_data[col1], clean_data[col2])
                                                    elif corr_method == "Spearman":
                                                        _, p_val = scipy_stats.spearmanr(clean_data[col1], clean_data[col2])
                                                    else:  # Kendall
                                                        _, p_val = scipy_stats.kendalltau(clean_data[col1], clean_data[col2])
                                                    p_matrix.loc[col1, col2] = p_val
                                                else:
                                                    p_matrix.loc[col1, col2] = 1.0
                                    # Prepare data for heatmap
                                    corr_data = (
                                        corr_matrix.stack()
                                        .reset_index()
                                        .rename(columns={0: "Correlation", "level_0": "Variable1", "level_1": "Variable2"})
                                    )
                                    p_data = p_matrix.stack().reset_index().rename(columns={0: "p-value", "level_0": "Variable1", "level_1": "Variable2"})
                                    corr_data = corr_data.merge(
                                        p_data[["Variable1", "Variable2", "p-value"]],
                                        on=["Variable1", "Variable2"],
                                        how="left"
                                    )
                                    # Replace NaN/None in Correlation and p-value columns
                                    corr_data["Correlation"] = corr_data["Correlation"].fillna(0).astype(float)
                                    corr_data["p-value"] = corr_data["p-value"].fillna(1.0).astype(float)
                                    # Correlation Heatmap
                                    chart = alt.Chart(corr_data).mark_rect().encode(
                                        x=alt.X("Variable1:N", title="", sort=valid_cols),
                                        y=alt.Y("Variable2:N", title="", sort=valid_cols),
                                        color=alt.Color(
                                            "Correlation:Q",
                                            scale=alt.Scale(scheme="viridis", domain=[-1, 1]),
                                            legend=alt.Legend(title="Correlation Coefficient")
                                        ),
                                        tooltip=[
                                            alt.Tooltip("Variable1:N", title="Variable 1"),
                                            alt.Tooltip("Variable2:N", title="Variable 2"),
                                            alt.Tooltip("Correlation:Q", title="Correlation", format=".3f"),
                                            alt.Tooltip("p-value:Q", title="p-value", format=".3f")
                                        ]
                                    ).properties(
                                        title=f"{corr_method} Correlation Heatmap",
                                        width=CONFIG["CHART_WIDTH"],
                                        height=CONFIG["CHART_HEIGHT"]
                                    )
                                    # Text annotations with dynamic color contrast
                                    text = chart.mark_text(baseline="middle").encode(
                                        text=alt.Text("Correlation:Q", format=".2f"),
                                        color=alt.condition(
                                            alt.expr.abs(alt.datum.Correlation) > 0.4,
                                            alt.value("black"),
                                            alt.value("white")
                                        )
                                    )
                                    st.altair_chart(chart + text, use_container_width=True)
                                    # Display significant correlations
                                    st.markdown("**Significant Correlations (p-value < 0.05)**")
                                    significant = corr_data[
                                        (corr_data["p-value"] < 0.05) &
                                        (corr_data["Variable1"] != corr_data["Variable2"]) &
                                        (corr_data["Correlation"].abs() >= 0.3) &
                                        (corr_data["Correlation"].notna())
                                    ]
                                    if not significant.empty:
                                        st.dataframe(
                                            significant[["Variable1", "Variable2", "Correlation", "p-value"]]
                                            .style.format({"Correlation": "{:.3f}", "p-value": "{:.3f}"})
                                            .set_caption("Correlations with |r| ≥ 0.3 and p < 0.05"),
                                            use_container_width=True
                                        )
                                    else:
                                        st.info("No significant correlations found with |r| ≥ 0.3 and p-value < 0.05.")
                                    # AI-driven insights
                                    st.markdown("**AI Insights**")
                                    if significant.empty:
                                        st.write("No strong correlations found. Consider exploring other variables, collecting more data, or checking for non-linear relationships.")
                                    else:
                                        strong_corrs = significant[significant["Correlation"].abs() >= 0.7]
                                        if not strong_corrs.empty:
                                            st.write("Strong correlations (|r| ≥ 0.7) detected. These may indicate important relationships. Consider regression analysis to explore potential causality.")
                                        else:
                                            st.write("Moderate correlations found. Review the significant correlations table to identify key relationships.")
                                        st.warning("Note: Correlation does not imply causation. Further analysis is recommended.")
                                    # Scatter plot for selected variable pair
                                    if st.checkbox("Show scatter plot for a variable pair", key="scatter_plot_toggle"):
                                        var_pairs = [(v1, v2) for v1 in valid_cols for v2 in valid_cols if v1 < v2]
                                        if var_pairs:
                                            var_pair = st.selectbox(
                                                "Select variable pair",
                                                var_pairs,
                                                format_func=lambda x: f"{x[0]} vs {x[1]}",
                                                key="scatter_pair_select"
                                            )
                                            scatter_data = df[[var_pair[0], var_pair[1]]].dropna()
                                            if len(scatter_data) < 2:
                                                st.warning(f"Not enough data for scatter plot of {var_pair[0]} vs {var_pair[1]}.")
                                            else:
                                                scatter_chart = alt.Chart(scatter_data).mark_circle(size=60).encode(
                                                    x=alt.X(var_pair[0], title=var_pair[0]),
                                                    y=alt.Y(var_pair[1], title=var_pair[1]),
                                                    tooltip=[var_pair[0], var_pair[1]]
                                                ).properties(
                                                    title=f"Scatter Plot: {var_pair[0]} vs {var_pair[1]}",
                                                    width=CONFIG["CHART_WIDTH"],
                                                    height=CONFIG["CHART_HEIGHT"]
                                                )
                                                st.altair_chart(scatter_chart, use_container_width=True)
                                        else:
                                            st.info("No valid variable pairs available for scatter plot.")
                                    # Export correlation matrix with PII warning
                                    st.markdown("**Export Correlation Matrix**")
                                    pii_columns = detect_pii(df)
                                    if any(col in valid_cols for col, _ in pii_columns):
                                        st.warning("Selected columns may contain sensitive data (e.g., PII). Ensure secure handling of exported files.")
                                    if st.button("Prepare Correlation Matrix for Download", key="prepare_corr_download"):
                                        buf = io.StringIO()
                                        corr_matrix.to_csv(buf, index=True)
                                        st.download_button(
                                            label="Download Correlation Matrix (CSV)",
                                            data=buf.getvalue(),
                                            file_name=f"{corr_method.lower()}_correlation_matrix.csv",
                                            mime="text/csv",
                                            help="Download the correlation matrix as a CSV file.",
                                            key="download_corr_matrix"
                                        )
                                        st.success("Correlation matrix ready for download.")
                except Exception as e:
                    st.error(f"Error in correlation analysis: {e}")
                    logger.error(f"Error in correlation analysis: {e}", exc_info=True)
            else:
                st.info("No numeric columns available for correlation analysis.")

        # Feature Importance
        try:
            st.subheader("Feature Importance")
            all_cols = df.columns.tolist()
            target_col = st.selectbox("Select target column for importance", ["(none)"] + all_cols, key="feature_importance_target")
            if target_col != "(none)":
                if sklearn is None:
                    st.warning("Feature importance requires scikit-learn. Install it to enable this feature.")
                else:
                    X = sampled_df.drop(columns=[target_col]).select_dtypes(include=['float64', 'int64'])
                    y = sampled_df[target_col]
                    if not X.empty:
                        mi_scores = (mutual_info_regression(X, y) if y.dtype in ['float64', 'int64'] 
                                    else mutual_info_classif(X, y))
                        mi_df = pd.DataFrame({'Feature': X.columns, 'Mutual Info': mi_scores}).sort_values(
                            by='Mutual Info', ascending=False
                        )
                        chart = alt.Chart(mi_df).mark_bar().encode(
                            x=alt.X('Mutual Info:Q', title='Mutual Information'),
                            y=alt.Y('Feature:N', sort='-x'),
                            tooltip=['Feature', 'Mutual Info']
                        ).properties(title=f"Feature Importance for {target_col}", 
                                    width=CONFIG["CHART_WIDTH"], height=CONFIG["CHART_HEIGHT"])
                        st.altair_chart(chart, use_container_width=True)
                    else:
                        st.info("No numeric features available for importance calculation.")
        except (ValueError, KeyError) as e:
            st.error(f"Error in feature importance: {e}")
            logger.error(f"Error in feature importance: {e}", exc_info=True)
        except Exception as e:
            st.error(f"Unexpected error in feature importance: {e}")
            logger.error(f"Unexpected error in feature importance: {e}", exc_info=True)

        # Outlier Visualization
        try:
            st.subheader("Outlier Visualization")
            num_cols = df_stats.get("numeric_cols", [])
            if num_cols:
                col = st.selectbox("Select column for box plot", num_cols, key="box_plot_col")
                chart = alt.Chart(sampled_df).mark_boxplot().encode(
                    y=f"{col}:Q",
                    tooltip=[col]
                ).properties(title=f"Box Plot for {col}", width=CONFIG["CHART_WIDTH"])
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("No numeric columns for outlier visualization.")
        except (ValueError, KeyError) as e:
            st.error(f"Error in box plot: {e}")
            logger.error(f"Error in outlier visualization: {e}", exc_info=True)

        # Missing Value Heatmap
        try:
            st.subheader("Missing Value Heatmap")
            missing_matrix = sampled_df.isna().astype(int)
            chart = alt.Chart(missing_matrix.stack().reset_index().rename(columns={0: 'Missing'})).mark_rect().encode(
                x='level_1:O',
                y='level_0:O',
                color=alt.Color('Missing:N', scale=alt.Scale(domain=[0, 1], range=['white', 'red'])),
                tooltip=['level_1', 'Missing']
            ).properties(title="Missing Value Heatmap", width=CONFIG["CHART_WIDTH"], height=CONFIG["CHART_HEIGHT"])
            st.altair_chart(chart, use_container_width=True)
        except (ValueError, KeyError) as e:
            st.error(f"Error in missing value heatmap: {e}")
            logger.error(f"Error in missing value heatmap: {e}", exc_info=True)

        # Time-Series Analysis
        try:
            st.subheader("Time-Series Analysis")
            datetime_cols = (df.select_dtypes(include=["datetime64"]).columns.compute().tolist() 
                            if isinstance(df, dd.DataFrame) 
                            else df.select_dtypes(include=["datetime64"]).columns.tolist())
            if datetime_cols:
                time_col = st.selectbox("Select datetime column", datetime_cols, key="ts_time_col")
                value_col = st.selectbox("Select value column", num_cols, key="ts_value_col")
                window = st.slider("Rolling window size", 3, 30, 7, key="ts_window")
                if pd.api.types.is_numeric_dtype(sampled_df[value_col]):
                    sampled_df['rolling_mean'] = sampled_df[value_col].rolling(window).mean()
                    chart = alt.Chart(sampled_df).mark_line().encode(
                        x=f"{time_col}:T",
                        y='rolling_mean:Q',
                        tooltip=[time_col, value_col, 'rolling_mean']
                    ).properties(title=f"Rolling Mean of {value_col}", 
                                width=CONFIG["CHART_WIDTH"], height=CONFIG["CHART_HEIGHT"])
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.error("Selected value column must be numeric.")
            else:
                st.info("No datetime columns for time-series analysis.")
        except (ValueError, KeyError) as e:
            st.error(f"Error in time-series analysis: {e}")
            logger.error(f"Error in time-series analysis: {e}", exc_info=True)

        # SQL Query Explorer
        try:
            st.subheader("SQL Query Explorer")
            query = st.text_area("Enter SQL query", value=f"SELECT * FROM df LIMIT 10", key="sql_query")
            if st.button("Run Query", key="run_sql"):
                if duckdb is None:
                    st.error("SQL query explorer requires duckdb. Install it to enable this feature.")
                else:
                    try:
                        validated_query = validate_sql_query(query)
                        result = duckdb.query(validated_query.replace('df', 'sampled_df')).df()
                        st.dataframe(result)
                        st.success("Query executed successfully.")
                    except ValueError as e:
                        st.error(f"Invalid query: {e}")
        except Exception as e:
            st.error(f"Error in SQL query explorer: {e}")
            logger.error(f"Error in SQL query explorer: {e}", exc_info=True)

    except Exception as e:
        st.error(f"Unexpected error in dashboard section: {e}. Please check the uploaded dataset or try again.")
        logger.error(f"Unexpected error in dashboard section: {e}", exc_info=True)

if __name__ == "__main__":
    section_dashboard()