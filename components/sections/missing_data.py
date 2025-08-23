import logging
import streamlit as st
import pandas as pd
from utils.data_utils import dtype_split, _arrowize, sample_for_preview
from preprocessing.steps import impute_missing, drop_missing
import threading

logger = logging.getLogger(__name__)

# Thread lock for session state updates
session_lock = threading.Lock()

def section_missing_data():
    st.header("🧩 Missing Data")
    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first.")
        return

    try:
        num_cols, cat_cols = dtype_split(df)
        st.subheader("Missing Values Summary")
        missing = df.isna().sum()
        missing = missing[missing > 0]
        if missing.empty:
            st.info("No missing values in the dataset!")
        else:
            st.dataframe(_arrowize(pd.DataFrame({"Missing Count": missing, "Ratio": missing / len(df)})))

        st.subheader("Handle Missing Values")
        option = st.radio(
            "Handle method",
            ["Impute", "Drop"],
            horizontal=True,
            key="missing_handle_method",
            help="Impute: Fill missing values; Drop: Remove rows/columns with missing values."
        )

        if option == "Impute":
            strategy = st.selectbox(
                "Imputation strategy",
                ["mean", "median", "mode", "constant", "ffill", "bfill", "knn", "random_forest"],
                key="impute_strategy",
                help="Mean/Median for numeric; Mode for categorical; Constant for custom value; Ffill/Bfill for time-series; KNN/Random Forest for numeric columns."
            )
            constant_value = None
            n_neighbors = 5
            n_estimators = 100
            if strategy == "constant":
                constant_value = st.text_input(
                    "Constant value",
                    value="0",
                    key="constant_value",
                    help="Enter a numeric or string value to fill missing entries."
                )
                try:
                    constant_value = float(constant_value) if "." in constant_value else int(constant_value)
                except ValueError:
                    pass  # Keep as string if not numeric
            elif strategy == "knn":
                n_neighbors = st.number_input(
                    "Number of neighbors",
                    min_value=1, value=5, step=1,
                    key="knn_neighbors",
                    help="Number of neighbors for KNN imputation (numeric columns only)."
                )
            elif strategy == "random_forest":
                n_estimators = st.number_input(
                    "Number of trees",
                    min_value=10, value=100, step=10,
                    key="rf_estimators",
                    help="Number of trees for Random Forest imputation (numeric columns only)."
                )
            cols = st.multiselect(
                "Columns to impute",
                df.columns.tolist(),
                default=missing.index.tolist(),
                key="impute_cols",
                help="Select columns with missing values to impute."
            )
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                if st.button("🔍 Preview Imputation", key="preview_impute", help="Preview the effect on a sampled dataset"):
                    if not cols:
                        st.warning("Please select at least one column.")
                        return
                    prev = sample_for_preview(df)
                    params = {
                        "columns": cols,
                        "strategy": strategy,
                        "constant_value": constant_value,
                        "n_neighbors": n_neighbors,
                        "n_estimators": n_estimators,
                        "preview": True
                    }
                    preview_df, msg = impute_missing(prev, **params)
                    with session_lock:
                        st.session_state.last_preview = (preview_df, msg)
                    st.info(msg)
                    st.dataframe(_arrowize(preview_df.head(10)))
            with c2:
                if st.button("📦 Add to Pipeline", key="add_impute", help="Add imputation step to the pipeline"):
                    if not cols:
                        st.warning("Please select at least one column.")
                        return
                    step = {
                        "kind": "impute",
                        "params": {
                            "columns": cols,
                            "strategy": strategy,
                            "constant_value": constant_value,
                            "n_neighbors": n_neighbors,
                            "n_estimators": n_estimators
                        }
                    }
                    with session_lock:
                        st.session_state.pipeline.append(step)
                    st.success("Added imputation step to pipeline.")
            with c3:
                if st.button("🔄 Reset Selection", key="reset_impute", help="Clear selected options"):
                    st.session_state["impute_cols"] = []
                    st.session_state["impute_strategy"] = "mean"
                    st.session_state["constant_value"] = "0"
                    st.session_state["knn_neighbors"] = 5
                    st.session_state["rf_estimators"] = 100
                    st.rerun()
        else:
            axis = st.radio(
                "Drop axis",
                ["rows", "columns"],
                horizontal=True,
                key="drop_axis",
                help="Drop rows or columns with missing values."
            )
            threshold = None
            cols = None
            if axis == "rows":
                use_threshold = st.checkbox("Use missing ratio threshold", key="drop_threshold", help="Drop rows with a specified missing ratio.")
                if use_threshold:
                    threshold = st.slider(
                        "Max missing ratio",
                        0.0, 1.0, 0.5, 0.05,
                        key="drop_threshold_value",
                        help="Drop rows where the missing ratio exceeds this value."
                    )
                else:
                    cols = st.multiselect(
                        "Columns to check for dropping rows",
                        df.columns.tolist(),
                        default=missing.index.tolist(),
                        key="drop_cols",
                        help="Select columns to consider for dropping rows."
                    )
            else:
                use_threshold = st.checkbox("Use missing ratio threshold", key="drop_threshold_col", help="Drop columns with a specified missing ratio.")
                if use_threshold:
                    threshold = st.slider(
                        "Max missing ratio",
                        0.0, 1.0, 0.5, 0.05,
                        key="drop_threshold_value_col",
                        help="Drop columns where the missing ratio exceeds this value."
                    )
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                if st.button("🔍 Preview Drop", key="preview_drop", help="Preview the effect on a sampled dataset"):
                    prev = sample_for_preview(df)
                    preview_df, msg = drop_missing(prev, axis, threshold, cols, preview=True)
                    with session_lock:
                        st.session_state.last_preview = (preview_df, msg)
                    st.info(msg)
                    st.dataframe(_arrowize(preview_df.head(10)))
            with c2:
                if st.button("📦 Add to Pipeline", key="add_drop", help="Add drop step to the pipeline"):
                    step = {
                        "kind": "drop_missing",
                        "params": {"axis": axis, "threshold": threshold, "columns": cols}
                    }
                    with session_lock:
                        st.session_state.pipeline.append(step)
                    st.success("Added drop step to pipeline.")
            with c3:
                if st.button("🔄 Reset Selection", key="reset_drop", help="Clear selected options"):
                    st.session_state["drop_cols"] = []
                    st.session_state["drop_axis"] = "rows"
                    st.session_state["drop_threshold"] = False
                    st.session_state["drop_threshold_col"] = False
                    st.session_state["drop_threshold_value"] = 0.5
                    st.session_state["drop_threshold_value_col"] = 0.5
                    st.rerun()

    except Exception as e:
        logger.error(f"Error in section_missing_data: {e}")
        st.error(f"Error in missing data section: {e}")
