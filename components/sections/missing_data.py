import logging
import streamlit as st
import pandas as pd
from utils.data_utils import dtype_split, _arrowize, sample_for_preview
from preprocessing.steps import impute_missing, drop_missing

logger = logging.getLogger(__name__)

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
            help="Impute: Fill missing values; Drop: Remove rows/columns with missing values."
        )

        if option == "Impute":
            strategy = st.selectbox(
                "Imputation strategy",
                ["mean", "median", "mode", "constant", "ffill", "bfill"],
                help="Mean/Median for numeric; Mode for categorical; Constant for custom value; Ffill/Bfill for time-series."
            )
            constant_value = None
            if strategy == "constant":
                constant_value = st.text_input(
                    "Constant value",
                    value="0",
                    help="Enter a numeric or string value to fill missing entries."
                )
                try:
                    constant_value = float(constant_value) if "." in constant_value else int(constant_value)
                except ValueError:
                    pass  # Keep as string if not numeric
            cols = st.multiselect(
                "Columns to impute",
                df.columns.tolist(),
                default=missing.index.tolist(),
                help="Select columns with missing values to impute."
            )
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                if st.button("🔍 Preview Imputation", help="Preview the effect on a sampled dataset"):
                    if not cols:
                        st.warning("Please select at least one column.")
                        return
                    prev = sample_for_preview(df)
                    preview_df, msg = impute_missing(prev, cols, strategy, constant_value, preview=True)
                    st.session_state.last_preview = (preview_df, msg)
                    st.info(msg)
                    st.dataframe(_arrowize(preview_df.head(10)))
            with c2:
                if st.button("📦 Add to Pipeline", help="Add imputation step to the pipeline"):
                    if not cols:
                        st.warning("Please select at least one column.")
                        return
                    step = {
                        "kind": "impute",
                        "params": {"columns": cols, "strategy": strategy, "constant_value": constant_value}
                    }
                    st.session_state.pipeline.append(step)
                    st.success("Added imputation step to pipeline.")
            with c3:
                if st.button("🔄 Reset Selection", help="Clear selected columns and strategy"):
                    st.session_state["impute_cols"] = []
                    st.rerun()
        else:  # Drop
            axis = st.radio(
                "Drop",
                ["rows", "columns"],
                horizontal=True,
                help="Drop rows or columns with missing values."
            )
            threshold = None
            cols = None
            if axis == "rows":
                use_threshold = st.checkbox("Use missing ratio threshold", help="Drop rows with a specified missing ratio.")
                if use_threshold:
                    threshold = st.slider(
                        "Max missing ratio",
                        0.0, 1.0, 0.5, 0.05,
                        help="Drop rows where the missing ratio exceeds this value."
                    )
                else:
                    cols = st.multiselect(
                        "Columns to check for dropping rows",
                        df.columns.tolist(),
                        default=missing.index.tolist(),
                        help="Select columns to consider for dropping rows."
                    )
            else:
                use_threshold = st.checkbox("Use missing ratio threshold", help="Drop columns with a specified missing ratio.")
                if use_threshold:
                    threshold = st.slider(
                        "Max missing ratio",
                        0.0, 1.0, 0.5, 0.05,
                        help="Drop columns where the missing ratio exceeds this value."
                    )
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                if st.button("🔍 Preview Drop", help="Preview the effect on a sampled dataset"):
                    prev = sample_for_preview(df)
                    preview_df, msg = drop_missing(prev, axis, threshold, cols, preview=True)
                    st.session_state.last_preview = (preview_df, msg)
                    st.info(msg)
                    st.dataframe(_arrowize(preview_df.head(10)))
            with c2:
                if st.button("📦 Add to Pipeline", help="Add drop step to the pipeline"):
                    step = {
                        "kind": "drop_missing",
                        "params": {"axis": axis, "threshold": threshold, "columns": cols}
                    }
                    st.session_state.pipeline.append(step)
                    st.success("Added drop step to pipeline.")
            with c3:
                if st.button("🔄 Reset Selection", help="Clear selected options"):
                    st.session_state["drop_cols"] = []
                    st.rerun()

    except Exception as e:
        logger.error(f"Error in section_missing_data: {e}")
        st.error(f"Error in missing data section: {e}")
