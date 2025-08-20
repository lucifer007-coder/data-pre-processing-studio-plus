import logging
import streamlit as st
import pandas as pd
from utils.data_utils import dtype_split, _arrowize, sample_for_preview
from preprocessing.steps import impute_missing, drop_missing

logger = logging.getLogger(__name__)

def section_missing_data():
    st.header("2) Missing Data")
    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first.")
        return

    try:
        num_cols, cat_cols = dtype_split(df)
        st.subheader("Missing Values")
        missing = df.isna().sum()
        missing = missing[missing > 0]
        if missing.empty:
            st.info("No missing values in the dataset!")
        else:
            st.dataframe(_arrowize(pd.DataFrame({"Missing Count": missing})))

        st.subheader("Handle Missing Values")
        option = st.radio("Handle method", ["Impute", "Drop"], horizontal=True)
        if option == "Impute":
            strategy = st.selectbox("Imputation strategy", ["mean", "median", "mode", "constant"])
            constant_value = None
            if strategy == "constant":
                constant_value = st.text_input("Constant value", value="0")
                try:
                    constant_value = float(constant_value) if "." in constant_value else int(constant_value)
                except ValueError:
                    constant_value = constant_value
            cols = st.multiselect("Columns to impute", df.columns.tolist(), default=missing.index.tolist())
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🔍 Preview Imputation"):
                    prev = sample_for_preview(df)
                    preview_df, msg = impute_missing(prev, cols, strategy, constant_value)
                    st.session_state.last_preview = (preview_df, msg)
                    st.info(msg)
                    st.dataframe(_arrowize(preview_df))
            with c2:
                if st.button("📦 Add to Pipeline (Impute)"):
                    step = {
                        "kind": "impute",
                        "params": {"columns": cols, "strategy": strategy, "constant_value": constant_value},
                    }
                    st.session_state.pipeline.append(step)
                    st.success("Added to pipeline.")
        else:  # Drop
            axis = st.radio("Drop", ["rows", "columns"], horizontal=True)
            threshold = None
            cols = None
            if axis == "rows":
                use_threshold = st.checkbox("Use missing ratio threshold")
                if use_threshold:
                    threshold = st.slider("Max missing ratio", 0.0, 1.0, 0.5, 0.05)
                else:
                    cols = st.multiselect("Columns to check for dropping rows", df.columns.tolist(), default=missing.index.tolist())
            else:
                use_threshold = st.checkbox("Use missing ratio threshold")
                if use_threshold:
                    threshold = st.slider("Max missing ratio", 0.0, 1.0, 0.5, 0.05)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🔍 Preview Drop"):
                    prev = sample_for_preview(df)
                    preview_df, msg = drop_missing(prev, axis, threshold, cols)
                    st.session_state.last_preview = (preview_df, msg)
                    st.info(msg)
                    st.dataframe(_arrowize(preview_df))
            with c2:
                if st.button("📦 Add to Pipeline (Drop)"):
                    step = {
                        "kind": "drop_missing",
                        "params": {"axis": axis, "threshold": threshold, "columns": cols},
                    }
                    st.session_state.pipeline.append(step)
                    st.success("Added to pipeline.")
    except Exception as e:
        logger.error(f"Error in section_missing_data: {e}")
        st.error(f"Error in missing data section: {e}")
