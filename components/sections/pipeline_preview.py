import logging
import streamlit as st
import pandas as pd
import dask.dataframe as dd
from utils.data_utils import dtype_split, _arrowize, sample_for_preview
from preprocessing.steps import handle_outliers

logger = logging.getLogger(__name__)

def section_outliers():
    st.header("📈 Outliers / Noisy Data")
    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first.")
        return

    try:
        num_cols, _ = dtype_split(df)
        if not num_cols:
            st.info("No numeric columns available.")
            return

        st.subheader("Configure Outlier Handling")
        cols = st.multiselect(
            "Numeric columns to check",
            num_cols,
            help="Select numeric columns to detect and handle outliers."
        )
        col1, col2 = st.columns(2)
        with col1:
            method = st.selectbox(
                "Detection method",
                ["iqr", "zscore"],
                help="IQR: Use interquartile range; Z-score: Use standard deviations."
            )
        with col2:
            factor = st.slider(
                "Threshold (Z-score or IQR k)",
                0.5, 5.0, 1.5, 0.1,
                help="Z-score threshold or IQR fence multiplier."
            )

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            if st.button("🔍 Preview Outlier Handling", help="Preview the effect on a sampled dataset"):
                if not cols:
                    st.warning("Please select at least one column.")
                    return
                prev = sample_for_preview(df)
                preview_df, msg = handle_outliers(prev, cols, method, factor, preview=True)
                st.session_state.last_preview = (preview_df, msg)
                st.info(msg)
                st.dataframe(_arrowize(preview_df.head(10)))
        with c2:
            if st.button("📦 Add to Pipeline", help="Add outlier handling step to the pipeline"):
                if not cols:
                    st.warning("Please select at least one column.")
                    return
                step = {
                    "kind": "outliers",
                    "params": {"columns": cols, "method": method, "factor": factor}
                }
                st.session_state.pipeline.append(step)
                st.success("Added outlier handling step to pipeline.")
        with c3:
            if st.button("🔄 Reset Selection", help="Clear selected columns and options"):
                st.session_state["outlier_cols"] = []
                st.rerun()

    except Exception as e:
        logger.error(f"Error in section_outliers: {e}")
        st.error(f"Error in outliers section: {e}")
