import logging
import streamlit as st
from utils.data_utils import dtype_split, _arrowize, sample_for_preview
from preprocessing.steps import handle_outliers

logger = logging.getLogger(__name__)

def section_outliers():
    st.header("3 & 7) Outliers / Noisy Data")
    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first.")
        return

    try:
        num_cols, _ = dtype_split(df)
        cols = st.multiselect("Numeric columns to check", num_cols)
        col1, col2, col3 = st.columns(3)
        with col1:
            detect_method = st.selectbox("Detection method", ["IQR", "Z-score"])
        with col2:
            zt = st.slider("Z-score threshold", 1.5, 5.0, 3.0, 0.1)
        with col3:
            ik = st.slider("IQR k (fence multiplier)", 0.5, 5.0, 1.5, 0.1)

        act = st.selectbox("Action on outliers", ["remove", "cap", "log1p"])
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔍 Preview Outlier Handling"):
                prev = sample_for_preview(df)
                preview_df, msg = handle_outliers(prev, cols, act, detect_method, z_thresh=zt, iqr_k=ik)
                st.session_state.last_preview = (preview_df, msg)
                st.info(msg)
                st.dataframe(_arrowize(preview_df))
        with c2:
            if st.button("📦 Add to Pipeline (Outliers)"):
                step = {
                    "kind": "outliers",
                    "params": {"columns": cols, "method": act, "detect_method": detect_method, "z_thresh": zt, "iqr_k": ik},
                }
                st.session_state.pipeline.append(step)
                st.success("Added to pipeline.")
    except Exception as e:
        logger.error(f"Error in section_outliers: {e}")
        st.error(f"Error in outliers section: {e}")
