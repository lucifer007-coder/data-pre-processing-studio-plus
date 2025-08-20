import logging
import streamlit as st
from data_preprocessing_studio.utils.data_utils import dtype_split, _arrowize, sample_for_preview
from data_preprocessing_studio.preprocessing.steps import scale_features

logger = logging.getLogger(__name__)

def section_scaling():
    st.header("6) Feature Scaling & Normalization")
    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first.")
        return

    try:
        num_cols, _ = dtype_split(df)
        cols = st.multiselect("Numeric columns to scale", num_cols)
        method = st.radio("Scaler", ["standard", "minmax"], horizontal=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔍 Preview Scaling"):
                prev = sample_for_preview(df)
                preview_df, msg = scale_features(prev, cols, method)
                st.session_state.last_preview = (preview_df, msg)
                st.info(msg)
                st.dataframe(_arrowize(preview_df))
        with c2:
            if st.button("📦 Add to Pipeline (Scaling)"):
                step = {"kind": "scale", "params": {"columns": cols, "method": method}}
                st.session_state.pipeline.append(step)
                st.success("Added to pipeline.")
    except Exception as e:
        logger.error(f"Error in section_scaling: {e}")
        st.error(f"Error in scaling section: {e}")