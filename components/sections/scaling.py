import logging
import streamlit as st
from utils.data_utils import dtype_split, _arrowize, sample_for_preview
from preprocessing.steps import scale_features

logger = logging.getLogger(__name__)

def section_scaling():
    st.header("📏 Feature Scaling & Normalization")
    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first.")
        return

    try:
        num_cols, _ = dtype_split(df)
        if not num_cols:
            st.info("No numeric columns available.")
            return

        st.subheader("Configure Scaling")
        cols = st.multiselect(
            "Numeric columns to scale",
            num_cols,
            help="Select numeric columns to scale."
        )
        method = st.radio(
            "Scaler",
            ["standard", "minmax", "robust"],
            horizontal=True,
            help="Standard: Zero mean, unit variance; MinMax: Scale to [0,1]; Robust: Scale using median and IQR."
        )
        keep_original = st.checkbox(
            "Keep original columns",
            help="Create new scaled columns instead of overwriting."
        )

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            if st.button("🔍 Preview Scaling", help="Preview the effect on a sampled dataset"):
                if not cols:
                    st.warning("Please select at least one column.")
                    return
                prev = sample_for_preview(df)
                preview_df, msg = scale_features(prev, cols, method, keep_original, preview=True)
                st.session_state.last_preview = (preview_df, msg)
                st.info(msg)
                st.dataframe(_arrowize(preview_df.head(10)))
        with c2:
            if st.button("📦 Add to Pipeline", help="Add scaling step to the pipeline"):
                if not cols:
                    st.warning("Please select at least one column.")
                    return
                step = {
                    "kind": "scale",
                    "params": {"columns": cols, "method": method, "keep_original": keep_original}
                }
                st.session_state.pipeline.append(step)
                st.success("Added scaling step to pipeline.")
        with c3:
            if st.button("🔄 Reset Selection", help="Clear selected columns and options"):
                st.session_state["scale_cols"] = []
                st.rerun()

    except Exception as e:
        logger.error(f"Error in section_scaling: {e}")
        st.error(f"Error in scaling section: {e}")
