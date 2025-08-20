import logging
import streamlit as st
from data_preprocessing_studio.utils.data_utils import dtype_split, _arrowize, sample_for_preview
from data_preprocessing_studio.preprocessing.steps import encode_categorical

logger = logging.getLogger(__name__)

def section_encoding():
    st.header("5) Categorical Data Handling")
    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first.")
        return

    try:
        _, cat_cols = dtype_split(df)
        cols = st.multiselect("Categorical columns", cat_cols)
        method = st.radio("Encoding method", ["onehot", "label"], horizontal=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔍 Preview Encoding"):
                prev = sample_for_preview(df)
                preview_df, msg = encode_categorical(prev, cols, method)
                st.session_state.last_preview = (preview_df, msg)
                st.info(msg)
                st.dataframe(_arrowize(preview_df))
        with c2:
            if st.button("📦 Add to Pipeline (Encoding)"):
                step = {"kind": "encode", "params": {"columns": cols, "method": method}}
                st.session_state.pipeline.append(step)
                st.success("Added to pipeline.")
    except Exception as e:
        logger.error(f"Error in section_encoding: {e}")
        st.error(f"Error in encoding section: {e}")