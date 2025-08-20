import logging
import streamlit as st
from data_preprocessing_studio.utils.data_utils import _arrowize, sample_for_preview
from data_preprocessing_studio.preprocessing.steps import remove_duplicates

logger = logging.getLogger(__name__)

def section_duplicates():
    st.header("4) Data Duplication")
    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first.")
        return

    try:
        subset = st.multiselect("Columns to consider duplicates on (leave empty for all columns)", df.columns.tolist())
        keep = st.selectbox("Keep", ["first", "last", "False"], help="If 'False', drop all duplicates.")
        karg = False if keep == "False" else keep
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔍 Preview Duplicate Removal"):
                prev = sample_for_preview(df)
                preview_df, msg = remove_duplicates(prev, subset or None, keep=karg)
                st.session_state.last_preview = (preview_df, msg)
                st.info(msg)
                st.dataframe(_arrowize(preview_df))
        with c2:
            if st.button("📦 Add to Pipeline (Duplicates)"):
                step = {"kind": "duplicates", "params": {"subset": subset or None, "keep": karg}}
                st.session_state.pipeline.append(step)
                st.success("Added to pipeline.")
    except Exception as e:
        logger.error(f"Error in section_duplicates: {e}")
        st.error(f"Error in duplicates section: {e}")