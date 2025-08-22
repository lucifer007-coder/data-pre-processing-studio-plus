import logging
import streamlit as st
from utils.data_utils import _arrowize, sample_for_preview
from preprocessing.steps import remove_duplicates

logger = logging.getLogger(__name__)

def section_duplicates():
    st.header("🗑️ Data Duplication")
    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first.")
        return

    try:
        st.subheader("Duplicate Analysis")
        duplicate_count = df.duplicated().sum()
        st.write(f"**Detected Duplicates**: {duplicate_count} rows")

        st.subheader("Configure Duplicate Removal")
        subset = st.multiselect(
            "Columns to consider for duplicates",
            df.columns.tolist(),
            help="Leave empty to consider all columns."
        )
        keep = st.selectbox(
            "Keep",
            ["first", "last", "False"],
            help="Keep first/last occurrence of duplicates or drop all (False)."
        )
        karg = False if keep == "False" else keep

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            if st.button("🔍 Preview Duplicate Removal", help="Preview the effect on a sampled dataset"):
                prev = sample_for_preview(df)
                preview_df, msg = remove_duplicates(prev, subset or None, karg, preview=True)
                st.session_state.last_preview = (preview_df, msg)
                st.info(msg)
                st.dataframe(_arrowize(preview_df.head(10)))
        with c2:
            if st.button("📦 Add to Pipeline", help="Add duplicate removal step to the pipeline"):
                step = {"kind": "duplicates", "params": {"subset": subset or None, "keep": karg}}
                st.session_state.pipeline.append(step)
                st.success("Added duplicate removal step to pipeline.")
        with c3:
            if st.button("🔄 Reset Selection", help="Clear selected columns"):
                st.session_state["duplicates_subset"] = []
                st.rerun()

    except Exception as e:
        logger.error(f"Error in section_duplicates: {e}")
        st.error(f"Error in duplicates section: {e}")
