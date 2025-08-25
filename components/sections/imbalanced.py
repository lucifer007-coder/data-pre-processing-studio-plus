import logging
import streamlit as st
import pandas as pd
import dask.dataframe as dd
from utils.data_utils import _arrowize, sample_for_preview, dtype_split
from preprocessing.steps import rebalance_dataset
from utils.stats_utils import compute_basic_stats

logger = logging.getLogger(__name__)

def section_imbalanced():
    st.header("⚖️ Imbalanced Data (Classification)")
    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first.")
        return

    try:
        cat_cols = dtype_split(df)[1]
        if not cat_cols:
            st.info("No categorical columns available for classification.")
            return

        st.subheader("Configure Rebalancing")
        target = st.selectbox(
            "Target column (classification)",
            ["(none)"] + cat_cols,
            help="Select the target column for classification. Must be categorical."
        )
        method = st.radio(
            "Rebalancing method",
            ["oversample", "undersample"],
            horizontal=True,
            help="Oversample: Duplicate minority classes; Undersample: Reduce majority classes."
        )
        ratio = st.slider(
            "Ratio",
            0.2, 3.0, 1.0, 0.1,
            help="For oversampling, ratio × majority class size; for undersampling, ratio × minority class size."
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔍 Preview Rebalancing", help="Preview the effect on a sampled dataset"):
                if target == "(none)":
                    st.warning("Please select a target column.")
                    return
                prev = sample_for_preview(df)
                preview_df, msg = rebalance_dataset(prev, target, method, ratio, preview=True)
                st.session_state.last_preview = (preview_df, msg)
                st.info(msg)
                st.dataframe(_arrowize(preview_df.head(10)))
                # Display class distribution
                if target in df.columns:
                    st.subheader("Class Distribution")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Before Rebalancing")
                        counts_before = df[target].value_counts(dropna=False).compute() if isinstance(df, dd.DataFrame) else df[target].value_counts(dropna=False)
                        st.dataframe(_arrowize(pd.DataFrame({"Count": counts_before})))
                    with col2:
                        st.write("After Rebalancing (Preview)")
                        counts_after = preview_df[target].value_counts(dropna=False)
                        st.dataframe(_arrowize(pd.DataFrame({"Count": counts_after})))

        with c2:
            if st.button("📦 Add to Pipeline", help="Add rebalancing step to the pipeline"):
                if target == "(none)":
                    st.warning("Please select a target column.")
                    return
                step = {
                    "kind": "rebalance",
                    "params": {"target": target, "method": method, "ratio": ratio}
                }
                st.session_state.pipeline.append(step)
                st.success(f"Added rebalancing step for '{target}' to pipeline.")

    except Exception as e:
        logger.error(f"Error in section_imbalanced: {e}")
        st.error(f"Error in imbalanced data section: {e}")
