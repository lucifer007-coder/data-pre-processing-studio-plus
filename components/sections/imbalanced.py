import logging
import streamlit as st
from utils.data_utils import _arrowize, sample_for_preview
from preprocessing.steps import rebalance_dataset

logger = logging.getLogger(__name__)

def section_imbalanced():
    st.header("8) Imbalanced Data (Classification)")
    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first.")
        return

    try:
        target = st.selectbox("Target column (classification)", ["(none)"] + st.session_state.df.columns.tolist())
        method = st.radio("Method", ["oversample", "undersample"], horizontal=True)
        ratio = st.slider("Ratio", 0.2, 3.0, 1.0, 0.1, help="Oversample to ratio×majority; Undersample to ratio×minority.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔍 Preview Rebalancing"):
                prev = sample_for_preview(df)
                if target != "(none)":
                    preview_df, msg = rebalance_dataset(prev, target, method, ratio)
                else:
                    preview_df, msg = prev, "No target chosen."
                st.session_state.last_preview = (preview_df, msg)
                st.info(msg)
                st.dataframe(_arrowize(preview_df))
        with c2:
            if st.button("📦 Add to Pipeline (Rebalance)"):
                if target == "(none)":
                    st.warning("Please select a target column.")
                else:
                    step = {
                        "kind": "rebalance",
                        "params": {"target": target, "method": method, "ratio": ratio},
                    }
                    st.session_state.pipeline.append(step)
                    st.success("Added to pipeline.")
    except Exception as e:
        logger.error(f"Error in section_imbalanced: {e}")
        st.error(f"Error in imbalanced data section: {e}")
