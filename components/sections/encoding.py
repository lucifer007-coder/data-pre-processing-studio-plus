import logging
import streamlit as st
import json
from utils.data_utils import dtype_split, _arrowize, sample_for_preview
from preprocessing.steps import encode_categorical

logger = logging.getLogger(__name__)

def section_encoding():
    st.header("🔢 Categorical Data Handling")
    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first.")
        return

    try:
        _, cat_cols = dtype_split(df)
        if not cat_cols:
            st.info("No categorical columns available.")
            return

        st.subheader("Configure Encoding")
        cols = st.multiselect(
            "Categorical columns",
            cat_cols,
            help="Select columns to encode."
        )
        method = st.radio(
            "Encoding method",
            ["onehot", "label", "ordinal"],
            horizontal=True,
            help="One-hot: Create dummy columns; Label: Integer encode; Ordinal: Map to ordered integers."
        )
        max_categories = st.number_input(
            "Max categories (0 for no limit)",
            min_value=0, value=0, step=1,
            help="Limit the number of categories to encode (replaces rare ones with 'Other')."
        )
        ordinal_mappings = None
        if method == "ordinal":
            mapping_input = st.text_area(
                "Ordinal mappings (JSON format)",
                placeholder='{"column_name": {"value1": 0, "value2": 1, ...}}',
                help="Provide a JSON dictionary mapping column values to integers."
            )
            try:
                if mapping_input:
                    ordinal_mappings = json.loads(mapping_input)
                    if not isinstance(ordinal_mappings, dict):
                        st.error("Ordinal mappings must be a JSON dictionary.")
                        return
            except json.JSONDecodeError:
                st.error("Invalid JSON format for ordinal mappings.")
                return

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔍 Preview Encoding", help="Preview the effect on a sampled dataset"):
                if not cols:
                    st.warning("Please select at least one column.")
                    return
                prev = sample_for_preview(df)
                params = {
                    "columns": cols,
                    "method": method,
                    "max_categories": max_categories if max_categories > 0 else None,
                    "ordinal_mappings": ordinal_mappings,
                    "preview": True
                }
                preview_df, msg = encode_categorical(prev, **params)
                st.session_state.last_preview = (preview_df, msg)
                st.info(msg)
                st.dataframe(_arrowize(preview_df.head(10)))
        with c2:
            if st.button("📦 Add to Pipeline", help="Add encoding step to the pipeline"):
                if not cols:
                    st.warning("Please select at least one column.")
                    return
                if method == "ordinal" and not ordinal_mappings:
                    st.warning("Ordinal encoding requires mappings.")
                    return
                step = {
                    "kind": "encode",
                    "params": {
                        "columns": cols,
                        "method": method,
                        "max_categories": max_categories if max_categories > 0 else None,
                        "ordinal_mappings": ordinal_mappings
                    }
                }
                st.session_state.pipeline.append(step)
                st.success("Added encoding step to pipeline.")

    except Exception as e:
        logger.error(f"Error in section_encoding: {e}")
        st.error(f"Error in encoding section: {e}")
