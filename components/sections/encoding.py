import logging
import numpy as np
import streamlit as st
import pandas as pd
import dask.dataframe as dd
import json
from utils.data_utils import dtype_split, _arrowize, sample_for_preview
from preprocessing.steps import encode_categorical
import threading

logger = logging.getLogger(__name__)

# Thread lock for session state updates
session_lock = threading.Lock()

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
            key="encode_cols",
            help="Select columns to encode."
        )
        method = st.radio(
            "Encoding method",
            ["onehot", "label", "ordinal"],
            horizontal=True,
            key="encode_method",
            help="One-hot: Create dummy columns; Label: Integer encode; Ordinal: Map to ordered integers."
        )
        high_cardinality = st.radio(
            "Handle high-cardinality",
            ["None", "Target encode", "Frequency encode", "Hashing encode"],
            horizontal=True,
            key="high_cardinality",
            help="Target: Encode based on target mean; Frequency: Encode by category count; Hashing: Map to fixed buckets."
        )
        max_categories = st.number_input(
            "Max categories (0 for no limit)",
            min_value=0, value=0, step=1,
            key="max_categories",
            help="Limit the number of categories to encode."
        )
        group_rare = st.checkbox(
            "Group remaining as 'Rare'",
            key="group_rare",
            help="Group categories beyond max_categories as 'Rare' before encoding.",
            disabled=max_categories == 0
        )
        ordinal_mappings = None
        target_column = None
        n_components = 8
        if method == "ordinal":
            mapping_input = st.text_area(
                "Ordinal mappings (JSON format)",
                placeholder='{"column_name": {"value1": 0, "value2": 1, ...}}',
                key="ordinal_mappings",
                help="Provide a JSON dictionary mapping column values to integers."
            )
            try:
                if mapping_input:
                    ordinal_mappings = json.loads(mapping_input)
            except json.JSONDecodeError:
                st.error("Invalid JSON format for ordinal mappings.")
                return
        if high_cardinality == "Target encode":
            target_column = st.selectbox(
                "Target column",
                ["(none)"] + (df.columns.compute().tolist() if isinstance(df, dd.DataFrame) else df.columns.tolist()),
                key="target_column",
                help="Select a target column for target encoding."
            )
        if high_cardinality == "Hashing encode":
            n_components = st.number_input(
                "Number of components",
                min_value=1, value=8, step=1,
                key="n_components",
                help="Number of features for hashing encoding."
            )

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            if st.button("🔍 Preview Encoding", key="preview_encode", help="Preview the effect on a sampled dataset"):
                if not cols:
                    st.warning("Please select at least one column.")
                    return
                if method == "ordinal" and not ordinal_mappings:
                    st.warning("Ordinal encoding requires mappings.")
                    return
                if high_cardinality == "Target encode" and target_column == "(none)":
                    st.warning("Please select a target column.")
                    return
                prev = sample_for_preview(df)
                params = {
                    "columns": cols,
                    "method": high_cardinality.lower().replace(" ", "_") if high_cardinality != "None" else method,
                    "max_categories": max_categories if max_categories > 0 else None,
                    "group_rare": group_rare,
                    "ordinal_mappings": ordinal_mappings,
                    "target_column": target_column,
                    "n_components": n_components
                }
                preview_df, msg = encode_categorical(prev, **params)
                with session_lock:
                    st.session_state.last_preview = (preview_df, msg)
                st.info(msg)
                st.dataframe(_arrowize(preview_df.head(10)))
        with c2:
            if st.button("📦 Add to Pipeline", key="add_encode", help="Add encoding step to the pipeline"):
                if not cols:
                    st.warning("Please select at least one column.")
                    return
                if method == "ordinal" and not ordinal_mappings:
                    st.warning("Ordinal encoding requires mappings.")
                    return
                if high_cardinality == "Target encode" and target_column == "(none)":
                    st.warning("Please select a target column.")
                    return
                step = {
                    "kind": "encode",
                    "params": {
                        "columns": cols,
                        "method": high_cardinality.lower().replace(" ", "_") if high_cardinality != "None" else method,
                        "max_categories": max_categories if max_categories > 0 else None,
                        "group_rare": group_rare,
                        "ordinal_mappings": ordinal_mappings,
                        "target_column": target_column,
                        "n_components": n_components
                    }
                }
                with session_lock:
                    st.session_state.pipeline.append(step)
                st.success("Added encoding step to pipeline.")
        with c3:
            if st.button("🔄 Reset Selection", key="reset_encode", help="Clear selected columns and options"):
                st.session_state["encode_cols"] = []
                st.session_state["encode_method"] = "onehot"
                st.session_state["high_cardinality"] = "None"
                st.session_state["max_categories"] = 0
                st.session_state["group_rare"] = False
                st.session_state["ordinal_mappings"] = ""
                st.session_state["target_column"] = "(none)"
                st.session_state["n_components"] = 8
                st.rerun()

    except Exception as e:
        logger.error(f"Error in section_encoding: {e}")
        st.error(f"Error in encoding section: {e}")
