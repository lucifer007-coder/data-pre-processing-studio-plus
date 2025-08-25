import logging
import streamlit as st
import pandas as pd
import dask.dataframe as dd
from utils.data_utils import dtype_split, _arrowize, sample_for_preview
from preprocessing.steps import normalize_text, standardize_dates, unit_convert, type_convert, drop_missing, extract_domain
import threading

logger = logging.getLogger(__name__)

# Thread lock for session state updates
session_lock = threading.Lock()

def section_inconsistency():
    st.header("📏 Data Inconsistency")
    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first.")
        return

    try:
        num_cols, cat_cols = dtype_split(df)
        columns = df.columns.compute().tolist() if isinstance(df, dd.DataFrame) else df.columns.tolist()

        # Text Normalization
        st.subheader("Text Normalization")
        text_cols = st.multiselect(
            "Text columns",
            [c for c in columns if c in cat_cols],
            key="text_cols",
            help="Select columns for text normalization."
        )
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            lower = st.checkbox("Lowercase", True, key="text_lower", help="Convert text to lowercase.")
        with c2:
            trim = st.checkbox("Trim spaces", True, key="text_trim", help="Remove leading/trailing spaces.")
        with c3:
            collapse = st.checkbox("Collapse spaces", True, key="text_collapse", help="Replace multiple spaces with a single space.")
        with c4:
            remove_special = st.checkbox("Remove special chars", False, key="text_remove_special", help="Remove non-alphanumeric characters.")
        cc1, cc2, cc3 = st.columns([1, 1, 1])
        with cc1:
            if st.button("🔍 Preview Text Normalization", key="preview_text_norm", help="Preview the effect on a sampled dataset"):
                if not text_cols:
                    st.warning("Please select at least one column.")
                    return
                prev = sample_for_preview(df)
                preview_df, msg = normalize_text(
                    prev, text_cols, lower, trim, collapse, remove_special, preview=True
                )
                with session_lock:
                    st.session_state.last_preview = (preview_df, msg)
                st.info(msg)
                st.dataframe(_arrowize(preview_df.head(10)))
        with cc2:
            if st.button("📦 Add to Pipeline", key="add_text_norm", help="Add text normalization step to pipeline"):
                if not text_cols:
                    st.warning("Please select at least one column.")
                    return
                step = {
                    "kind": "normalize_text",
                    "params": {
                        "columns": text_cols,
                        "lower": lower,
                        "trim": trim,
                        "collapse": collapse,
                        "remove_special": remove_special
                    }
                }
                with session_lock:
                    st.session_state.pipeline.append(step)
                st.success("Added text normalization step to pipeline.")
        with cc3:
            if st.button("🔄 Reset Text Selection", key="reset_text", help="Clear selected columns"):
                st.session_state["text_cols"] = []
                st.rerun()

        # Date Standardization
        st.subheader("Date Standardization")
        # Detect date-like columns
        date_cols = []
        for col in columns:
            try:
                if isinstance(df, dd.DataFrame):
                    is_datetime = df[col].dtype == 'datetime64[ns]'
                    has_date_pattern = df[col].astype(str).str.match(r'\d{4}-\d{2}-\d{2}', na=False).any().compute()
                    is_date_like = is_datetime or has_date_pattern
                else:
                    is_datetime = df[col].dtype == 'datetime64[ns]'
                    has_date_pattern = df[col].astype(str).str.match(r'\d{4}-\d{2}-\d{2}', na=False).any()
                    is_date_like = is_datetime or has_date_pattern
                if is_date_like:
                    date_cols.append(col)
            except Exception as e:
                logger.warning(f"Error checking column {col} for date-like data: {e}")
        
        date_cols_selected = st.multiselect(
            "Date columns",
            date_cols,
            key="date_cols",
            help="Select columns with date-like data."
        )
        date_format = st.text_input(
            "Date format",
            value="%Y-%m-%d",
            key="date_format",
            help="Specify the output format (e.g., %Y-%m-%d for YYYY-MM-DD)."
        )
        dc1, dc2, dc3 = st.columns([1, 1, 1])
        with dc1:
            if st.button("🔍 Preview Date Standardization", key="preview_date", help="Preview the effect on a sampled dataset"):
                if not date_cols_selected:
                    st.warning("Please select at least one column.")
                    return
                prev = sample_for_preview(df)
                preview_df, msg = standardize_dates(prev, date_cols_selected, date_format, preview=True)
                with session_lock:
                    st.session_state.last_preview = (preview_df, msg)
                st.info(msg)
                st.dataframe(_arrowize(preview_df.head(10)))
        with dc2:
            if st.button("📦 Add to Pipeline", key="add_date", help="Add date standardization step to pipeline"):
                if not date_cols_selected:
                    st.warning("Please select at least one column.")
                    return
                step = {
                    "kind": "standardize_dates",
                    "params": {"columns": date_cols_selected, "format": date_format}
                }
                with session_lock:
                    st.session_state.pipeline.append(step)
                st.success("Added date standardization step to pipeline.")
        with dc3:
            if st.button("🔄 Reset Date Selection", key="reset_date", help="Clear selected columns"):
                st.session_state["date_cols"] = []
                st.rerun()

        # Unit Conversion
        st.subheader("Unit Conversion")
        unit_col = st.selectbox(
            "Numeric column",
            [c for c in columns if c in num_cols],
            key="unit_col",
            help="Select a numeric column for unit conversion."
        )
        factor = st.number_input(
            "Conversion factor",
            min_value=0.0, value=1.0, step=0.1,
            key="unit_factor",
            help="Multiply values by this factor (e.g., 0.001 for kg to g)."
        )
        uc1, uc2, uc3 = st.columns([1, 1, 1])
        with uc1:
            if st.button("🔍 Preview Unit Conversion", key="preview_unit", help="Preview the effect on a sampled dataset"):
                if not unit_col:
                    st.warning("Please select a column.")
                    return
                prev = sample_for_preview(df)
                preview_df, msg = unit_convert(prev, unit_col, factor, preview=True)
                with session_lock:
                    st.session_state.last_preview = (preview_df, msg)
                st.info(msg)
                st.dataframe(_arrowize(preview_df.head(10)))
        with uc2:
            if st.button("📦 Add to Pipeline", key="add_unit", help="Add unit conversion step to pipeline"):
                if not unit_col:
                    st.warning("Please select a column.")
                    return
                step = {
                    "kind": "unit_convert",
                    "params": {"column": unit_col, "factor": factor}
                }
                with session_lock:
                    st.session_state.pipeline.append(step)
                st.success("Added unit conversion step to pipeline.")
        with uc3:
            if st.button("🔄 Reset Unit Selection", key="reset_unit", help="Clear selected column"):
                st.session_state["unit_col"] = None
                st.rerun()

        # Boolean Type Conversion
        st.subheader("Boolean Type Conversion")
        bool_cols = []
        for col in columns:
            try:
                if isinstance(df, dd.DataFrame):
                    is_boolean = df[col].isin([0, 1, '0', '1', True, False]).all().compute()
                else:
                    is_boolean = df[col].isin([0, 1, '0', '1', True, False]).all()
                if is_boolean:
                    bool_cols.append(col)
            except Exception as e:
                logger.warning(f"Error checking column {col} for boolean data: {e}")
        
        bool_cols_selected = st.multiselect(
            "Columns to convert to boolean",
            bool_cols,
            key="bool_cols",
            help="Select columns with 0/1 or '0'/'1' values to convert to boolean."
        )
        bo1, bo2, bo3 = st.columns([1, 1, 1])
        with bo1:
            if st.button("🔍 Preview Boolean Conversion", key="preview_bool", help="Preview the effect on a sampled dataset"):
                if not bool_cols_selected:
                    st.warning("Please select at least one column.")
                    return
                prev = sample_for_preview(df)
                preview_df = prev.copy()
                msg = ""
                for col in bool_cols_selected:
                    preview_df, temp_msg = type_convert(preview_df, col, "bool", preview=True)
                    msg += f"{temp_msg}\n"
                with session_lock:
                    st.session_state.last_preview = (preview_df, msg)
                st.info(msg)
                st.dataframe(_arrowize(preview_df.head(10)))
        with bo2:
            if st.button("📦 Add to Pipeline", key="add_bool", help="Add boolean conversion step to pipeline"):
                if not bool_cols_selected:
                    st.warning("Please select at least one column.")
                    return
                for col in bool_cols_selected:
                    step = {"kind": "type_convert", "params": {"column": col, "type": "bool"}}
                    with session_lock:
                        st.session_state.pipeline.append(step)
                st.success(f"Added boolean conversion step for {len(bool_cols_selected)} columns to pipeline.")
        with bo3:
            if st.button("🔄 Reset Boolean Selection", key="reset_bool", help="Clear selected columns"):
                st.session_state["bool_cols"] = []
                st.rerun()

        # Numeric-to-Categorical Handling
        st.subheader("Numeric-to-Categorical Handling")
        num_cat_cols = st.multiselect(
            "Numeric columns",
            [c for c in columns if c in num_cols],
            key="num_cat_cols",
            help="Select numeric columns to convert to categorical."
        )
        nc1, nc2, nc3 = st.columns([1, 1, 1])
        with nc1:
            if st.button("🔍 Preview Categorical Conversion", key="preview_num_cat", help="Preview the effect on a sampled dataset"):
                if not num_cat_cols:
                    st.warning("Please select at least one column.")
                    return
                prev = sample_for_preview(df)
                preview_df, msg = type_convert(prev, num_cat_cols[0], "category", preview=True)
                with session_lock:
                    st.session_state.last_preview = (preview_df, msg)
                st.info(msg)
                st.dataframe(_arrowize(preview_df.head(10)))
        with nc2:
            if st.button("📦 Add to Pipeline", key="add_num_cat", help="Add categorical conversion step to the pipeline"):
                if not num_cat_cols:
                    st.warning("Please select at least one column.")
                    return
                for col in num_cat_cols:
                    step = {"kind": "type_convert", "params": {"column": col, "type": "category"}}
                    with session_lock:
                        st.session_state.pipeline.append(step)
                st.success(f"Added categorical conversion step for {len(num_cat_cols)} columns to pipeline.")
        with nc3:
            if st.button("🔄 Reset Categorical Selection", key="reset_num_cat", help="Clear selected columns"):
                st.session_state["num_cat_cols"] = []
                st.rerun()

        # Domain Extraction from URLs
        st.subheader("Domain Extraction from URLs")
        url_cols = []
        for col in columns:
            try:
                if isinstance(df, dd.DataFrame):
                    is_url = df[col].astype(str).str.startswith(('http://', 'https://')).any().compute()
                else:
                    is_url = df[col].astype(str).str.startswith(('http://', 'https://')).any()
                if is_url:
                    url_cols.append(col)
            except Exception as e:
                logger.warning(f"Error checking column {col} for URL data: {e}")
        
        url_cols_selected = st.multiselect(
            "URL columns",
            url_cols,
            key="url_cols",
            help="Select columns containing URLs to extract domains from."
        )
        de1, de2, de3 = st.columns([1, 1, 1])
        with de1:
            if st.button("🔍 Preview Domain Extraction", key="preview_domain", help="Preview domain extraction on a sampled dataset"):
                if not url_cols_selected:
                    st.warning("Please select at least one column.")
                    return
                prev = sample_for_preview(df)
                preview_df, msg = extract_domain(prev, url_cols_selected[0], preview=True)
                with session_lock:
                    st.session_state.last_preview = (preview_df, msg)
                st.info(msg)
                st.dataframe(_arrowize(preview_df.head(10)))
        with de2:
            if st.button("📦 Add to Pipeline", key="add_domain", help="Add domain extraction step to pipeline"):
                if not url_cols_selected:
                    st.warning("Please select at least one column.")
                    return
                for col in url_cols_selected:
                    step = {"kind": "extract_domain", "params": {"column": col}}
                    with session_lock:
                        st.session_state.pipeline.append(step)
                st.success(f"Added domain extraction step for {len(url_cols_selected)} columns to pipeline.")
        with de3:
            if st.button("🔄 Reset URL Selection", key="reset_domain", help="Clear selected columns"):
                st.session_state["url_cols"] = []
                st.rerun()

    except Exception as e:
        logger.error(f"Error in section_inconsistency: {e}")
        st.error(f"Error in data inconsistency section: {e}")
