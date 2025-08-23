import logging
import streamlit as st
from utils.data_utils import dtype_split, _arrowize, sample_for_preview
from preprocessing.steps import normalize_text, standardize_dates, unit_convert, type_convert, drop_missing, extract_domain
import pandas as pd
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

        # Text Normalization
        st.subheader("Text Normalization")
        text_cols = st.multiselect(
            "Text columns",
            [c for c in df.columns if c in cat_cols],
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
            if st.button("📦 Add to Pipeline", key="add_text_norm", help="Add text normalization step to the pipeline"):
                if not text_cols:
                    st.warning("Please select at least one column.")
                    return
                step = {
                    "kind": "normalize_text",
                    "params": {
                        "columns": text_cols,
                        "lowercase": lower,
                        "trim": trim,
                        "collapse": collapse,
                        "remove_special": remove_special
                    }
                }
                with session_lock:
                    st.session_state.pipeline.append(step)
                st.success("Added text normalization step to pipeline.")
        with cc3:
            if st.button("🔄 Reset Text Selection", key="reset_text_norm", help="Clear selected text columns and options"):
                st.session_state["text_cols"] = []
                st.session_state["text_lower"] = True
                st.session_state["text_trim"] = True
                st.session_state["text_collapse"] = True
                st.session_state["text_remove_special"] = False
                st.rerun()

        # Date Standardization
        st.subheader("Date Standardization")
        date_cols = st.multiselect(
            "Date columns",
            [c for c in df.columns if c in cat_cols or c in df.select_dtypes(include=["datetime64"]).columns.tolist()],
            key="date_cols",
            help="Select columns to standardize date formats."
        )
        date_format = st.text_input(
            "Date format",
            value="%Y-%m-%d",
            key="date_format",
            help="Specify the target date format (e.g., %Y-%m-%d)."
        )
        dc1, dc2, dc3 = st.columns([1, 1, 1])
        with dc1:
            if st.button("🔍 Preview Date Standardization", key="preview_date_std", help="Preview the effect on a sampled dataset"):
                if not date_cols:
                    st.warning("Please select at least one column.")
                    return
                prev = sample_for_preview(df)
                preview_df, msg = standardize_dates(prev, date_cols, date_format, preview=True)
                with session_lock:
                    st.session_state.last_preview = (preview_df, msg)
                st.info(msg)
                st.dataframe(_arrowize(preview_df.head(10)))
        with dc2:
            if st.button("📦 Add to Pipeline", key="add_date_std", help="Add date standardization step to the pipeline"):
                if not date_cols:
                    st.warning("Please select at least one column.")
                    return
                step = {"kind": "standardize_dates", "params": {"columns": date_cols, "format": date_format}}
                with session_lock:
                    st.session_state.pipeline.append(step)
                st.success("Added date standardization step to pipeline.")
        with dc3:
            if st.button("🔄 Reset Date Selection", key="reset_date_std", help="Clear selected date columns and options"):
                st.session_state["date_cols"] = []
                st.session_state["date_format"] = "%Y-%m-%d"
                st.rerun()

        # Unit Conversion
        st.subheader("Unit Conversion")
        uc_col = st.selectbox(
            "Numeric column",
            ["(none)"] + num_cols,
            key="uc_col",
            help="Select a numeric column to convert units."
        )
        colA, colB, colC = st.columns(3)
        with colA:
            factor = st.number_input(
                "Multiply by factor",
                value=1.0, step=0.1,
                key="uc_factor",
                help="Enter the conversion factor (e.g., 0.001 for km to m)."
            )
        with colB:
            new_name = st.text_input(
                "New column name (optional)",
                "",
                key="uc_new_name",
                help="Leave blank to overwrite the original column."
            )
        with colC:
            st.caption("Tip: Use this to create normalized units (e.g., convert km to m).")
        uc1, uc2, uc3 = st.columns([1, 1, 1])
        with uc1:
            if st.button("🔍 Preview Unit Conversion", key="preview_unit_convert", help="Preview the effect on a sampled dataset"):
                if uc_col == "(none)":
                    st.warning("Please select a column.")
                    return
                if factor == 0:
                    st.error("Factor cannot be zero.")
                    return
                prev = sample_for_preview(df)
                preview_df, msg = unit_convert(prev, uc_col, factor, new_name or None, preview=True)
                with session_lock:
                    st.session_state.last_preview = (preview_df, msg)
                st.info(msg)
                st.dataframe(_arrowize(preview_df.head(10)))
        with uc2:
            if st.button("📦 Add to Pipeline", key="add_unit_convert", help="Add unit conversion step to the pipeline"):
                if uc_col == "(none)":
                    st.warning("Please select a column.")
                    return
                if factor == 0:
                    st.error("Factor cannot be zero.")
                    return
                step = {
                    "kind": "unit_convert",
                    "params": {"column": uc_col, "factor": float(factor), "new_name": new_name or None}
                }
                with session_lock:
                    st.session_state.pipeline.append(step)
                st.success("Added unit conversion step to pipeline.")
        with uc3:
            if st.button("🔄 Reset Unit Selection", key="reset_unit_convert", help="Clear selected column and options"):
                st.session_state["uc_col"] = "(none)"
                st.session_state["uc_factor"] = 1.0
                st.session_state["uc_new_name"] = ""
                st.rerun()

        # Unparsable Date Handling
        st.subheader("Unparsable Date Handling")
        unparsable_cols = st.multiselect(
            "Date columns",
            [c for c in cat_cols],
            key="unparsable_cols",
            help="Select columns with potentially unparsable date strings."
        )
        unparsable_format = st.text_input(
            "Target date format",
            value="%Y-%m-%d",
            key="unparsable_format",
            help="Specify the target date format (e.g., %Y-%m-%d) for standardization."
        )
        action = st.radio(
            "Action",
            ["Standardize", "Drop"],
            horizontal=True,
            key="unparsable_action",
            help="Standardize: Convert to specified format; Drop: Remove selected columns."
        )
        ud1, ud2, ud3 = st.columns([1, 1, 1])
        with ud1:
            if st.button("🔍 Preview Unparsable Date Handling", key="preview_unparsable", help="Preview the effect on a sampled dataset"):
                if not unparsable_cols:
                    st.warning("Please select at least one column.")
                    return
                prev = sample_for_preview(df)
                if action == "Standardize":
                    preview_df, msg = standardize_dates(prev, unparsable_cols, unparsable_format, preview=True)
                else:
                    preview_df, msg = drop_missing(prev, axis="columns", columns=unparsable_cols, preview=True)
                with session_lock:
                    st.session_state.last_preview = (preview_df, msg)
                st.info(msg)
                st.dataframe(_arrowize(preview_df.head(10)))
        with ud2:
            if st.button("📦 Add to Pipeline", key="add_unparsable", help="Add unparsable date handling step to the pipeline"):
                if not unparsable_cols:
                    st.warning("Please select at least one column.")
                    return
                if action == "Standardize":
                    step = {"kind": "standardize_dates", "params": {"columns": unparsable_cols, "format": unparsable_format}}
                else:
                    step = {"kind": "drop_missing", "params": {"axis": "columns", "columns": unparsable_cols}}
                with session_lock:
                    st.session_state.pipeline.append(step)
                st.success(f"Added {action.lower()} step for unparsable dates to pipeline.")
        with ud3:
            if st.button("🔄 Reset Unparsable Date Selection", key="reset_unparsable", help="Clear selected columns and options"):
                st.session_state["unparsable_cols"] = []
                st.session_state["unparsable_format"] = "%Y-%m-%d"
                st.session_state["unparsable_action"] = "Standardize"
                st.rerun()

        # Currency Symbol Handling
        st.subheader("Currency Symbol Handling")
        currency_cols = st.multiselect(
            "Currency columns",
            [c for c in cat_cols],
            key="currency_cols",
            help="Select columns with currency values (e.g., '$100.50')."
        )
        currency_factor = st.number_input(
            "Conversion factor",
            value=1.0,
            step=0.1,
            key="currency_factor",
            help="Factor to extract numeric values (e.g., 1.0 for no conversion)."
        )
        currency_new_name = st.text_input(
            "New column name (optional)",
            "",
            key="currency_new_name",
            help="Leave blank to overwrite the original column."
        )
        cu1, cu2, cu3 = st.columns([1, 1, 1])
        with cu1:
            if st.button("🔍 Preview Currency Handling", key="preview_currency", help="Preview the effect on a sampled dataset"):
                if not currency_cols:
                    st.warning("Please select at least one column.")
                    return
                if currency_factor == 0:
                    st.error("Conversion factor cannot be zero.")
                    return
                prev = sample_for_preview(df)
                preview_df, msg = unit_convert(prev, currency_cols[0], currency_factor, currency_new_name or None, preview=True)
                with session_lock:
                    st.session_state.last_preview = (preview_df, msg)
                st.info(msg)
                st.dataframe(_arrowize(preview_df.head(10)))
        with cu2:
            if st.button("📦 Add to Pipeline", key="add_currency", help="Add currency handling step to the pipeline"):
                if not currency_cols:
                    st.warning("Please select at least one column.")
                    return
                if currency_factor == 0:
                    st.error("Conversion factor cannot be zero.")
                    return
                for col in currency_cols:
                    step = {
                        "kind": "unit_convert",
                        "params": {"column": col, "factor": float(currency_factor), "new_name": currency_new_name or None}
                    }
                    with session_lock:
                        st.session_state.pipeline.append(step)
                st.success(f"Added unit conversion step for {len(currency_cols)} currency columns to pipeline.")
        with cu3:
            if st.button("🔄 Reset Currency Selection", key="reset_currency", help="Clear selected columns and options"):
                st.session_state["currency_cols"] = []
                st.session_state["currency_factor"] = 1.0
                st.session_state["currency_new_name"] = ""
                st.rerun()

        # URL/File-Path Handling
        st.subheader("URL/File-Path Handling")
        url_cols = st.multiselect(
            "URL/File-Path columns",
            [c for c in cat_cols],
            key="url_cols",
            help="Select columns with URLs or file paths."
        )
        url_action = st.radio(
            "Action",
            ["Extract Domain", "Drop"],
            horizontal=True,
            key="url_action",
            help="Extract Domain: Create new column with domains; Drop: Remove selected columns."
        )
        url_new_name = st.text_input(
            "New column name for domains (optional)",
            "",
            key="url_new_name",
            help="Leave blank to overwrite the original column (if extracting domains).",
            disabled=url_action == "Drop"
        )
        ur1, ur2, ur3 = st.columns([1, 1, 1])
        with ur1:
            if st.button("🔍 Preview URL Handling", key="preview_url", help="Preview the effect on a sampled dataset"):
                if not url_cols:
                    st.warning("Please select at least one column.")
                    return
                prev = sample_for_preview(df)
                if url_action == "Extract Domain":
                    preview_df, msg = extract_domain(prev, url_cols[0], url_new_name or None, preview=True)
                else:
                    preview_df, msg = drop_missing(prev, axis="columns", columns=url_cols, preview=True)
                with session_lock:
                    st.session_state.last_preview = (preview_df, msg)
                st.info(msg)
                st.dataframe(_arrowize(preview_df.head(10)))
        with ur2:
            if st.button("📦 Add to Pipeline", key="add_url", help="Add URL handling step to the pipeline"):
                if not url_cols:
                    st.warning("Please select at least one column.")
                    return
                if url_action == "Extract Domain":
                    for col in url_cols:
                        step = {"kind": "extract_domain", "params": {"column": col, "new_name": url_new_name or None}}
                        with session_lock:
                            st.session_state.pipeline.append(step)
                    st.success(f"Added extract domain step for {len(url_cols)} columns to pipeline.")
                else:
                    step = {"kind": "drop_missing", "params": {"axis": "columns", "columns": url_cols}}
                    with session_lock:
                        st.session_state.pipeline.append(step)
                    st.success(f"Added drop step for {len(url_cols)} URL columns to pipeline.")
        with ur3:
            if st.button("🔄 Reset URL Selection", key="reset_url", help="Clear selected columns and options"):
                st.session_state["url_cols"] = []
                st.session_state["url_action"] = "Extract Domain"
                st.session_state["url_new_name"] = ""
                st.rerun()

        # Boolean-Disguised Handling
        st.subheader("Boolean-Disguised Handling")
        bool_cols = st.multiselect(
            "Boolean columns",
            [c for c in df.columns if pd.api.types.is_integer_dtype(df[c])],
            key="bool_cols",
            help="Select integer columns with 0/1 values to cast to boolean."
        )
        bo1, bo2, bo3 = st.columns([1, 1, 1])
        with bo1:
            if st.button("🔍 Preview Boolean Handling", key="preview_bool", help="Preview the effect on a sampled dataset"):
                if not bool_cols:
                    st.warning("Please select at least one column.")
                    return
                prev = sample_for_preview(df)
                preview_df, msg = type_convert(prev, bool_cols[0], "bool", preview=True)
                with session_lock:
                    st.session_state.last_preview = (preview_df, msg)
                st.info(msg)
                st.dataframe(_arrowize(preview_df.head(10)))
        with bo2:
            if st.button("📦 Add to Pipeline", key="add_bool", help="Add boolean conversion step to the pipeline"):
                if not bool_cols:
                    st.warning("Please select at least one column.")
                    return
                for col in bool_cols:
                    step = {"kind": "type_convert", "params": {"column": col, "type": "bool"}}
                    with session_lock:
                        st.session_state.pipeline.append(step)
                st.success(f"Added boolean conversion step for {len(bool_cols)} columns to pipeline.")
        with bo3:
            if st.button("🔄 Reset Boolean Selection", key="reset_bool", help="Clear selected columns"):
                st.session_state["bool_cols"] = []
                st.rerun()

        # Numeric-to-Categorical Handling
        st.subheader("Numeric-to-Categorical Handling")
        cat_cols = st.multiselect(
            "Numeric columns",
            [c for c in num_cols],
            key="num_cat_cols",
            help="Select numeric columns to convert to categorical."
        )
        nc1, nc2, nc3 = st.columns([1, 1, 1])
        with nc1:
            if st.button("🔍 Preview Categorical Conversion", key="preview_num_cat", help="Preview the effect on a sampled dataset"):
                if not cat_cols:
                    st.warning("Please select at least one column.")
                    return
                prev = sample_for_preview(df)
                preview_df, msg = type_convert(prev, cat_cols[0], "category", preview=True)
                with session_lock:
                    st.session_state.last_preview = (preview_df, msg)
                st.info(msg)
                st.dataframe(_arrowize(preview_df.head(10)))
        with nc2:
            if st.button("📦 Add to Pipeline", key="add_num_cat", help="Add categorical conversion step to the pipeline"):
                if not cat_cols:
                    st.warning("Please select at least one column.")
                    return
                for col in cat_cols:
                    step = {"kind": "type_convert", "params": {"column": col, "type": "category"}}
                    with session_lock:
                        st.session_state.pipeline.append(step)
                st.success(f"Added categorical conversion step for {len(cat_cols)} columns to pipeline.")
        with nc3:
            if st.button("🔄 Reset Categorical Selection", key="reset_num_cat", help="Clear selected columns"):
                st.session_state["num_cat_cols"] = []
                st.rerun()

    except Exception as e:
        logger.error(f"Error in section_inconsistency: {e}")
        st.error(f"Error in data inconsistency section: {e}")
