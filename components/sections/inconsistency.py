import logging
import streamlit as st
from utils.data_utils import dtype_split, _arrowize, sample_for_preview
from preprocessing.steps import normalize_text, standardize_dates, unit_convert

logger = logging.getLogger(__name__)

def section_inconsistency():
    st.header("📏 Data Inconsistency")
    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first.")
        return

    try:
        num_cols, cat_cols = dtype_split(df)

        st.subheader("Text Normalization")
        text_cols = st.multiselect(
            "Text columns",
            [c for c in df.columns if c in cat_cols],
            help="Select columns for text normalization."
        )
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            lower = st.checkbox("Lowercase", True, help="Convert text to lowercase.")
        with c2:
            trim = st.checkbox("Trim spaces", True, help="Remove leading/trailing spaces.")
        with c3:
            collapse = st.checkbox("Collapse spaces", True, help="Replace multiple spaces with a single space.")
        with c4:
            remove_special = st.checkbox("Remove special chars", False, help="Remove non-alphanumeric characters.")
        cc1, cc2, cc3 = st.columns([1, 1, 1])
        with cc1:
            if st.button("🔍 Preview Text Normalization", help="Preview the effect on a sampled dataset"):
                if not text_cols:
                    st.warning("Please select at least one column.")
                    return
                prev = sample_for_preview(df)
                preview_df, msg = normalize_text(
                    prev, text_cols, lower, trim, collapse, remove_special, preview=True
                )
                st.session_state.last_preview = (preview_df, msg)
                st.info(msg)
                st.dataframe(_arrowize(preview_df.head(10)))
        with cc2:
            if st.button("📦 Add to Pipeline", help="Add text normalization step to the pipeline"):
                if not text_cols:
                    st.warning("Please select at least one column.")
                    return
                step = {
                    "kind": "normalize_text",
                    "params": {
                        "columns": text_cols,
                        "lowercase": lower,
                        "trim": trim,
                        "collapse_spaces": collapse,
                        "remove_special": remove_special
                    }
                }
                st.session_state.pipeline.append(step)
                st.success("Added text normalization step to pipeline.")
        with cc3:
            if st.button("🔄 Reset Selection", help="Clear selected columns and options"):
                st.session_state["text_cols"] = []
                st.rerun()

        st.subheader("Date Standardization")
        date_cols = st.multiselect(
            "Date-like columns",
            df.columns.tolist(),
            help="Select columns containing date strings."
        )
        fmt = st.text_input(
            "Output date format",
            "%Y-%m-%d",
            help="Enter a Python datetime format (e.g., %Y-%m-%d for YYYY-MM-DD)."
        )
        dc1, dc2, dc3 = st.columns([1, 1, 1])
        with dc1:
            if st.button("🔍 Preview Date Standardization", help="Preview the effect on a sampled dataset"):
                if not date_cols:
                    st.warning("Please select at least one column.")
                    return
                try:
                    pd.to_datetime("2023-01-01", format=fmt)
                except ValueError:
                    st.error("Invalid date format.")
                    return
                prev = sample_for_preview(df)
                preview_df, msg = standardize_dates(prev, date_cols, fmt, preview=True)
                st.session_state.last_preview = (preview_df, msg)
                st.info(msg)
                st.dataframe(_arrowize(preview_df.head(10)))
        with dc2:
            if st.button("📦 Add to Pipeline", help="Add date standardization step to the pipeline"):
                if not date_cols:
                    st.warning("Please select at least one column.")
                    return
                try:
                    pd.to_datetime("2023-01-01", format=fmt)
                except ValueError:
                    st.error("Invalid date format.")
                    return
                step = {"kind": "standardize_dates", "params": {"columns": date_cols, "output_format": fmt}}
                st.session_state.pipeline.append(step)
                st.success("Added date standardization step to pipeline.")
        with dc3:
            if st.button("🔄 Reset Selection", help="Clear selected columns and format"):
                st.session_state["date_cols"] = []
                st.rerun()

        st.subheader("Unit Conversion")
        uc_col = st.selectbox(
            "Column to convert",
            ["(none)"] + num_cols,
            help="Select a numeric column to convert units."
        )
        colA, colB, colC = st.columns(3)
        with colA:
            factor = st.number_input(
                "Multiply by factor",
                value=1.0, step=0.1,
                help="Enter the conversion factor (e.g., 0.001 for km to m)."
            )
        with colB:
            new_name = st.text_input(
                "New column name (optional)",
                "",
                help="Leave blank to overwrite the original column."
            )
        with colC:
            st.caption("Tip: Use this to create normalized units (e.g., convert km to m).")
        uc1, uc2, uc3 = st.columns([1, 1, 1])
        with uc1:
            if st.button("🔍 Preview Unit Conversion", help="Preview the effect on a sampled dataset"):
                if uc_col == "(none)":
                    st.warning("Please select a column.")
                    return
                if factor == 0:
                    st.error("Factor cannot be zero.")
                    return
                prev = sample_for_preview(df)
                preview_df, msg = unit_convert(prev, uc_col, factor, new_name or None, preview=True)
                st.session_state.last_preview = (preview_df, msg)
                st.info(msg)
                st.dataframe(_arrowize(preview_df.head(10)))
        with uc2:
            if st.button("📦 Add to Pipeline", help="Add unit conversion step to the pipeline"):
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
                st.session_state.pipeline.append(step)
                st.success("Added unit conversion step to pipeline.")
        with uc3:
            if st.button("🔄 Reset Selection", help="Clear selected column and options"):
                st.session_state["uc_col"] = "(none)"
                st.rerun()

    except Exception as e:
        logger.error(f"Error in section_inconsistency: {e}")
        st.error(f"Error in data inconsistency section: {e}")
