import logging
import streamlit as st
from utils.data_utils import dtype_split, _arrowize, sample_for_preview
from preprocessing.steps import normalize_text, standardize_dates, unit_convert

logger = logging.getLogger(__name__)

def section_inconsistency():
    st.header("2) Data Inconsistency")
    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first.")
        return

    try:
        num_cols, cat_cols = dtype_split(df)

        st.subheader("Text Normalization")
        text_cols = st.multiselect("Text columns", [c for c in df.columns if c in cat_cols])
        c1, c2, c3 = st.columns(3)
        with c1:
            lower = st.checkbox("lowercase", True)
        with c2:
            trim = st.checkbox("trim spaces", True)
        with c3:
            collapse = st.checkbox("collapse multiple spaces", True)
        cc1, cc2 = st.columns(2)
        with cc1:
            if st.button("🔍 Preview Text Normalization"):
                prev = sample_for_preview(df)
                preview_df, msg = normalize_text(prev, text_cols, lower, trim, collapse)
                st.session_state.last_preview = (preview_df, msg)
                st.info(msg)
                st.dataframe(_arrowize(preview_df))
        with cc2:
            if st.button("📦 Add to Pipeline (Normalize Text)"):
                step = {
                    "kind": "normalize_text",
                    "params": {"columns": text_cols, "lowercase": lower, "trim": trim, "collapse_spaces": collapse},
                }
                st.session_state.pipeline.append(step)
                st.success("Added to pipeline.")

        st.subheader("Date Standardization")
        date_cols = st.multiselect("Date-like columns", df.columns.tolist(), help="Columns that contain date strings.")
        fmt = st.text_input("Output date format", "%Y-%m-%d", help="Python datetime format string.")
        dc1, dc2 = st.columns(2)
        with dc1:
            if st.button("🔍 Preview Date Standardization"):
                prev = sample_for_preview(df)
                preview_df, msg = standardize_dates(prev, date_cols, fmt)
                st.session_state.last_preview = (preview_df, msg)
                st.info(msg)
                st.dataframe(_arrowize(preview_df))
        with dc2:
            if st.button("📦 Add to Pipeline (Standardize Dates)"):
                step = {"kind": "standardize_dates", "params": {"columns": date_cols, "output_format": fmt}}
                st.session_state.pipeline.append(step)
                st.success("Added to pipeline.")

        st.subheader("Unit Conversion")
        uc_col = st.selectbox("Column to convert", ["(none)"] + df.columns.tolist())
        colA, colB, colC = st.columns(3)
        with colA:
            factor = st.number_input("Multiply by factor", value=1.0, step=0.1)
        with colB:
            new_name = st.text_input("New column name (optional)", "")
        with colC:
            st.caption("Tip: Use this to create normalized units.")
        uc1, uc2 = st.columns(2)
        with uc1:
            if st.button("🔍 Preview Unit Conversion"):
                prev = sample_for_preview(df)
                if uc_col != "(none)":
                    preview_df, msg = unit_convert(prev, uc_col, factor, new_name or None)
                else:
                    preview_df, msg = prev, "No column selected."
                st.session_state.last_preview = (preview_df, msg)
                st.info(msg)
                st.dataframe(_arrowize(preview_df))
        with uc2:
            if st.button("📦 Add to Pipeline (Unit Convert)"):
                if uc_col == "(none)":
                    st.warning("Please select a column.")
                else:
                    step = {
                        "kind": "unit_convert",
                        "params": {"column": uc_col, "factor": factor, "new_name": new_name or None},
                    }
                    st.session_state.pipeline.append(step)
                    st.success("Added to pipeline.")
    except Exception as e:
        logger.error(f"Error in section_inconsistency: {e}")
        st.error(f"Error in data inconsistency section: {e}")
