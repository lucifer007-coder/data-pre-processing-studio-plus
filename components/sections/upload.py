import logging
import streamlit as st
import pandas as pd
from utils.data_utils import _arrowize, sample_for_preview
from session import init_session

logger = logging.getLogger(__name__)

@st.cache_data(show_spinner="Reading file …", ttl=600)
def _load_csv(file_obj, max_rows=None):
    """
    Load CSV with optional row limit.  Uses pyarrow backend.
    """
    if max_rows:
        df = pd.read_csv(file_obj, nrows=max_rows, engine="pyarrow")
    else:
        df = pd.read_csv(file_obj, engine="pyarrow")
    return _arrowize(df)

def section_upload():
    st.title("🧹 Data Preprocessing Studio")
    st.caption("Upload a CSV, chain preprocessing steps, preview changes, and download the cleaned dataset.")

    file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="CSV only.  Large files are sampled by default.",
    )

    if file:
        preview_only = st.checkbox("Fast preview (first 50 000 rows)", value=True)
        try:
            df = _load_csv(file, max_rows=50_000 if preview_only else None)

            # Reset session
            st.session_state.raw_df = df.copy()
            st.session_state.df = df.copy()
            st.session_state.history = []
            st.session_state.pipeline = []
            st.session_state.changelog = ["📥 Loaded dataset."]
            st.session_state.last_preview = None

            st.success(f"Loaded dataset with shape {df.shape}.")

            with st.expander("Peek at data", expanded=True):
                st.dataframe(sample_for_preview(df))

        except pd.errors.EmptyDataError:
            logger.error("Empty CSV file.")
            st.error("The uploaded file is empty. Please choose a non-empty CSV.")
        except pd.errors.ParserError as e:
            logger.error(f"CSV parsing failed: {e}")
            st.error(f"Could not parse CSV file. Check for formatting issues.\n\n{e}")
        except Exception as e:
            logger.error(f"Unexpected error while reading CSV: {e}")
            st.error(f"Unexpected error: {e}")
    else:
        st.info("Please upload a CSV file to get started.")
