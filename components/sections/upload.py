import logging
import streamlit as st
import pandas as pd
from utils.data_utils import _arrowize, sample_for_preview
from session import init_session

logger = logging.getLogger(__name__)

def section_upload():
    st.title("🧹 Data Preprocessing Studio")
    st.caption("Upload a CSV, chain preprocessing steps, preview changes, and download the cleaned dataset.")

    file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="CSV only. For large files, previews are sampled.",
    )

    if file:
        try:
            # 1. Load the file
            df = pd.read_csv(file)

            # 2. Reset all session state
            st.session_state.raw_df = df.copy()
            st.session_state.df = df.copy()
            st.session_state.history = []
            st.session_state.pipeline = []
            st.session_state.changelog = ["📥 Loaded dataset."]
            st.session_state.last_preview = None

            # 3. Notify the user
            st.success(f"Loaded dataset with shape {df.shape}.")

            # 4. Display a preview
            with st.expander("Peek at data", expanded=True):
                st.dataframe(_arrowize(sample_for_preview(df)))

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
