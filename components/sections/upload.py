import logging
import streamlit as st
import pandas as pd
from utils.data_utils import _arrowize, sample_for_preview
from session import init_session

logger = logging.getLogger(__name__)

def section_upload():
    st.title("🧹 Data Preprocessing Studio")
    st.caption("Upload a CSV, chain preprocessing steps, preview changes, and download the cleaned dataset.")

    # Check if a dataset already exists in session state
    if st.session_state.get('df') is not None:
        with st.expander("Peek at data", expanded=True):
            st.dataframe(_arrowize(sample_for_preview(st.session_state.df)))
        st.info(f"Dataset loaded with shape {st.session_state.df.shape}. Upload a new CSV to replace it.")

    file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="CSV only. For large files, previews are sampled.",
        key="csv_uploader"  # Ensure unique key to avoid widget conflicts
    )

    if file:
        try:
            # Load the file
            df = pd.read_csv(file)

            # Reset session state
            st.session_state.raw_df = df.copy()
            st.session_state.df = df.copy()
            st.session_state.history = []
            st.session_state.pipeline = []
            st.session_state.changelog = ["📥 Loaded dataset."]
            st.session_state.last_preview = None

            # Notify the user
            st.success(f"Loaded dataset with shape {df.shape}.")

            # Display preview
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
    elif st.session_state.get('df') is None:
        st.info("Please upload a CSV file to get started.")