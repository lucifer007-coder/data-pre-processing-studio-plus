import logging
import streamlit as st
import pandas as pd
import dask.dataframe as dd
from utils.data_utils import _arrowize, sample_for_preview
from session import init_session
from utils.bundle_io import import_bundle
import threading

logger = logging.getLogger(__name__)

# Thread lock for session state updates
session_lock = threading.Lock()

def section_upload():
    st.title("🧹 Data Preprocessing Studio")
    st.caption("Upload a CSV or .dps bundle, chain preprocessing steps, preview changes, and download the cleaned dataset.")

    col1, col2 = st.columns(2)
    with col1:
        file = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
            help="Only CSV files are supported. Previews are sampled for large files.",
            key="data_file_uploader"
        )
    with col2:
        bundle_file = st.file_uploader(
            "Resume from .dps bundle",
            type=["dps"],
            help="Upload a .dps file to resume a previous session.",
            key="bundle_file_uploader"
        )

    if file:
        try:
            file_extension = file.name.split('.')[-1].lower()
            if file_extension != 'csv':
                st.error("Only CSV files are supported.")
                return
            with st.spinner("Loading CSV dataset..."):
                file_size_mb = file.size / 1_000_000
                if file_size_mb < 100:  # Use Pandas for files <100MB
                    df = pd.read_csv(file)
                else:  # Use Dask for larger files
                    df = dd.read_csv(file, assume_missing=True, blocksize="64MB")
                with session_lock:
                    st.session_state.df = df
                    st.session_state.raw_df = df
                    st.session_state.raw_filename = file.name
                shape = df.shape.compute() if isinstance(df, dd.DataFrame) else df.shape
                st.success(f"Loaded CSV: '{file.name}' with shape {shape}.")

        except pd.errors.EmptyDataError:
            logger.error("Empty CSV file.")
            st.error("The uploaded CSV file is empty. Please choose a non-empty CSV.")
        except pd.errors.ParserError as e:
            logger.error(f"CSV parsing failed: {e}")
            st.error(f"Could not parse CSV file. Check for formatting issues.\n\n{e}")
        except Exception as e:
            logger.error(f"Unexpected error while reading CSV: {e}")
            st.error(f"Unexpected error: {e}")

    elif bundle_file:
        try:
            with st.spinner("Loading bundle..."):
                bundle_str = bundle_file.getvalue().decode('utf-8')
                if st.session_state.get('df') is not None:
                    action = st.radio(
                        "Existing dataset detected. Replace with bundle data or keep?",
                        ["Replace", "Keep"],
                        key="bundle_replace_action"
                    )
                    if action == "Keep":
                        st.info("Keeping current dataset. Bundle not loaded.")
                        return

                if import_bundle(bundle_str):
                    st.session_state.just_imported_bundle = True
                    shape = st.session_state.df.shape.compute() if isinstance(st.session_state.df, dd.DataFrame) else st.session_state.df.shape
                    st.success(f"Loaded DPS bundle: '{st.session_state.raw_filename}' with shape {shape}. Navigating to Pipeline & Preview...")
                    st.session_state.section = "Pipeline & Preview"
                    st.rerun()
                else:
                    st.error("Failed to load bundle.")

        except Exception as e:
            logger.error(f"Error loading bundle: {e}")
            st.error(f"Error loading bundle: {e}")

    # Display current dataset in "Peek at data"
    if st.session_state.get('df') is not None:
        file_name = st.session_state.get('raw_filename', 'unknown file')
        file_type = 'CSV' if file_name.endswith('.csv') else 'DPS bundle'
        with st.expander("Peek at data", expanded=True):
            st.dataframe(_arrowize(sample_for_preview(st.session_state.df)))
        shape = st.session_state.df.shape.compute() if isinstance(st.session_state.df, dd.DataFrame) else st.session_state.df.shape
        st.info(f"Loaded {file_type}: '{file_name}' with shape {shape}. Upload a new file to replace it.")
    else:
        st.info("Please upload a CSV file or .dps bundle to get started.")
