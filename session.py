import logging
import streamlit as st
import pandas as pd
from typing import Optional, Tuple, List
import gc
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def init_session() -> None:
    """
    Initialize session state variables with default values.

    Ensures all necessary session state keys are set to prevent KeyError issues.
    """
    default_state = {
        "raw_df": None,  # Original loaded DataFrame
        "df": None,      # Working DataFrame
        "history": [],   # Stack of (label, df_snapshot) tuples
        "pipeline": [],  # List of transformation step dictionaries
        "changelog": [], # User-readable log messages
        "last_preview": None  # Cached preview results (df, summary)
    }
    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = value
    logger.info("Session state initialized.")

def push_history(label: str) -> None:
    """
    Save a snapshot of the current DataFrame to the history stack.

    Args:
        label (str): Descriptive label for the snapshot.

    Raises:
        ValueError: If the DataFrame is not a valid pandas DataFrame.
        MemoryError: If memory allocation fails during copying.
    """
    try:
        if st.session_state.df is not None:
            if not isinstance(st.session_state.df, pd.DataFrame):
                raise ValueError("Current DataFrame is not a valid pandas DataFrame.")
            st.session_state.history.append((label, st.session_state.df.copy(deep=True)))
            st.session_state.changelog.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saved snapshot: {label}")
            logger.info(f"History snapshot saved: {label}")
    except (ValueError, MemoryError) as e:
        logger.error(f"Error pushing history: {e}")
        st.error(f"Failed to save history snapshot: {e}")
    except Exception as e:
        logger.error(f"Unexpected error pushing history: {e}")
        st.error(f"Unexpected error saving history snapshot: {e}")

def undo_last() -> None:
    """
    Undo the last applied step by restoring the previous DataFrame snapshot.

    Restores the most recent DataFrame from history and updates the changelog.
    If history is empty, informs the user.

    Raises:
        ValueError: If the history snapshot is invalid.
    """
    try:
        if not st.session_state.history:
            st.info("History is empty. Nothing to undo.")
            logger.info("Undo attempted but history is empty.")
            return