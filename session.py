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
    try:
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
    except Exception as e:
        logger.error(f"Error initializing session: {e}")
        st.error(f"Failed to initialize session: {e}")

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
        label, df_prev = st.session_state.history.pop()
        if not isinstance(df_prev, pd.DataFrame):
            raise ValueError("Invalid history snapshot: not a pandas DataFrame.")
        st.session_state.df = df_prev
        changelog_msg = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ↩️ Undid: {label}"
        st.session_state.changelog.append(changelog_msg)
        st.success(f"Undid: {label}")
        logger.info(f"Undo successful: {label}")
        st.rerun()  # Refresh the app to reflect state changes
    except ValueError as e:
        logger.error(f"Error in undo_last: {e}")
        st.error(f"Failed to undo: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in undo_last: {e}")
        st.error(f"Unexpected error during undo: {e}")

def reset_all() -> None:
    """
    Clear all session state data and reset the application.

    Resets all session state variables to their initial values and clears memory.

    Raises:
        Exception: If an unexpected error occurs during reset.
    """
    try:
        init_session()  # Reinitialize session state
        gc.collect()  # Force garbage collection to free memory
        st.session_state.changelog.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Reset all data and pipeline.")
        st.success("Reset all data and pipeline.")
        logger.info("Application reset successfully.")
        st.rerun()  # Refresh the app to reflect reset
    except Exception as e:
        logger.error(f"Error in reset_all: {e}")
        st.error(f"Failed to reset: {e}")

# Ensure no module-level Streamlit calls
if __name__ == "__main__":
    logger.info("session.py loaded successfully.")