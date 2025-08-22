import logging
import streamlit as st
import pandas as pd

logger = logging.getLogger(__name__)

def init_session():
    """Initialize session state variables."""
    if "raw_df" not in st.session_state:
        st.session_state.raw_df = None  # the first loaded DataFrame
    if "df" not in st.session_state:
        st.session_state.df = None  # the working DataFrame
    if "history" not in st.session_state:
        st.session_state.history = []  # stack of (label, df_snapshot)
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = []  # list of step dicts
    if "changelog" not in st.session_state:
        st.session_state.changelog = []  # user-readable messages
    if "last_preview" not in st.session_state:
        st.session_state.last_preview = None  # cache preview results (df, summary)

def push_history(label: str):
    """Save a snapshot for undo, with a helpful label."""
    try:
        if st.session_state.df is not None:
            st.session_state.history.append((label, st.session_state.df.copy()))
    except Exception as e:
        logger.error(f"Error pushing history: {e}")
        st.error(f"Failed to save history snapshot: {e}")

def undo_last():
    """Undo the last applied step by restoring the previous snapshot."""
    try:
        if st.session_state.history:
            label, df_prev = st.session_state.history.pop()
            st.session_state.df = df_prev
            st.session_state.changelog.append(f"↩️ Undo: {label}")
            st.success(f"Undid: {label}")
        else:
            st.info("History is empty. Nothing to undo.")
    except Exception as e:
        logger.error(f"Error in undo_last: {e}")
        st.error(f"Failed to undo: {e}")

def reset_all():
    """Clear everything and start fresh."""
    try:
        st.session_state.raw_df = None
        st.session_state.df = None
        st.session_state.history = []
        st.session_state.pipeline = []
        st.session_state.changelog = []
        st.session_state.last_preview = None
        st.success("Reset all data and pipeline.")
    except Exception as e:
        logger.error(f"Error in reset_all: {e}")
        st.error(f"Failed to reset: {e}")
