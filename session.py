import streamlit as st
import pandas as pd
import dask.dataframe as dd
import threading

def init_session():
    """Initialize session state with default values."""
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'raw_df' not in st.session_state:
        st.session_state.raw_df = None
    if 'raw_filename' not in st.session_state:
        st.session_state.raw_filename = None
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = []
    if 'changelog' not in st.session_state:
        st.session_state.changelog = []
    if 'last_preview' not in st.session_state:
        st.session_state.last_preview = None
    if 'semantic_map' not in st.session_state:
        st.session_state.semantic_map = {}
    if 'section' not in st.session_state:
        st.session_state.section = "Upload"
    if 'just_imported_bundle' not in st.session_state:
        st.session_state.just_imported_bundle = False
    # Initialize session lock for thread safety
    if 'session_lock' not in st.session_state:
        st.session_state.session_lock = threading.Lock()

def reset_all():
    """Reset all session state."""
    st.session_state.df = None
    st.session_state.raw_df = None
    st.session_state.raw_filename = None
    st.session_state.history = []
    st.session_state.pipeline = []
    st.session_state.changelog = []
    st.session_state.last_preview = None
    st.session_state.semantic_map = {}
    st.session_state.section = "Upload"
    st.session_state.just_imported_bundle = False
    # Reinitialize session lock after reset
    st.session_state.session_lock = threading.Lock()

def undo_last():
    """Undo the last applied step."""
    if st.session_state.history:
        st.session_state.df = st.session_state.history.pop()
        st.session_state.changelog.append("↩️ Undid last action.")

def push_history(description: str):
    """
    Save the current DataFrame state to history.
    Args:
        description (str): Description of the action for changelog.
    """
    if st.session_state.get('df') is not None:
        if isinstance(st.session_state.df, dd.DataFrame):
            st.session_state.history.append(st.session_state.df.copy())
        else:
            st.session_state.history.append(st.session_state.df.copy())
        st.session_state.changelog.append(f"📝 {description}")
    else:
        st.warning("No dataset to save in history.")
