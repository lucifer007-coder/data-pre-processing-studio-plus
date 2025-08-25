import json
import base64
import pandas as pd
import dask.dataframe as dd
import streamlit as st
import logging
from datetime import datetime
from io import StringIO

logger = logging.getLogger(__name__)

def export_bundle(sample_mode: bool = False) -> str:
    """
    Export session state to a JSON string.
    Args:
        sample_mode (bool): If True, include only first 5000 rows of raw_df.
    Returns:
        str: JSON string containing session state.
    """
    try:
        raw_df = st.session_state.get('raw_df')
        if raw_df is None:
            raise ValueError("No dataset loaded to export.")

        # Convert raw_df to CSV and encode as base64
        if isinstance(raw_df, dd.DataFrame):
            csv_buffer = raw_df.head(5000).compute().to_csv(index=False) if sample_mode else raw_df.compute().to_csv(index=False)
        else:
            csv_buffer = raw_df.head(5000).to_csv(index=False) if sample_mode else raw_df.to_csv(index=False)
        raw_csv = base64.b64encode(csv_buffer.encode()).decode()

        # Check size limit
        if len(raw_csv) > 10 * 1024 * 1024 and not sample_mode:  # 10 MB
            st.warning("Dataset exceeds 10 MB. Enable sample mode or export without raw data.")
            if st.button("Export without raw data"):
                raw_csv = ""
                st.session_state.raw_filename = "none.csv"

        # Build bundle
        bundle = {
            "version": "1.0",
            "raw_csv": raw_csv,
            "raw_filename": st.session_state.get('raw_filename', 'dataset.csv'),
            "pipeline": st.session_state.get('pipeline', []),
            "changelog": st.session_state.get('changelog', []),
            "semantic_map": st.session_state.get('semantic_map', {}),
            "branch": "main",
            "created": datetime.utcnow().isoformat() + "Z",
            "encoding": "utf-8",
            "delimiter": ","
        }

        # Convert to JSON
        json_str = json.dumps(bundle, ensure_ascii=False)
        return json_str
    except Exception as e:
        logger.error(f"Error exporting bundle: {e}")
        st.error(f"Error exporting bundle: {e}")
        raise

def import_bundle(json_str: str) -> bool:
    """
    Import session state from a JSON string.
    Args:
        json_str (str): JSON string of the bundle.
    Returns:
        bool: True if import succeeds, False otherwise.
    """
    try:
        if len(json_str) > 20 * 1024 * 1024:  # 20 MB
            logger.error("Bundle file is too large (>20 MB).")
            st.error("Bundle file is too large (>20 MB).")
            return False

        with st.spinner("Parsing bundle..."):
            bundle = json.loads(json_str)

        # Validate version
        if bundle.get("version") != "1.0":
            logger.warning(f"Unsupported bundle version: {bundle.get('version')}")
            st.error(f"Unsupported bundle version: {bundle.get('version')}")
            return False

        # Restore raw_df
        if bundle.get("raw_csv"):
            with st.spinner("Decoding dataset..."):
                decoded_csv = base64.b64decode(bundle["raw_csv"]).decode(bundle.get("encoding", "utf-8"))
                csv_io = StringIO(decoded_csv)
                file_size_mb = len(decoded_csv) / 1_000_000
                if file_size_mb < 100:
                    df = pd.read_csv(csv_io, encoding=bundle.get("encoding", "utf-8"), sep=bundle.get("delimiter", ","))
                else:
                    df = dd.read_csv(csv_io, encoding=bundle.get("encoding", "utf-8"), sep=bundle.get("delimiter", ","), blocksize="64MB")
                st.session_state.raw_df = df
        else:
            st.session_state.raw_df = None

        # Restore session state
        with st.spinner("Restoring session state..."):
            st.session_state.df = st.session_state.raw_df.copy() if st.session_state.raw_df is not None else None
            st.session_state.raw_filename = bundle.get("raw_filename", "dataset.csv")
            st.session_state.pipeline = bundle.get("pipeline", [])
            st.session_state.changelog = bundle.get("changelog", [])
            st.session_state.semantic_map = bundle.get("semantic_map", {})
            st.session_state.history = []
            st.session_state.last_preview = None
            st.session_state.just_imported_bundle = True

        logger.info(f"Bundle imported successfully: {st.session_state.raw_filename}")
        return True
    except Exception as e:
        logger.error(f"Error importing bundle: {e}")
        st.error(f"Error importing bundle: {e}")
        return False
