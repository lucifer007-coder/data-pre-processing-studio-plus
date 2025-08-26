import json
import base64
import pandas as pd
import dask.dataframe as dd
import streamlit as st
import logging
import os
from datetime import datetime
from io import StringIO
from typing import Dict, List, Any, Union

logger = logging.getLogger(__name__)

# Constants
MAX_BUNDLE_SIZE = 15 * 1024 * 1024  # 15 MB unified limit
MAX_DECODED_SIZE = 50 * 1024 * 1024  # 50 MB for decoded content
DASK_THRESHOLD_MB = 100
SAMPLE_ROWS = 5000
SUPPORTED_VERSIONS = ["1.0"]
VALID_ENCODINGS = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
VALID_DELIMITERS = [",", ";", "\t", "|"]

def _sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks."""
    if not filename or not isinstance(filename, str):
        return "dataset.csv"
    
    # Remove path components and keep only the basename
    safe_filename = os.path.basename(filename)
    
    # Remove any remaining dangerous characters
    safe_chars = []
    for char in safe_filename:
        if char.isalnum() or char in ".-_":
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    
    result = "".join(safe_chars)
    
    # Ensure it's not empty and has a reasonable extension
    if not result or result.startswith("."):
        result = "dataset.csv"
    elif "." not in result:
        result += ".csv"
    
    # Limit length
    if len(result) > 255:
        result = result[:251] + ".csv"
    
    return result

def _validate_bundle_structure(bundle: Dict[str, Any]) -> bool:
    """Validate the structure and content of a bundle."""
    try:
        # Check required fields and types
        if not isinstance(bundle, dict):
            logger.error("Bundle must be a dictionary")
            return False
        
        # Validate version
        version = bundle.get("version")
        if version not in SUPPORTED_VERSIONS:
            logger.error(f"Unsupported version: {version}")
            return False
        
        # Validate pipeline
        pipeline = bundle.get("pipeline", [])
        if not isinstance(pipeline, list):
            logger.error("Pipeline must be a list")
            return False
        
        # Validate semantic_map
        semantic_map = bundle.get("semantic_map", {})
        if not isinstance(semantic_map, dict):
            logger.error("Semantic map must be a dictionary")
            return False
        
        # Validate changelog
        changelog = bundle.get("changelog", [])
        if not isinstance(changelog, list):
            logger.error("Changelog must be a list")
            return False
        
        # Validate encoding
        encoding = bundle.get("encoding", "utf-8")
        if encoding not in VALID_ENCODINGS:
            logger.error(f"Unsupported encoding: {encoding}")
            return False
        
        # Validate delimiter
        delimiter = bundle.get("delimiter", ",")
        if delimiter not in VALID_DELIMITERS:
            logger.error(f"Unsupported delimiter: {delimiter}")
            return False
        
        # Validate raw_csv if present
        raw_csv = bundle.get("raw_csv", "")
        if raw_csv and not isinstance(raw_csv, str):
            logger.error("raw_csv must be a string")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating bundle structure: {e}")
        return False

def _safe_dataframe_copy(df: Union[pd.DataFrame, dd.DataFrame]) -> Union[pd.DataFrame, dd.DataFrame, None]:
    """Safely copy a DataFrame, handling both pandas and dask DataFrames."""
    if df is None:
        return None
    
    try:
        if isinstance(df, dd.DataFrame):
            # For Dask DataFrames, we need to create a new reference
            # since copy() method doesn't exist
            return df.persist() if hasattr(df, 'persist') else df
        else:
            # For pandas DataFrames
            return df.copy()
    except Exception as e:
        logger.error(f"Error copying DataFrame: {e}")
        return df  # Return original if copy fails

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
            csv_buffer = raw_df.head(SAMPLE_ROWS).compute().to_csv(index=False) if sample_mode else raw_df.compute().to_csv(index=False)
        else:
            csv_buffer = raw_df.head(SAMPLE_ROWS).to_csv(index=False) if sample_mode else raw_df.to_csv(index=False)
        
        # Use explicit encoding for consistency
        csv_bytes = csv_buffer.encode('utf-8')
        raw_csv = base64.b64encode(csv_bytes).decode('ascii')

        # Check size limit with proper error handling
        if len(raw_csv) > MAX_BUNDLE_SIZE and not sample_mode:
            error_msg = f"Dataset exceeds {MAX_BUNDLE_SIZE // (1024*1024)} MB. Please enable sample mode to export first {SAMPLE_ROWS} rows only."
            logger.warning(error_msg)
            st.warning(error_msg)
            # Return empty string to indicate size limit exceeded
            # UI should handle this case separately
            return ""

        # Sanitize filename
        raw_filename = st.session_state.get('raw_filename', 'dataset.csv')
        safe_filename = _sanitize_filename(raw_filename)

        # Build bundle with validated data
        bundle = {
            "version": "1.0",
            "raw_csv": raw_csv,
            "raw_filename": safe_filename,
            "pipeline": st.session_state.get('pipeline', []),
            "changelog": st.session_state.get('changelog', []),
            "semantic_map": st.session_state.get('semantic_map', {}),
            "branch": "main",
            "created": datetime.utcnow().isoformat() + "Z",
            "encoding": "utf-8",
            "delimiter": ","
        }

        # Convert to JSON with proper error handling
        json_str = json.dumps(bundle, ensure_ascii=False, separators=(',', ':'))
        
        # Final size check
        if len(json_str.encode('utf-8')) > MAX_BUNDLE_SIZE:
            error_msg = f"Final bundle exceeds {MAX_BUNDLE_SIZE // (1024*1024)} MB limit."
            logger.error(error_msg)
            st.error(error_msg)
            return ""
        
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
        # Input validation
        if not isinstance(json_str, str):
            logger.error("Bundle must be a string")
            st.error("Invalid bundle format")
            return False
        
        if len(json_str.encode('utf-8')) > MAX_BUNDLE_SIZE:
            logger.error(f"Bundle file is too large (>{MAX_BUNDLE_SIZE // (1024*1024)} MB).")
            st.error(f"Bundle file is too large (>{MAX_BUNDLE_SIZE // (1024*1024)} MB).")
            return False

        with st.spinner("Parsing bundle..."):
            try:
                bundle = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON format: {e}")
                st.error("Invalid bundle format: corrupted JSON")
                return False

        # Validate bundle structure and content
        if not _validate_bundle_structure(bundle):
            st.error("Invalid bundle structure")
            return False

        # Validate version
        if bundle.get("version") not in SUPPORTED_VERSIONS:
            logger.warning(f"Unsupported bundle version: {bundle.get('version')}")
            st.error(f"Unsupported bundle version: {bundle.get('version')}")
            return False

        # Restore raw_df with enhanced security checks
        if bundle.get("raw_csv"):
            with st.spinner("Decoding dataset..."):
                try:
                    # Validate base64 and check decoded size
                    raw_csv_data = bundle["raw_csv"]
                    if not isinstance(raw_csv_data, str):
                        raise ValueError("raw_csv must be a string")
                    
                    # Estimate decoded size (base64 expands by ~33%)
                    estimated_decoded_size = len(raw_csv_data) * 3 // 4
                    if estimated_decoded_size > MAX_DECODED_SIZE:
                        raise ValueError(f"Decoded data would exceed {MAX_DECODED_SIZE // (1024*1024)} MB limit")
                    
                    decoded_csv = base64.b64decode(raw_csv_data).decode(bundle.get("encoding", "utf-8"))
                    
                    # Additional size check after decoding
                    if len(decoded_csv.encode('utf-8')) > MAX_DECODED_SIZE:
                        raise ValueError(f"Decoded CSV exceeds {MAX_DECODED_SIZE // (1024*1024)} MB limit")
                    
                    csv_io = StringIO(decoded_csv)
                    file_size_mb = len(decoded_csv) / 1_000_000
                    
                    # Use validated encoding and delimiter
                    encoding = bundle.get("encoding", "utf-8")
                    delimiter = bundle.get("delimiter", ",")
                    
                    if file_size_mb < DASK_THRESHOLD_MB:
                        df = pd.read_csv(csv_io, encoding=encoding, sep=delimiter)
                    else:
                        # For large files, use Dask
                        csv_io.seek(0)  # Reset StringIO position
                        df = dd.read_csv(csv_io, encoding=encoding, sep=delimiter, blocksize="64MB")
                    
                    st.session_state.raw_df = df
                    
                except (base64.binascii.Error, UnicodeDecodeError, ValueError) as e:
                    logger.error(f"Error decoding CSV data: {e}")
                    st.error(f"Error decoding dataset: {e}")
                    return False
        else:
            st.session_state.raw_df = None

        # Restore session state with validated data
        with st.spinner("Restoring session state..."):
            st.session_state.df = _safe_dataframe_copy(st.session_state.raw_df)
            st.session_state.raw_filename = _sanitize_filename(bundle.get("raw_filename", "dataset.csv"))
            
            # Validate and restore complex objects
            pipeline = bundle.get("pipeline", [])
            if isinstance(pipeline, list):
                st.session_state.pipeline = pipeline
            else:
                st.session_state.pipeline = []
            
            changelog = bundle.get("changelog", [])
            if isinstance(changelog, list):
                st.session_state.changelog = changelog
            else:
                st.session_state.changelog = []
            
            semantic_map = bundle.get("semantic_map", {})
            if isinstance(semantic_map, dict):
                st.session_state.semantic_map = semantic_map
            else:
                st.session_state.semantic_map = {}
            
            # Reset other session state variables
            st.session_state.history = []
            st.session_state.last_preview = None
            st.session_state.just_imported_bundle = True

        logger.info(f"Bundle imported successfully: {st.session_state.raw_filename}")
        return True
        
    except Exception as e:
        logger.error(f"Error importing bundle: {e}")
        st.error(f"Error importing bundle: {e}")
        return False
