import pandas as pd
import numpy as np
import dask.dataframe as dd
from config import RANDOM_STATE, PREVIEW_ROWS
import logging

logger = logging.getLogger(__name__)

def _arrowize(df):
    """
    Return a *shallow* copy of `df` whose column dtypes are guaranteed to
    serialize to Arrow without raising ArrowInvalid.
    """
    if df is None:
        return df
    
    # Convert Dask DataFrame to Pandas once
    if isinstance(df, dd.DataFrame):
        df = df.compute()
    
    # Check if empty after conversion
    if df.empty:
        return df
        
    df_out = df.copy()
    for col in df_out.columns:
        s = df_out[col]
        inferred = pd.api.types.infer_dtype(s, skipna=True)
        if inferred in ("mixed", "string", "mixed-integer"):
            parsed = pd.to_numeric(s, errors="coerce")
            # Log data loss for transparency
            lost_values = s[parsed.isna() & s.notna()]
            if not lost_values.empty:
                logger.warning(f"Column {col}: Converting {len(lost_values)} non-numeric values to NaN")
            
            if parsed.notna().any():
                df_out[col] = parsed.astype("float64")
            else:
                df_out[col] = s.astype(str)
        elif inferred == "boolean":
            df_out[col] = s.astype(bool)
    return df_out

def sample_for_preview(df, n=PREVIEW_ROWS):
    """Safely sample DataFrame for preview purposes."""
    # Input validation
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    
    try:
        if df is None:
            return df
            
        # Handle Dask DataFrame with memory protection
        if isinstance(df, dd.DataFrame):
            total_rows = df.shape[0].compute()
            
            # Check if empty
            if total_rows == 0:
                return df.compute()
            
            # Memory protection for very large datasets
            max_memory_rows = 1000000
            if total_rows > max_memory_rows:
                logger.warning(f"Dataset has {total_rows} rows, using head() instead of full computation")
                return df.head(n)
            
            if total_rows <= n:
                return df.compute()
            
            # Safe division - total_rows is guaranteed > 0 here
            return df.sample(frac=n/total_rows, random_state=RANDOM_STATE).compute()
        
        # Handle Pandas DataFrame
        if df.empty:
            return df
            
        if len(df) <= n:
            return df.copy()
        return df.sample(n=n, random_state=RANDOM_STATE).copy()
        
    except (ValueError, KeyError, MemoryError) as e:
        logger.error(f"Error in sample_for_preview: {e}")
        return df if df is not None else pd.DataFrame()
    except Exception as e:
        logger.critical(f"Unexpected error in sample_for_preview: {e}")
        raise  # Re-raise unexpected errors

def dtype_split(df):
    """Return (numeric_columns, categorical_columns)."""
    if not isinstance(df, (pd.DataFrame, dd.DataFrame)):
        raise TypeError("Input must be a pandas or dask DataFrame")
    
    # Use consistent type handling for both Dask and Pandas
    if isinstance(df, dd.DataFrame):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.compute().tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.compute().tolist()
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return numeric_cols, categorical_cols
