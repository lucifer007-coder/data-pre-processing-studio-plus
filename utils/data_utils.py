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
    if df is None or (isinstance(df, pd.DataFrame) and df.empty) or (isinstance(df, dd.DataFrame) and df.compute().empty):
        return df
    if isinstance(df, dd.DataFrame):
        df = df.compute()  # Convert to Pandas for Arrow serialization
    df_out = df.copy()
    for col in df_out.columns:
        s = df_out[col]
        inferred = pd.api.types.infer_dtype(s, skipna=True)
        if inferred in ("mixed", "string", "mixed-integer"):
            parsed = pd.to_numeric(s, errors="coerce")
            if parsed.notna().any():
                df_out[col] = parsed.astype("float64")
            else:
                df_out[col] = s.astype(str)
        elif inferred == "boolean":
            df_out[col] = s.astype(bool)
    return df_out

def sample_for_preview(df, n=PREVIEW_ROWS):
    """Safely sample DataFrame for preview purposes."""
    try:
        if df is None or (isinstance(df, pd.DataFrame) and df.empty) or (isinstance(df, dd.DataFrame) and df.compute().empty):
            return df
        if isinstance(df, dd.DataFrame):
            total_rows = df.shape[0].compute()
            if total_rows <= n:
                return df.compute()
            return df.sample(frac=n/total_rows, random_state=RANDOM_STATE).compute()
        if len(df) <= n:
            return df.copy()
        return df.sample(n=n, random_state=RANDOM_STATE).copy()
    except Exception as e:
        logger.error(f"Error in sample_for_preview: {e}")
        return df if df is not None else pd.DataFrame()

def dtype_split(df):
    """Return (numeric_columns, categorical_columns)."""
    if not isinstance(df, (pd.DataFrame, dd.DataFrame)):
        raise TypeError("Input must be a pandas or dask DataFrame")
    if isinstance(df, dd.DataFrame):
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.compute().tolist()
        categorical_cols = df.select_dtypes(exclude=['int64', 'float64']).columns.compute().tolist()
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return numeric_cols, categorical_cols
