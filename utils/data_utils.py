import pandas as pd
import numpy as np
from typing import List, Tuple
from ..config import RANDOM_STATE, PREVIEW_ROWS

def _arrowize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a *shallow* copy of `df` whose column dtypes are guaranteed to
    serialize to Arrow without raising ArrowInvalid.
    """
    if df is None or df.empty:
        return df

    df_out = df.copy()
    for col in df_out.columns:
        s = df_out[col]
        inferred = pd.api.types.infer_dtype(s, skipna=True)

        # CASE 1: numeric column that slipped into 'object' dtype
        # (usually because of strings such as 'NA', 'NULL', etc.)
        if inferred in ("mixed", "string", "mixed-integer"):
            # Try to parse as float; unparseable values become NaN
            parsed = pd.to_numeric(s, errors="coerce")
            # If at least one value survived, cast the column
            if parsed.notna().any():
                df_out[col] = parsed.astype("float64")
            else:
                # Fallback: plain string
                df_out[col] = s.astype(str)

        # CASE 2: Boolean columns might be "True"/"False" strings
        elif inferred == "boolean":
            df_out[col] = s.astype(bool)

        # CASE 3: Everything else is kept as-is
        else:
            pass

    return df_out

def sample_for_preview(df: pd.DataFrame, n: int = PREVIEW_ROWS) -> pd.DataFrame:
    """Safely sample DataFrame for preview purposes."""
    try:
        if df is None or df.empty:
            return df
        if len(df) <= n:
            return df.copy()
        return df.sample(n=n, random_state=RANDOM_STATE).copy()
    except Exception as e:
        logger.error(f"Error in sample_for_preview: {e}")
        return df if df is not None else pd.DataFrame()

def dtype_split(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return (numeric_columns, categorical_columns)."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return numeric_cols, categorical_cols