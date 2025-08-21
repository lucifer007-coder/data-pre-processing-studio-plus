import pandas as pd
import numpy as np
from typing import List, Tuple
from config import RANDOM_STATE, PREVIEW_ROWS

def _arrowize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a *shallow* copy whose dtypes are:
      - pyarrow-backed string[pyarrow] for object columns
      - numeric down-casted
    """
    if df is None or df.empty:
        return df

    df_out = df.copy()

    # 1. Force pyarrow-backed dtypes
    for col in df_out.columns:
        s = df_out[col]
        inferred = pd.api.types.infer_dtype(s, skipna=True)

        # numeric slipped into object
        if inferred in ("mixed", "string", "mixed-integer"):
            parsed = pd.to_numeric(s, errors="coerce")
            if parsed.notna().any():
                df_out[col] = parsed.astype("float64")
            else:
                df_out[col] = s.astype("string[pyarrow]")

        # boolean strings
        elif inferred == "boolean":
            df_out[col] = s.astype(bool)

        # already numeric → down-cast
        elif pd.api.types.is_integer_dtype(s):
            df_out[col] = pd.to_numeric(s, downcast="integer")
        elif pd.api.types.is_float_dtype(s):
            df_out[col] = pd.to_numeric(s, downcast="float")

        # everything else → pyarrow string
        elif s.dtype == "object":
            df_out[col] = s.astype("string[pyarrow]")

    return df_out.convert_dtypes(dtype_backend="pyarrow")


def sample_for_preview(df: pd.DataFrame, n: int = PREVIEW_ROWS) -> pd.DataFrame:
    """Return at most `n` rows (or full frame) for safe preview."""
    try:
        if df is None or df.empty:
            return df
        if len(df) <= n:
            return df.copy()
        return df.sample(n=n, random_state=RANDOM_STATE).copy()
    except Exception:
        return df if df is not None else pd.DataFrame()


def dtype_split(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return (numeric_columns, categorical_columns) using pyarrow dtypes."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return numeric_cols, categorical_cols
