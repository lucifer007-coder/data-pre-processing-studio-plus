import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import streamlit as st
from utils.data_utils import dtype_split

logger = logging.getLogger(__name__)

@st.cache_data
def compute_basic_stats(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Compute comprehensive basic statistics for a DataFrame."""
    if df is None:
        logger.warning("DataFrame is None, returning empty statistics")
        return {
            "shape": (0, 0),
            "columns": [],
            "dtypes": {},
            "missing_total": 0,
            "missing_by_col": {},
            "numeric_cols": [],
            "categorical_cols": [],
            "describe_numeric": {},
            "memory_usage_mb": 0.0,
            "duplicate_rows": 0
        }

    if not isinstance(df, pd.DataFrame):
        logger.error(f"Expected pandas DataFrame or None, got {type(df)}")
        st.error(f"Invalid data type: expected DataFrame, got {type(df)}")
        return {}

    try:
        num_cols, cat_cols = dtype_split(df)
        missing_series = df.isna().sum()
        missing_total = int(missing_series.sum())
        missing_by_col = {k: int(v) for k, v in missing_series.sort_values(ascending=False).to_dict().items()}

        describe_numeric = {}
        if num_cols:
            numeric_df = df[num_cols]
            if len(numeric_df) > 1_000_000:
                numeric_df = numeric_df.sample(n=min(100_000, len(numeric_df)), random_state=42)
            describe_numeric = numeric_df.describe().to_dict()
            for col in describe_numeric:
                for stat in describe_numeric[col]:
                    val = describe_numeric[col][stat]
                    if pd.isna(val):
                        describe_numeric[col][stat] = None
                    elif isinstance(val, (np.integer, np.floating)):
                        describe_numeric[col][stat] = float(val) if np.isfinite(val) else None

        memory_usage_mb = round(float(df.memory_usage(deep=True).sum() / (1024 * 1024)), 2)
        duplicate_rows = int(df.duplicated().sum())

        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_total": missing_total,
            "missing_by_col": missing_by_col,
            "numeric_cols": num_cols,
            "categorical_cols": cat_cols,
            "describe_numeric": describe_numeric,
            "memory_usage_mb": memory_usage_mb,
            "duplicate_rows": duplicate_rows
        }
    except Exception as e:
        logger.error(f"Unexpected error in compute_basic_stats: {e}")
        return {
            "shape": (0, 0),
            "columns": [],
            "dtypes": {},
            "missing_total": 0,
            "missing_by_col": {},
            "numeric_cols": [],
            "categorical_cols": [],
            "describe_numeric": {},
            "memory_usage_mb": 0.0,
            "duplicate_rows": 0,
            "error": str(e)
        }

def compare_stats(before: Optional[Dict[str, Any]], after: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare statistics between two DataFrames (before and after processing)."""
    if before is None:
        before = {}
    if after is None:
        after = {}
    if not isinstance(before, dict):
        before = {}
    if not isinstance(after, dict):
        after = {}

    try:
        shape_before = before.get("shape", (0, 0))
        shape_after = after.get("shape", (0, 0))
        missing_before = before.get("missing_total", 0)
        missing_after = after.get("missing_total", 0)
        cols_before = before.get("columns", [])
        cols_after = after.get("columns", [])

        rows_change = shape_after[0] - shape_before[0]
        cols_change = shape_after[1] - shape_before[1]
        missing_change = missing_after - missing_before

        rows_pct_change = (rows_change / shape_before[0] * 100) if shape_before[0] > 0 else 0.0
        missing_pct_change = (missing_change / missing_before * 100) if missing_before > 0 else 0.0

        set_before = set(cols_before)
        set_after = set(cols_after)
        added_columns = list(set_after - set_before)
        removed_columns = list(set_before - set_after)

        return {
            "shape_before": shape_before,
            "shape_after": shape_after,
            "rows_change": rows_change,
            "rows_pct_change": round(rows_pct_change, 2),
            "columns_change": cols_change,
            "missing_total_before": int(missing_before),
            "missing_total_after": int(missing_after),
            "missing_change": int(missing_change),
            "missing_pct_change": round(missing_pct_change, 2),
            "n_columns_before": len(cols_before),
            "n_columns_after": len(cols_after),
            "added_columns": added_columns,
            "removed_columns": removed_columns,
        }
    except Exception as e:
        logger.error(f"Error in compare_stats: {e}")
        st.error(f"Error comparing statistics: {e}")
        return {
            "shape_before": (0, 0),
            "shape_after": (0, 0),
            "rows_change": 0,
            "rows_pct_change": 0.0,
            "columns_change": 0,
            "missing_total_before": 0,
            "missing_total_after": 0,
            "missing_change": 0,
            "missing_pct_change": 0.0,
            "n_columns_before": 0,
            "n_columns_after": 0,
            "added_columns": [],
            "removed_columns": [],
            "error": str(e)
        }
