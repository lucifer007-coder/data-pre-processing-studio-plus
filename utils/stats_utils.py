import logging
import pandas as pd
import numpy as np
import dask.dataframe as dd
from typing import Dict, Any, Optional
import streamlit as st
from utils.data_utils import dtype_split

logger = logging.getLogger(__name__)

@st.cache_data(
    show_spinner="Computing stats …",
    ttl=600,
    max_entries=3
)
def compute_basic_stats(df: Optional[pd.DataFrame | dd.DataFrame]) -> Dict[str, Any]:
    """
    Return comprehensive statistics for a DataFrame.
    Cached per DataFrame identity + shape + dtypes to avoid recomputation.
    """
    if df is None or (isinstance(df, pd.DataFrame) and df.empty) or (isinstance(df, dd.DataFrame) and df.compute().empty):
        logger.warning("DataFrame is None or empty, returning empty statistics")
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

    if not isinstance(df, (pd.DataFrame, dd.DataFrame)):
        logger.error(f"Expected pandas or dask DataFrame or None, got {type(df)}")
        st.error(f"Invalid data type: expected DataFrame, got {type(df)}")
        return {}

    try:
        if isinstance(df, dd.DataFrame):
            shape = df.shape.compute()
            columns = df.columns.compute().tolist()
            dtypes = df.dtypes.compute().to_dict()
            missing_series = df.isna().sum().compute()
            missing_total = int(missing_series.sum())
            missing_by_col = {k: int(v) for k, v in missing_series.sort_values(ascending=False).items()}
            num_cols, cat_cols = dtype_split(df)
            describe_numeric = {}
            if num_cols:
                numeric_df = df[num_cols]
                if shape[0] > 1_000_000:
                    numeric_df = numeric_df.sample(frac=100_000/shape[0], random_state=42)
                desc = numeric_df.describe().compute()
                describe_numeric = desc.to_dict()
            memory_usage_mb = df.memory_usage(deep=True).sum().compute() / 1_000_000
            duplicate_rows = df.duplicated().sum().compute()
        else:
            shape = df.shape
            columns = df.columns.tolist()
            dtypes = df.dtypes.to_dict()
            missing_series = df.isna().sum()
            missing_total = int(missing_series.sum())
            missing_by_col = {k: int(v) for k, v in missing_series.sort_values(ascending=False).items()}
            num_cols, cat_cols = dtype_split(df)
            describe_numeric = {}
            if num_cols:
                numeric_df = df[num_cols]
                if len(numeric_df) > 1_000_000:
                    numeric_df = numeric_df.sample(n=100_000, random_state=42)
                desc = numeric_df.describe()
                describe_numeric = desc.to_dict()
            memory_usage_mb = df.memory_usage(deep=True).sum() / 1_000_000
            duplicate_rows = df.duplicated().sum()

        return {
            "shape": shape,
            "columns": columns,
            "dtypes": dtypes,
            "missing_total": missing_total,
            "missing_by_col": missing_by_col,
            "numeric_cols": num_cols,
            "categorical_cols": cat_cols,
            "describe_numeric": describe_numeric,
            "memory_usage_mb": float(memory_usage_mb),
            "duplicate_rows": int(duplicate_rows)
        }

    except Exception as e:
        logger.error(f"Error in compute_basic_stats: {e}")
        st.error(f"Error computing statistics: {e}")
        return {}

def compare_stats(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """Compare statistics between two DataFrames."""
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
            "error": str(e),
        }
