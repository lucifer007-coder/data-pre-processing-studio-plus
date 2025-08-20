import logging
from typing import List, Tuple, Dict, Any
import pandas as pd
from preprocessing.steps import (
    impute_missing,
    drop_missing,
    normalize_text,
    standardize_dates,
    unit_convert,
    handle_outliers,
    remove_duplicates,
    encode_categorical,
    scale_features,
    rebalance_dataset,
    type_convert,
    skewness_transform,
    mask_pii,
)

logger = logging.getLogger(__name__)

def apply_step(df: pd.DataFrame, step: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
    try:
        kind = step.get("kind")
        params = step.get("params", {})
        if kind == "impute":
            return impute_missing(df, **params)
        elif kind == "drop_missing":
            return drop_missing(df, **params)
        elif kind == "normalize_text":
            return normalize_text(df, **params)
        elif kind == "standardize_dates":
            return standardize_dates(df, **params)
        elif kind == "unit_convert":
            return unit_convert(df, **params)
        elif kind == "outliers":
            return handle_outliers(df, **params)
        elif kind == "duplicates":
            return remove_duplicates(df, **params)
        elif kind == "encode":
            return encode_categorical(df, **params)
        elif kind == "scale":
            return scale_features(df, **params)
        elif kind == "rebalance":
            return rebalance_dataset(df, **params)
        elif kind == "type_convert":
            return type_convert(df, **params)
        elif kind == "skewness_transform":
            return skewness_transform(df, **params)
        elif kind == "mask_pii":
            return mask_pii(df, **params)
        else:
            return df, f"Unknown step kind: {kind}"
    except Exception as e:
        logger.error(f"Error applying step {step.get('kind', 'unknown')}: {e}")
        return df, f"Error in step {step.get('kind', 'unknown')}: {e}"

def run_pipeline(df: pd.DataFrame, pipeline: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, List[str]]:
    try:
        df = df.copy()
        messages = []
        for idx, step in enumerate(pipeline, start=1):
            df, msg = apply_step(df, step)
            messages.append(f"{idx}. {msg}")
        return df, messages
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
        return df, [f"Pipeline error: {e}"]
