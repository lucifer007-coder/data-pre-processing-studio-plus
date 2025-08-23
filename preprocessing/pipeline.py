import logging
import time
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
    smooth_time_series,
    resample_time_series,
    clean_text,
    extract_tfidf,
    resize_image,
    normalize_image,
)

logger = logging.getLogger(__name__)

# Step registry for extensibility
STEP_REGISTRY = {
    "impute": impute_missing,
    "drop_missing": drop_missing,
    "normalize_text": normalize_text,
    "standardize_dates": standardize_dates,
    "unit_convert": unit_convert,
    "outliers": handle_outliers,
    "duplicates": remove_duplicates,
    "encode": encode_categorical,
    "scale": scale_features,
    "rebalance": rebalance_dataset,
    "type_convert": type_convert,
    "skewness_transform": skewness_transform,
    "mask_pii": mask_pii,
    "smooth_time_series": smooth_time_series,
    "resample_time_series": resample_time_series,
    "clean_text": clean_text,
    "extract_tfidf": extract_tfidf,
    "resize_image": resize_image,
    "normalize_image": normalize_image,
}

def validate_pipeline(df: pd.DataFrame, pipeline: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """
    Validate the pipeline by checking step kinds and parameters.
    Returns (is_valid, errors).
    """
    errors = []
    required_params = {
        "impute": ["columns", "strategy"],
        "drop_missing": ["axis"],
        "normalize_text": ["columns"],
        "standardize_dates": ["columns"],
        "unit_convert": ["column", "factor"],
        "outliers": ["columns", "method"],
        "duplicates": [],
        "encode": ["columns", "method"],
        "scale": ["columns", "method"],
        "rebalance": ["target", "method"],
        "type_convert": ["column", "type"],
        "skewness_transform": ["column", "transform"],
        "mask_pii": ["column"],
        "smooth_time_series": ["column", "window", "method"],
        "resample_time_series": ["time_column", "freq", "agg_func"],
        "clean_text": ["column", "remove_stopwords"],
        "extract_tfidf": ["column", "max_features"],
        "resize_image": ["column", "width", "height"],
        "normalize_image": ["column"],
    }

    for idx, step in enumerate(pipeline, start=1):
        kind = step.get("kind", "unknown")
        params = step.get("params", {})
        if kind not in STEP_REGISTRY:
            errors.append(f"Step {idx}: Unknown step kind '{kind}'")
            continue
        if not isinstance(params, dict):
            errors.append(f"Step {idx}: 'params' must be a dictionary")
            continue
        for param in required_params.get(kind, []):
            if param not in params:
                errors.append(f"Step {idx}: Missing required parameter '{param}' for step '{kind}'")
        if kind in ["impute", "normalize_text", "standardize_dates", "outliers", "encode", "scale"]:
            columns = params.get("columns", [])
            invalid_cols = [c for c in columns if c not in df.columns]
            if invalid_cols:
                errors.append(f"Step {idx}: Invalid columns {invalid_cols} in step '{kind}'")
        elif kind in ["unit_convert", "type_convert", "skewness_transform", "mask_pii", "smooth_time_series", "clean_text", "extract_tfidf", "resize_image", "normalize_image"]:
            column = params.get("column")
            if column and column not in df.columns:
                errors.append(f"Step {idx}: Column '{column}' not found in step '{kind}'")
        elif kind == "resample_time_series":
            time_column = params.get("time_column")
            if time_column and time_column not in df.columns:
                errors.append(f"Step {idx}: Time column '{time_column}' not found in step '{kind}'")
        elif kind == "rebalance":
            target = params.get("target")
            if target and target not in df.columns:
                errors.append(f"Step {idx}: Target column '{target}' not found in step '{kind}'")
    return len(errors) == 0, errors

def apply_step(df: pd.DataFrame, step: Dict[str, Any], preview: bool = False) -> Tuple[pd.DataFrame, str]:
    """
    Apply a single preprocessing step to the DataFrame.
    Returns (transformed_df, message).
    """
    start_time = time.time()
    try:
        kind = step.get("kind")
        if not kind:
            return df, "Error: Step 'kind' not specified."
        params = step.get("params", {})
        if not isinstance(params, dict):
            return df, "Error: Step 'params' must be a dictionary."
        if kind not in STEP_REGISTRY:
            return df, f"Unknown step kind: {kind}"
        params_with_preview = {**params, "preview": preview}
        df, msg = STEP_REGISTRY[kind](df, **params_with_preview)
        logger.info(f"Step {kind} took {time.time() - start_time:.2f} seconds")
        return df, msg
    except KeyError as e:
        logger.error(f"Missing key in step {kind}: {e}")
        return df, f"Error: Missing key {e} in step {kind}"
    except ValueError as e:
        logger.error(f"Invalid value in step {kind}: {e}")
        return df, f"Error: Invalid value in step {kind}: {e}"
    except Exception as e:
        logger.error(f"Unexpected error in step {kind}: {e}")
        return df, f"Error in step {kind}: {e}"

def run_pipeline(df: pd.DataFrame, pipeline: List[Dict[str, Any]], preview: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply a sequence of preprocessing steps to the DataFrame.
    Returns (transformed_df, messages).
    """
    try:
        is_valid, errors = validate_pipeline(df, pipeline)
        if not is_valid:
            return df, [f"Pipeline validation failed: {err}" for err in errors]
        df_out = df.copy()
        messages = []
        for idx, step in enumerate(pipeline, start=1):
            df_out, msg = apply_step(df_out, step, preview=preview)
            messages.append(f"{idx}. {msg}")
        return df_out, messages
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
        return df, [f"Pipeline error: {e}"]
