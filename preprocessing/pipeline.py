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
    extract_domain,
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
    "extract_domain": extract_domain,
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
        "skewness_transform": ["columns", "method"],
        "mask_pii": ["columns"],
        "smooth_time_series": ["column", "method"],
        "resample_time_series": ["time_column", "freq"],
        "clean_text": ["column"],
        "extract_tfidf": ["column"],
        "resize_image": ["column", "width", "height"],
        "normalize_image": ["column"],
        "extract_domain": ["column"],
    }
    for step in pipeline:
        kind = step.get("kind")
        if not kind:
            errors.append("Step 'kind' not specified.")
            continue
        if kind not in STEP_REGISTRY:
            errors.append(f"Unknown step kind: {kind}")
            continue
        params = step.get("params", {})
        if not isinstance(params, dict):
            errors.append(f"Step {kind}: 'params' must be a dictionary.")
            continue
        missing_params = [p for p in required_params.get(kind, []) if p not in params]
        if missing_params:
            errors.append(f"Step {kind}: Missing required parameters: {', '.join(missing_params)}")
        if "column" in params and params["column"] not in df.columns:
            errors.append(f"Step {kind}: Column {params['column']} not found in DataFrame.")
        if "columns" in params and not all(c in df.columns for c in params["columns"]):
            errors.append(f"Step {kind}: Some columns not found in DataFrame.")
        if kind == "encode" and params.get("method") == "target" and "target_column" not in params:
            errors.append(f"Step {kind}: Target column required for target encoding.")
        if kind == "encode" and params.get("method") == "ordinal" and "ordinal_mappings" not in params:
            errors.append(f"Step {kind}: Ordinal mappings required for ordinal encoding.")
        if kind == "impute" and params.get("strategy") == "knn" and "n_neighbors" not in params:
            errors.append(f"Step {kind}: n_neighbors required for KNN imputation.")
        if kind == "impute" and params.get("strategy") == "random_forest" and "n_estimators" not in params:
            errors.append(f"Step {kind}: n_estimators required for Random Forest imputation.")
        if kind == "impute" and params.get("strategy") == "constant" and "constant_value" not in params:
            errors.append(f"Step {kind}: constant_value required for constant imputation.")
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
