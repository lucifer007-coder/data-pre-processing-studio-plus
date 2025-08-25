import logging
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List, Tuple, Dict, Any
import pandas as pd
import dask.dataframe as dd
import streamlit as st
from config import MAX_WORKERS
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
    extract_domain,
)

# Check for dask.distributed availability
try:
    from dask.distributed import Client, get_client
    DASK_DISTRIBUTED_AVAILABLE = True
except ImportError:
    DASK_DISTRIBUTED_AVAILABLE = False
    Client = None
    get_client = None

logger = logging.getLogger(__name__)

# Step registry with dependency metadata for extensibility
STEP_REGISTRY = {
    "impute": {"func": impute_missing, "depends_on": []},
    "drop_missing": {"func": drop_missing, "depends_on": []},
    "normalize_text": {"func": normalize_text, "depends_on": []},
    "standardize_dates": {"func": standardize_dates, "depends_on": []},
    "unit_convert": {"func": unit_convert, "depends_on": []},
    "outliers": {"func": handle_outliers, "depends_on": []},
    "duplicates": {"func": remove_duplicates, "depends_on": []},
    "encode": {"func": encode_categorical, "depends_on": []},
    "scale": {"func": scale_features, "depends_on": []},
    "rebalance": {"func": rebalance_dataset, "depends_on": []},
    "type_convert": {"func": type_convert, "depends_on": []},
    "skewness_transform": {"func": skewness_transform, "depends_on": []},
    "mask_pii": {"func": mask_pii, "depends_on": []},
    "smooth_time_series": {"func": smooth_time_series, "depends_on": []},
    "resample_time_series": {"func": resample_time_series, "depends_on": []},
    "clean_text": {"func": clean_text, "depends_on": []},
    "extract_tfidf": {"func": extract_tfidf, "depends_on": ["clean_text"]},
    "extract_domain": {"func": extract_domain, "depends_on": []},
}

def validate_pipeline(df: pd.DataFrame | dd.DataFrame, pipeline: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
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
        "resample_time_series": ["time_column", "freq", "agg_func"],
        "clean_text": ["column"],
        "extract_tfidf": ["column", "max_features"],
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
        for param in required_params.get(kind, []):
            if param not in params:
                errors.append(f"Step {kind}: Missing required parameter '{param}'.")
        # Validate specific parameter values
        if kind == "impute":
            if params.get("strategy") not in ["mean", "median", "mode", "constant", "ffill", "bfill", "knn", "random_forest"]:
                errors.append(f"Step {kind}: Invalid strategy '{params.get('strategy')}'.")
            if params.get("strategy") == "knn" and (not isinstance(params.get("n_neighbors"), int) or params.get("n_neighbors") <= 0):
                errors.append(f"Step {kind}: 'n_neighbors' must be a positive integer.")
        if kind == "outliers":
            if params.get("method") not in ["iqr", "zscore"]:
                errors.append(f"Step {kind}: Invalid method '{params.get('method')}'.")
        if kind == "rebalance":
            if params.get("method") not in ["oversample", "undersample"]:
                errors.append(f"Step {kind}: Invalid method '{params.get('method')}'.")
        if "columns" in params and not all(col in df.columns for col in params.get("columns", [])):
            errors.append(f"Step {kind}: One or more columns not found in DataFrame.")

    # Validate dependency order
    executed_steps = set()
    for step in pipeline:
        kind = step.get("kind")
        if kind not in STEP_REGISTRY:
            continue
        for dep in STEP_REGISTRY[kind]["depends_on"]:
            if dep not in executed_steps:
                errors.append(f"Step {kind}: Required dependency '{dep}' not executed before this step.")
        executed_steps.add(kind)

    return len(errors) == 0, errors

async def apply_step_async(df: pd.DataFrame | dd.DataFrame, step: Dict[str, Any], preview: bool = False, executor: ThreadPoolExecutor = None) -> Tuple[pd.DataFrame | dd.DataFrame, str]:
    """
    Asynchronously apply a single preprocessing step to the DataFrame.
    Returns (transformed_df, message).
    """
    start_time = time.time()
    try:
        kind = step.get("kind")
        if not kind:
            return df, "Error: Step 'kind' not specified."
        if kind not in STEP_REGISTRY:
            return df, f"Error: Unknown step kind: {kind}"
        params = step.get("params", {})
        if not isinstance(params, dict):
            return df, f"Error: Step 'params' must be a dictionary."
        
        params_with_preview = {**params, "preview": preview}
        loop = asyncio.get_running_loop()
        
        if isinstance(df, dd.DataFrame):
            # Use Dask's scheduler if available, otherwise fall back to synchronous
            if DASK_DISTRIBUTED_AVAILABLE:
                try:
                    client = get_client()  # Get existing Dask client
                    result = await loop.run_in_executor(None, lambda: client.submit(STEP_REGISTRY[kind]["func"], df, **params_with_preview).result())
                except RuntimeError as e:
                    logger.warning(f"No Dask client available, falling back to synchronous execution for {kind}: {e}")
                    st.warning(
                        f"Dask distributed processing not available for step '{kind}'. Falling back to synchronous execution. "
                        "For better performance, install dask[distributed] with: "
                        "pip install \"dask[distributed]\" --upgrade"
                    )
                    result = STEP_REGISTRY[kind]["func"](df, **params_with_preview)
            else:
                logger.warning(f"dask.distributed not installed, using synchronous execution for {kind}")
                st.warning(
                    f"Dask distributed processing not available for step '{kind}'. Falling back to synchronous execution. "
                    "For better performance, install dask[distributed] with: "
                    "pip install \"dask[distributed]\" --upgrade"
                )
                result = STEP_REGISTRY[kind]["func"](df, **params_with_preview)
        else:
            # Use ThreadPoolExecutor for Pandas DataFrames
            result = await loop.run_in_executor(executor, partial(STEP_REGISTRY[kind]["func"], df, **params_with_preview))
        
        df_out, msg = result
        logger.info(f"Step {kind} took {time.time() - start_time:.2f} seconds")
        return df_out, msg
    except KeyError as e:
        logger.error(f"Missing key in step {kind}: {e}")
        return df, f"Error: Missing key {e} in step {kind}"
    except ValueError as e:
        logger.error(f"Invalid value in step {kind}: {e}")
        return df, f"Error: Invalid value in step {kind}: {e}"
    except Exception as e:
        logger.error(f"Unexpected error in step {kind}: {e}")
        return df, f"Error in step {kind}: {e}"

async def run_pipeline_async(df: pd.DataFrame | dd.DataFrame, pipeline: List[Dict[str, Any]], preview: bool = False) -> Tuple[pd.DataFrame | dd.DataFrame, List[str]]:
    """
    Asynchronously apply a sequence of preprocessing steps to the DataFrame.
    Groups dependent steps into sequential tasks to ensure correctness.
    Returns (transformed_df, messages).
    """
    try:
        is_valid, errors = validate_pipeline(df, pipeline)
        if not is_valid:
            logger.error(f"Pipeline validation failed: {errors}")
            return df, [f"Pipeline validation failed: {err}" for err in errors]
        
        df_out = df.copy() if isinstance(df, pd.DataFrame) else df if preview else df.copy()
        messages = []
        
        # Group steps by dependencies
        task_groups = []
        current_group = []
        executed_steps = set()
        for step in pipeline:
            kind = step.get("kind")
            if not kind or kind not in STEP_REGISTRY:
                current_group.append(step)
                continue
            if any(dep in executed_steps for dep in STEP_REGISTRY[kind]["depends_on"]):
                task_groups.append(current_group)
                current_group = [step]
            else:
                current_group.append(step)
            executed_steps.add(kind)
        if current_group:
            task_groups.append(current_group)
        
        # Execute groups sequentially, parallelize within groups
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for group_idx, group in enumerate(task_groups, 1):
                if not group:
                    continue
                tasks = [apply_step_async(df_out, step, preview, executor) for step in group]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for idx, result in enumerate(results, len(messages) + 1):
                    if isinstance(result, tuple) and len(result) == 2:
                        df_out, msg = result
                        messages.append(f"{idx}. {msg}")
                        if "Error" in msg:
                            logger.warning(f"Stopping pipeline at step {idx} due to error: {msg}")
                            return df_out, messages  # Early stopping on error
                    else:
                        msg = f"Step {idx}: Error in step execution."
                        logger.error(msg)
                        messages.append(msg)
                        return df_out, messages  # Early stopping on unexpected result
        
        return df_out, messages
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
        return df, [f"Pipeline error: {e}"]

def run_pipeline(df: pd.DataFrame | dd.DataFrame, pipeline: List[Dict[str, Any]], preview: bool = False) -> Tuple[pd.DataFrame | dd.DataFrame, List[str]]:
    """
    Apply a sequence of preprocessing steps to the DataFrame.
    Wraps async execution for synchronous compatibility.
    Returns (transformed_df, messages).
    """
    return asyncio.run(run_pipeline_async(df, pipeline, preview))
