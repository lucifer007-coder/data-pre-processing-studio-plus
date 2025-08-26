import logging
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List, Tuple, Dict, Any
from types import MappingProxyType
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

def validate_step_function(func):
    """Validate that function is safe to execute"""
    if not callable(func):
        raise ValueError("Step function must be callable")
    return func

# Step registry with dependency metadata for extensibility - made immutable for security
STEP_REGISTRY = MappingProxyType({
    "impute": {"func": validate_step_function(impute_missing), "depends_on": []},
    "drop_missing": {"func": validate_step_function(drop_missing), "depends_on": []},
    "normalize_text": {"func": validate_step_function(normalize_text), "depends_on": []},
    "standardize_dates": {"func": validate_step_function(standardize_dates), "depends_on": []},
    "unit_convert": {"func": validate_step_function(unit_convert), "depends_on": []},
    "outliers": {"func": validate_step_function(handle_outliers), "depends_on": []},
    "duplicates": {"func": validate_step_function(remove_duplicates), "depends_on": []},
    "encode": {"func": validate_step_function(encode_categorical), "depends_on": []},
    "scale": {"func": validate_step_function(scale_features), "depends_on": []},
    "rebalance": {"func": validate_step_function(rebalance_dataset), "depends_on": []},
    "type_convert": {"func": validate_step_function(type_convert), "depends_on": []},
    "skewness_transform": {"func": validate_step_function(skewness_transform), "depends_on": []},
    "mask_pii": {"func": validate_step_function(mask_pii), "depends_on": []},
    "smooth_time_series": {"func": validate_step_function(smooth_time_series), "depends_on": []},
    "resample_time_series": {"func": validate_step_function(resample_time_series), "depends_on": []},
    "clean_text": {"func": validate_step_function(clean_text), "depends_on": []},
    "extract_tfidf": {"func": validate_step_function(extract_tfidf), "depends_on": ["clean_text"]},
    "extract_domain": {"func": validate_step_function(extract_domain), "depends_on": []},
})

def validate_pipeline(df: pd.DataFrame | dd.DataFrame, pipeline: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """
    Validate the pipeline by checking step kinds and parameters.
    Returns (is_valid, errors).
    """
    errors = []
    
    # Validate inputs
    if not isinstance(pipeline, list):
        return False, ["Pipeline must be a list"]
    
    if not isinstance(df, (pd.DataFrame, dd.DataFrame)):
        return False, ["df must be a pandas or dask DataFrame"]
    
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

    for i, step in enumerate(pipeline):
        if not isinstance(step, dict):
            errors.append(f"Step {i}: must be a dictionary")
            continue
            
        kind = step.get("kind")
        if not kind:
            errors.append(f"Step {i}: 'kind' not specified.")
            continue
        if kind not in STEP_REGISTRY:
            errors.append(f"Step {i}: Unknown step kind: {kind}")
            continue
        params = step.get("params", {})
        if not isinstance(params, dict):
            errors.append(f"Step {i} ({kind}): 'params' must be a dictionary.")
            continue
        for param in required_params.get(kind, []):
            if param not in params:
                errors.append(f"Step {i} ({kind}): Missing required parameter '{param}'.")
        
        # Validate specific parameter values
        if kind == "impute":
            if params.get("strategy") not in ["mean", "median", "mode", "constant", "ffill", "bfill", "knn", "random_forest"]:
                errors.append(f"Step {i} ({kind}): Invalid strategy '{params.get('strategy')}'.")
            if params.get("strategy") == "knn" and (not isinstance(params.get("n_neighbors"), int) or params.get("n_neighbors") <= 0):
                errors.append(f"Step {i} ({kind}): 'n_neighbors' must be a positive integer.")
        if kind == "outliers":
            if params.get("method") not in ["iqr", "zscore"]:
                errors.append(f"Step {i} ({kind}): Invalid method '{params.get('method')}'.")
        if kind == "rebalance":
            if params.get("method") not in ["oversample", "undersample"]:
                errors.append(f"Step {i} ({kind}): Invalid method '{params.get('method')}'.")
        
        # Efficient column validation
        if "columns" in params:
            try:
                missing_cols = [col for col in params.get("columns", []) if col not in df.columns]
                if missing_cols:
                    errors.append(f"Step {i} ({kind}): Columns not found: {missing_cols}")
            except Exception:
                # Skip column validation for complex Dask operations
                logger.warning(f"Could not validate columns for step {i} ({kind})")

    # Validate dependency order
    executed_steps = set()
    for i, step in enumerate(pipeline):
        kind = step.get("kind")
        if kind not in STEP_REGISTRY:
            continue
        for dep in STEP_REGISTRY[kind]["depends_on"]:
            if dep not in executed_steps:
                errors.append(f"Step {i} ({kind}): Required dependency '{dep}' not executed before this step.")
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
                    client = get_client()
                    future = client.submit(STEP_REGISTRY[kind]["func"], df, **params_with_preview)
                    result = await loop.run_in_executor(None, future.result)
                except (RuntimeError, OSError):
                    # No client available, use compute()
                    logger.warning(f"No Dask client available, falling back to synchronous execution for {kind}")
                    st.warning(
                        f"Dask distributed processing not available for step '{kind}'. Falling back to synchronous execution. "
                        "For better performance, install dask[distributed] with: "
                        "pip install \"dask[distributed]\" --upgrade"
                    )
                    result = df.map_partitions(STEP_REGISTRY[kind]["func"], **params_with_preview).compute()
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
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        logger.error(f"Processing error in step {kind}: {e}", exc_info=True)
        return df, f"Error in step {kind}: {str(e)}"
    except Exception as e:
        logger.critical(f"Unexpected error in step {kind}: {e}", exc_info=True)
        raise  # Re-raise unexpected exceptions

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
        
        # Safe DataFrame copy operations
        if isinstance(df, pd.DataFrame):
            df_out = df.copy() if not preview else df
        else:  # Dask DataFrame
            df_out = df  # Dask is lazy, no need to copy
        
        messages = []
        
        # Group steps by dependencies - FIXED LOGIC
        task_groups = []
        current_group = []
        executed_steps = set()
        
        for step in pipeline:
            kind = step.get("kind")
            if not kind or kind not in STEP_REGISTRY:
                current_group.append(step)
                continue
            
            # Check if ALL dependencies are satisfied
            deps_satisfied = all(dep in executed_steps for dep in STEP_REGISTRY[kind]["depends_on"])
            if not deps_satisfied and current_group:
                # Start new group if dependencies aren't met
                task_groups.append(current_group)
                current_group = [step]
            else:
                current_group.append(step)
            executed_steps.add(kind)
        
        if current_group:
            task_groups.append(current_group)
        
        # Execute groups sequentially to maintain data integrity
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for group_idx, group in enumerate(task_groups, 1):
                if not group:
                    continue
                
                # Execute steps in group sequentially to maintain data integrity
                for step in group:
                    df_out, msg = await apply_step_async(df_out, step, preview, executor)
                    messages.append(f"{len(messages) + 1}. {msg}")
                    if "Error" in msg:
                        logger.warning(f"Stopping pipeline at step {len(messages)} due to error: {msg}")
                        return df_out, messages  # Early stopping on error
        
        return df_out, messages
    except Exception as e:
        logger.error(f"Error running pipeline: {e}", exc_info=True)
        return df, [f"Pipeline error: {e}"]

def run_pipeline(df: pd.DataFrame | dd.DataFrame, pipeline: List[Dict[str, Any]], preview: bool = False) -> Tuple[pd.DataFrame | dd.DataFrame, List[str]]:
    """
    Apply a sequence of preprocessing steps to the DataFrame.
    Wraps async execution for synchronous compatibility.
    Returns (transformed_df, messages).
    """
    try:
        loop = asyncio.get_running_loop()
        # We're in an existing loop, create a task
        return loop.run_until_complete(run_pipeline_async(df, pipeline, preview))
    except RuntimeError:
        # No loop running, safe to create one
        return asyncio.run(run_pipeline_async(df, pipeline, preview))
