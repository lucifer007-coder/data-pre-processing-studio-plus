import logging
import time
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Any, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
import regex as re
from utils.data_utils import dtype_split

logger = logging.getLogger(__name__)

def impute_missing(
    df: pd.DataFrame,
    columns: List[str],
    strategy: str = "mean",
    constant_value: Optional[Any] = None,
    preview: bool = False,
) -> Tuple[pd.DataFrame, str]:
    """
    Impute missing values in specified columns using the given strategy.
    Returns (transformed_df, message).
    """
    start_time = time.time()
    try:
        if not columns:
            columns = df.columns.tolist()
        valid_cols = [c for c in columns if c in df.columns]
        if not valid_cols:
            return df, "No valid columns selected for imputation."
        df_out = df if preview else df.copy()
        num_cols, cat_cols = dtype_split(df_out)
        imputed_cols = []
        if strategy in ("mean", "median"):
            target_cols = [c for c in valid_cols if c in num_cols]
            if target_cols:
                values = df_out[target_cols].agg(strategy).to_dict()
                for c in target_cols:
                    if df_out[c].isna().all():
                        continue
                    imputed_count = df_out[c].isna().sum()
                    df_out[c] = df_out[c].fillna(values[c])
                    imputed_cols.append(f"{c} ({imputed_count} values with {strategy}={values[c]:.2f})")
                msg = f"Imputed {len(target_cols)} numeric columns: {', '.join(imputed_cols)}"
            else:
                msg = "No numeric columns selected for mean/median imputation."
        elif strategy == "mode":
            for c in valid_cols:
                mode = df_out[c].mode(dropna=True)
                if not mode.empty:
                    imputed_count = df_out[c].isna().sum()
                    df_out[c] = df_out[c].fillna(mode.iloc[0])
                    imputed_cols.append(f"{c} ({imputed_count} values with mode={mode.iloc[0]})")
            msg = f"Imputed {len(valid_cols)} columns: {', '.join(imputed_cols)}"
        elif strategy == "constant":
            val = constant_value if constant_value is not None else 0
            df_out[valid_cols] = df_out[valid_cols].fillna(val)
            msg = f"Imputed {len(valid_cols)} columns with constant value {val}."
        elif strategy == "ffill":
            df_out[valid_cols] = df_out[valid_cols].fillna(method="ffill")
            msg = f"Forward-filled {len(valid_cols)} columns."
        elif strategy == "bfill":
            df_out[valid_cols] = df_out[valid_cols].fillna(method="bfill")
            msg = f"Back-filled {len(valid_cols)} columns."
        else:
            msg = f"Unknown strategy '{strategy}'."
        logger.info(f"impute_missing took {time.time() - start_time:.2f} seconds")
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in impute_missing: {e}")
        return df, f"Error in imputation: {e}"

def drop_missing(
    df: pd.DataFrame,
    axis: str = "rows",
    threshold: Optional[float] = None,
    columns: Optional[List[str]] = None,
    preview: bool = False,
) -> Tuple[pd.DataFrame, str]:
    """
    Drop rows or columns with missing values based on axis and threshold.
    Returns (transformed_df, message).
    """
    start_time = time.time()
    try:
        if axis not in ["rows", "columns"]:
            return df, f"Invalid axis '{axis}'; must be 'rows' or 'columns'."
        valid_cols = columns if columns else df.columns.tolist()
        valid_cols = [c for c in valid_cols if c in df.columns]
        if not valid_cols:
            return df, "No valid columns selected for dropping missing values."
        df_out = df if preview else df.copy()
        if axis == "rows":
            initial = len(df_out)
            if columns:
                to_drop = df_out[valid_cols].isna().any(axis=1)
                drop_count = to_drop.sum()
                df_out = df_out[~to_drop]
                msg = f"Dropped {drop_count} rows with missing values in {valid_cols}."
            elif threshold is not None:
                if not 0 <= threshold <= 1:
                    return df, f"Threshold must be between 0 and 1, got {threshold}."
                row_missing_ratio = df_out[valid_cols].isna().mean(axis=1)
                to_drop = row_missing_ratio >= threshold
                drop_count = to_drop.sum()
                df_out = df_out[~to_drop]
                msg = f"Dropped {drop_count} rows with missing ratio ≥ {threshold:.2f}."
            else:
                to_drop = df_out[valid_cols].isna().any(axis=1)
                drop_count = to_drop.sum()
                df_out = df_out[~to_drop]
                msg = f"Dropped {drop_count} rows with any missing values."
        else:
            initial = len(df_out.columns)
            if threshold is not None:
                if not 0 <= threshold <= 1:
                    return df, f"Threshold must be between 0 and 1, got {threshold}."
                col_missing_ratio = df_out[valid_cols].isna().mean()
                to_drop = col_missing_ratio[col_missing_ratio >= threshold].index.tolist()
                df_out = df_out.drop(columns=to_drop)
                msg = f"Dropped {len(to_drop)} columns with missing ratio ≥ {threshold:.2f}: {to_drop}"
            else:
                to_drop = [c for c in valid_cols if df_out[c].isna().any()]
                df_out = df_out.drop(columns=to_drop)
                msg = f"Dropped {len(to_drop)} columns with any missing values: {to_drop}"
        logger.info(f"drop_missing took {time.time() - start_time:.2f} seconds")
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in drop_missing: {e}")
        return df, f"Error dropping missing values: {e}"

def normalize_text(
    df: pd.DataFrame,
    columns: List[str],
    lowercase: bool = True,
    trim: bool = True,
    collapse_spaces: bool = True,
    remove_special: bool = False,
    preview: bool = False,
) -> Tuple[pd.DataFrame, str]:
    """
    Normalize text in specified columns (lowercase, trim, collapse spaces, remove special chars).
    Returns (transformed_df, message).
    """
    start_time = time.time()
    try:
        valid_cols = [c for c in columns if c in df.columns]
        if not valid_cols:
            return df, "No valid columns selected for text normalization."
        df_out = df if preview else df.copy()
        operations = []
        for c in valid_cols:
            df_out[c] = df_out[c].astype(str)
            if trim:
                df_out[c] = df_out[c].str.strip()
                operations.append("trimmed")
            if lowercase:
                df_out[c] = df_out[c].str.lower()
                operations.append("lowercased")
            if collapse_spaces:
                df_out[c] = df_out[c].str.replace(r"\s+", " ", regex=True)
                operations.append("collapsed spaces")
            if remove_special:
                df_out[c] = df_out[c].str.replace(r"[^\w\s]", "", regex=True)
                operations.append("removed special characters")
        msg = f"Normalized text for columns {valid_cols}: {', '.join(set(operations))}."
        logger.info(f"normalize_text took {time.time() - start_time:.2f} seconds")
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in normalize_text: {e}")
        return df, f"Error normalizing text: {e}"

def standardize_dates(
    df: pd.DataFrame,
    columns: List[str],
    output_format: str = "%Y-%m-%d",
    preview: bool = False,
) -> Tuple[pd.DataFrame, str]:
    """
    Standardize date formats in specified columns.
    Returns (transformed_df, message).
    """
    start_time = time.time()
    try:
        valid_cols = [c for c in columns if c in df.columns]
        if not valid_cols:
            return df, "No valid columns selected for date standardization."
        df_out = df if preview else df.copy()
        for c in valid_cols:
            dt = pd.to_datetime(df_out[c], errors="coerce", infer_datetime_format=True)
            df_out[c] = dt.dt.strftime(output_format)
        msg = f"Standardized date format to {output_format} for columns: {valid_cols}"
        logger.info(f"standardize_dates took {time.time() - start_time:.2f} seconds")
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in standardize_dates: {e}")
        return df, f"Error standardizing dates: {e}"

def unit_convert(
    df: pd.DataFrame,
    column: str,
    factor: float,
    new_name: Optional[str] = None,
    preview: bool = False,
) -> Tuple[pd.DataFrame, str]:
    """
    Convert units in a numeric column by multiplying by a factor.
    Returns (transformed_df, message).
    """
    start_time = time.time()
    try:
        if not isinstance(factor, (int, float)):
            return df, f"Error: 'factor' must be a number, got {type(factor)}."
        if factor == 0:
            return df, "Error: 'factor' cannot be zero."
        if column not in df.columns:
            return df, f"Column {column} not found for unit conversion."
        if not pd.api.types.is_numeric_dtype(df[column]):
            return df, f"Column {column} is not numeric; skipped unit conversion."
        df_out = df if preview else df.copy()
        target = new_name if new_name else column
        df_out[target] = df_out[column] * factor
        if new_name:
            msg = f"Created '{new_name}' by converting '{column}' with factor {factor}."
        else:
            msg = f"Converted '{column}' in place with factor {factor}."
        logger.info(f"unit_convert took {time.time() - start_time:.2f} seconds")
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in unit_convert: {e}")
        return df, f"Error in unit conversion: {e}"

def handle_outliers(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "iqr",
    factor: float = 1.5,
    preview: bool = False,
) -> Tuple[pd.DataFrame, str]:
    """
    Handle outliers in numeric columns using IQR or z-score.
    Returns (transformed_df, message).
    """
    start_time = time.time()
    try:
        if not isinstance(factor, (int, float)) or factor <= 0:
            return df, f"Error: 'factor' must be a positive number, got {factor}."
        valid_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if not valid_cols:
            return df, "No valid numeric columns selected for outlier handling."
        df_out = df if preview else df.copy()
        if method == "iqr":
            for c in valid_cols:
                q1 = df_out[c].quantile(0.25)
                q3 = df_out[c].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - factor * iqr
                upper = q3 + factor * iqr
                outliers = (df_out[c] < lower) | (df_out[c] > upper)
                df_out.loc[outliers, c] = np.nan
            msg = f"Handled outliers in {len(valid_cols)} columns using IQR (factor={factor})."
        elif method == "zscore":
            for c in valid_cols:
                z_scores = np.abs((df_out[c] - df_out[c].mean()) / df_out[c].std())
                outliers = z_scores > factor
                df_out.loc[outliers, c] = np.nan
            msg = f"Handled outliers in {len(valid_cols)} columns using z-score (threshold={factor})."
        else:
            msg = f"Unknown outlier method: {method}."
        logger.info(f"handle_outliers took {time.time() - start_time:.2f} seconds")
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in handle_outliers: {e}")
        return df, f"Error handling outliers: {e}"

def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = "first",
    preview: bool = False,
) -> Tuple[pd.DataFrame, str]:
    """
    Remove duplicate rows based on all or specified columns.
    Returns (transformed_df, message).
    """
    start_time = time.time()
    try:
        if keep not in ["first", "last", False]:
            return df, f"Invalid 'keep' value: {keep}; must be 'first', 'last', or False."
        valid_subset = [c for c in (subset or df.columns) if c in df.columns]
        if not valid_subset:
            return df, "No valid columns selected for duplicate removal."
        df_out = df if preview else df.copy()
        initial = len(df_out)
        df_out = df_out.drop_duplicates(subset=valid_subset, keep=keep)
        removed = initial - len(df_out)
        msg = f"Removed {removed} duplicate rows based on columns: {valid_subset if subset else 'all'}."
        logger.info(f"remove_duplicates took {time.time() - start_time:.2f} seconds")
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in remove_duplicates: {e}")
        return df, f"Error removing duplicates: {e}"

def encode_categorical(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "onehot",
    max_categories: Optional[int] = None,
    ordinal_mappings: Optional[Dict[str, Dict[Any, int]]] = None,
    preview: bool = False,
) -> Tuple[pd.DataFrame, str]:
    """
    Encode categorical columns using one-hot, label, or ordinal encoding.
    Returns (transformed_df, message).
    """
    start_time = time.time()
    try:
        valid_cols = [c for c in columns if c in df.columns]
        if not valid_cols:
            return df, "No valid columns selected for encoding."
        df_out = df if preview else df.copy()
        if method == "onehot":
            for c in valid_cols:
                if max_categories and df_out[c].nunique() > max_categories:
                    top_cats = df_out[c].value_counts().index[:max_categories]
                    df_out[c] = df_out[c].where(df_out[c].isin(top_cats), "Other")
            df_encoded = pd.get_dummies(df_out, columns=valid_cols, prefix=valid_cols)
            new_cols = [col for col in df_encoded.columns if col not in df_out.columns]
            msg = f"One-hot encoded {len(valid_cols)} columns, added {len(new_cols)} new columns."
            logger.info(f"encode_categorical took {time.time() - start_time:.2f} seconds")
            return df_encoded, msg
        elif method == "label":
            le_info = []
            for c in valid_cols:
                if max_categories and df_out[c].nunique() > max_categories:
                    top_cats = df_out[c].value_counts().index[:max_categories]
                    df_out[c] = df_out[c].where(df_out[c].isin(top_cats), "Other")
                le = LabelEncoder()
                df_out[c] = le.fit_transform(df_out[c].astype(str))
                le_info.append(f"{c} ({len(le.classes_)} classes)")
            msg = f"Label encoded columns: {', '.join(le_info)}. Note: Ensure label encoding suits your model."
            logger.info(f"encode_categorical took {time.time() - start_time:.2f} seconds")
            return df_out, msg
        elif method == "ordinal":
            if not ordinal_mappings:
                return df, "Error: Ordinal encoding requires 'ordinal_mappings' dictionary."
            encoded_cols = []
            for c in valid_cols:
                if c in ordinal_mappings:
                    df_out[c] = df_out[c].map(ordinal_mappings[c]).fillna(-1).astype(int)
                    encoded_cols.append(c)
            if not encoded_cols:
                return df_out, "No columns matched provided ordinal mappings."
            msg = f"Ordinal encoded columns: {encoded_cols}"
            logger.info(f"encode_categorical took {time.time() - start_time:.2f} seconds")
            return df_out, msg
        else:
            return df, f"Unknown encoding method: {method}"
    except Exception as e:
        logger.error(f"Error in encode_categorical: {e}")
        return df, f"Error encoding categorical data: {e}"

def scale_features(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "standard",
    keep_original: bool = False,
    preview: bool = False,
) -> Tuple[pd.DataFrame, str]:
    """
    Scale numeric features using StandardScaler, MinMaxScaler, or RobustScaler.
    Returns (transformed_df, message).
    """
    start_time = time.time()
    try:
        valid_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if not valid_cols:
            return df, "No valid numeric columns selected for scaling."
        if df[valid_cols].isna().any().any():
            return df, "Columns contain missing values; impute first."
        df_out = df if preview else df.copy()
        if keep_original:
            new_cols = [f"{c}_scaled" for c in valid_cols]
        else:
            new_cols = valid_cols
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        elif method == "robust":
            scaler = RobustScaler()
        else:
            return df, f"Unknown scaling method: {method}"
        df_out[new_cols] = scaler.fit_transform(df_out[valid_cols].values)
        msg = f"Applied {method} scaling to columns: {valid_cols}"
        logger.info(f"scale_features took {time.time() - start_time:.2f} seconds")
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in scale_features: {e}")
        return df, f"Error scaling features: {e}"

def rebalance_dataset(
    df: pd.DataFrame,
    target: str,
    method: str = "oversample",
    ratio: float = 1.0,
    random_state: int = 42,
    preview: bool = False,
) -> Tuple[pd.DataFrame, str]:
    """
    Rebalance dataset using random oversampling or undersampling.
    Returns (transformed_df, message).
    """
    start_time = time.time()
    try:
        if not isinstance(ratio, (int, float)) or ratio <= 0:
            return df, f"Error: 'ratio' must be a positive number, got {ratio}."
        if target not in df.columns:
            return df, f"Target column '{target}' not found."
        df_out = df if preview else df.copy()
        counts = df_out[target].value_counts(dropna=False)
        if counts.empty or len(counts) <= 1:
            return df, "Target column has only one class or is empty; skipping rebalancing."
        if pd.api.types.is_numeric_dtype(df_out[target]) and method != "binning":
            return df, "Numeric target detected; use binning for regression tasks."
        if method == "oversample":
            majority_count = counts.max()
            desired = int(round(majority_count * ratio))
            dfs = []
            for cls, cnt in counts.items():
                subset = df_out[df_out[target] == cls]
                if cnt < desired:
                    add = subset.sample(n=desired - cnt, replace=True, random_state=random_state)
                    dfs.append(pd.concat([subset, add]))
                else:
                    dfs.append(subset)
            df_bal = pd.concat(dfs).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
            msg = f"Oversampled minority classes to ~{desired} rows each (ratio={ratio})."
        elif method == "undersample":
            minority_count = counts.min()
            desired = int(round(minority_count * ratio))
            dfs = []
            for cls, cnt in counts.items():
                subset = df_out[df_out[target] == cls]
                if cnt > desired:
                    dfs.append(subset.sample(n=desired, replace=False, random_state=random_state))
                else:
                    dfs.append(subset)
            df_bal = pd.concat(dfs).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
            msg = f"Undersampled majority classes to ~{desired} rows each (ratio={ratio})."
        else:
            return df, f"Unknown rebalancing method: {method}"
        logger.info(f"rebalance_dataset took {time.time() - start_time:.2f} seconds")
        return df_bal, msg
    except Exception as e:
        logger.error(f"Error in rebalance_dataset: {e}")
        return df, f"Error rebalancing dataset: {e}"

def type_convert(
    df: pd.DataFrame,
    column: str,
    type: str = "numeric",
    preview: bool = False,
) -> Tuple[pd.DataFrame, str]:
    """
    Convert a column to a specified data type.
    Returns (transformed_df, message).
    """
    start_time = time.time()
    try:
        if column not in df.columns:
            return df, f"Column {column} not found."
        df_out = df if preview else df.copy()
        if type == "numeric":
            df_out[column] = pd.to_numeric(df_out[column], errors="coerce")
            msg = f"Converted {column} to numeric type."
        elif type == "string":
            df_out[column] = df_out[column].astype(str)
            msg = f"Converted {column} to string type."
        elif type == "category":
            df_out[column] = df_out[column].astype("category")
            msg = f"Converted {column} to category type."
        else:
            msg = f"Unsupported type conversion: {type}."
        logger.info(f"type_convert took {time.time() - start_time:.2f} seconds")
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in type_convert: {e}")
        return df, f"Error converting type: {e}"

def skewness_transform(
    df: pd.DataFrame,
    column: str,
    transform: str = "log",
    preview: bool = False,
) -> Tuple[pd.DataFrame, str]:
    """
    Apply a transformation to reduce skewness in a numeric column.
    Returns (transformed_df, message).
    """
    start_time = time.time()
    try:
        if column not in df.columns:
            return df, f"Column {column} not found."
        if not pd.api.types.is_numeric_dtype(df[column]):
            return df, f"Column {column} is not numeric."
        df_out = df if preview else df.copy()
        if transform == "log":
            if df_out[column].min() <= 0:
                return df, f"Cannot apply log transform to {column} due to non-positive values."
            df_out[column] = np.log(df_out[column])
            msg = f"Applied log transform to {column}."
        elif transform == "square_root":
            if df_out[column].min() < 0:
                return df, f"Cannot apply square root transform to {column} due to negative values."
            df_out[column] = np.sqrt(df_out[column])
            msg = f"Applied square root transform to {column}."
        elif transform == "boxcox":
            from scipy.stats import boxcox
            if df_out[column].min() <= 0:
                return df, f"Cannot apply Box-Cox transform to {column} due to non-positive values."
            df_out[column], _ = boxcox(df_out[column])
            msg = f"Applied Box-Cox transform to {column}."
        else:
            msg = f"Unsupported transform: {transform}."
        logger.info(f"skewness_transform took {time.time() - start_time:.2f} seconds")
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in skewness_transform: {e}")
        return df, f"Error in skewness transform: {e}"

def mask_pii(
    df: pd.DataFrame,
    column: str,
    pii_types: Optional[List[str]] = None,
    preview: bool = False,
) -> Tuple[pd.DataFrame, str]:
    """
    Mask personally identifiable information in a column using regex patterns.
    Returns (transformed_df, message).
    """
    start_time = time.time()
    try:
        if column not in df.columns:
            return df, f"Column {column} not found."
        df_out = df if preview else df.copy()
        pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,4}|\d{3}[-.\s]\d{3}[-.\s]\d{4})\b',
            'credit_card': r'\b(?:\d[ -]*?){13,16}\b'
        }
        applied = []
        df_out[column] = df_out[column].astype(str)
        for pii_type in pii_types or pii_patterns.keys():
            if pii_type in pii_patterns:
                pattern = pii_patterns[pii_type]
                df_out[column] = df_out[column].str.replace(pattern, f"[MASKED_{pii_type.upper()}]", regex=True)
                applied.append(pii_type)
        if not applied:
            return df, f"No valid PII types specified for {column}."
        msg = f"Masked PII ({', '.join(applied)}) in {column} using regex."
        logger.info(f"mask_pii took {time.time() - start_time:.2f} seconds")
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in mask_pii: {e}")
        return df, f"Error masking PII: {e}"
