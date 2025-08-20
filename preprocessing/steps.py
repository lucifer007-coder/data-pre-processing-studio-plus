import logging
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import regex as re
from utils.data_utils import dtype_split

logger = logging.getLogger(__name__)

def impute_missing(
    df: pd.DataFrame,
    columns: List[str],
    strategy: str = "mean",
    constant_value: Optional[Any] = None,
) -> Tuple[pd.DataFrame, str]:
    try:
        df = df.copy()
        if not columns:
            columns = df.columns.tolist()
        num_cols, cat_cols = dtype_split(df)

        if strategy in ("mean", "median"):
            target_cols = [c for c in columns if c in num_cols]
            for c in target_cols:
                val = df[c].mean() if strategy == "mean" else df[c].median()
                df[c] = df[c].fillna(val)
            msg = f"Imputed {len(target_cols)} numeric columns using {strategy}."
        elif strategy == "mode":
            target_cols = columns
            for c in target_cols:
                mode = df[c].mode(dropna=True)
                if not mode.empty:
                    df[c] = df[c].fillna(mode.iloc[0])
            msg = f"Imputed {len(target_cols)} columns using mode."
        elif strategy == "constant":
            val = constant_value if constant_value is not None else 0
            df[columns] = df[columns].fillna(val)
            msg = f"Imputed {len(columns)} columns with constant value {val}."
        else:
            msg = "No imputation performed (unknown strategy)."
        return df, msg
    except Exception as e:
        logger.error(f"Error in impute_missing: {e}")
        return df, f"Error in imputation: {e}"

def drop_missing(
    df: pd.DataFrame,
    axis: str = "rows",
    threshold: Optional[float] = None,
    columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, str]:
    try:
        df = df.copy()
        if axis == "rows":
            initial = len(df)
            if columns:
                df = df.dropna(subset=columns)
                msg = f"Dropped rows with missing in selected columns {columns}. Removed {initial - len(df)} rows."
            elif threshold is not None:
                row_missing_ratio = df.isna().mean(axis=1)
                df = df.loc[row_missing_ratio < threshold]
                msg = f"Dropped rows with missing ratio ≥ {threshold:.2f}. Removed {initial - len(df)} rows."
            else:
                df = df.dropna()
                msg = f"Dropped rows with any missing values. Removed {initial - len(df)} rows."
        else:
            initial = len(df.columns)
            if threshold is not None:
                col_missing_ratio = df.isna().mean(axis=0)
                to_drop = col_missing_ratio[col_missing_ratio >= threshold].index.tolist()
                df = df.drop(columns=to_drop)
                msg = f"Dropped columns with missing ratio ≥ {threshold:.2f}: {to_drop}"
            else:
                to_drop = [c for c in df.columns if df[c].isna().any()]
                df = df.drop(columns=to_drop)
                msg = f"Dropped columns containing any missing values: {to_drop}"
        return df, msg
    except Exception as e:
        logger.error(f"Error in drop_missing: {e}")
        return df, f"Error dropping missing values: {e}"

def normalize_text(
    df: pd.DataFrame,
    columns: List[str],
    lowercase: bool = True,
    trim: bool = True,
    collapse_spaces: bool = True,
) -> Tuple[pd.DataFrame, str]:
    try:
        df = df.copy()
        for c in columns:
            if c in df.columns:
                df[c] = df[c].astype(str)
                if trim:
                    df[c] = df[c].str.strip()
                if lowercase:
                    df[c] = df[c].str.lower()
                if collapse_spaces:
                    df[c] = df[c].str.replace(r"\s+", " ", regex=True)
        return df, f"Normalized text for columns: {columns}"
    except Exception as e:
        logger.error(f"Error in normalize_text: {e}")
        return df, f"Error normalizing text: {e}"

def standardize_dates(
    df: pd.DataFrame, columns: List[str], output_format: str = "%Y-%m-%d"
) -> Tuple[pd.DataFrame, str]:
    try:
        df = df.copy()
        for c in columns:
            dt = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
            df[c] = dt.dt.strftime(output_format)
        return df, f"Standardized date format to {output_format} for columns: {columns}"
    except Exception as e:
        logger.error(f"Error in standardize_dates: {e}")
        return df, f"Error standardizing dates: {e}"

def unit_convert(
    df: pd.DataFrame, column: str, factor: float, new_name: Optional[str] = None
) -> Tuple[pd.DataFrame, str]:
    try:
        df = df.copy()
        if column not in df.columns:
            return df, f"Column {column} not found for unit conversion."
        if not pd.api.types.is_numeric_dtype(df[column]):
            return df, f"Column {column} is not numeric; skipped unit conversion."
        target = new_name if new_name else column
        df[target] = df[column] * factor
        if new_name:
            msg = f"Created '{new_name}' by converting '{column}' with factor {factor}."
        else:
            msg = f"Converted '{column}' in place with factor {factor}."
        return df, msg
    except Exception as e:
        logger.error(f"Error in unit_convert: {e}")
        return df, f"Error in unit conversion: {e}"

def detect_outliers_mask(
    df: pd.DataFrame, columns: List[str], method: str = "IQR", z_thresh: float = 3.0, iqr_k: float = 1.5
) -> pd.Series:
    try:
        mask = pd.Series(False, index=df.index)
        for c in columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                continue
            x = df[c]
            if method == "IQR":
                q1, q3 = x.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower, upper = q1 - iqr_k * iqr, q3 + iqr_k * iqr
                mask_c = (x < lower) | (x > upper)
            else:  # Z-score
                mu, sigma = x.mean(), x.std(ddof=0)
                if sigma == 0 or np.isnan(sigma):
                    mask_c = pd.Series(False, index=x.index)
                else:
                    z = (x - mu) / sigma
                    mask_c = z.abs() > z_thresh
            mask = mask | mask_c
        return mask
    except Exception as e:
        logger.error(f"Error in detect_outliers_mask: {e}")
        return pd.Series(False, index=df.index)

def handle_outliers(
    df: pd.DataFrame,
    columns: List[str],
    method: str,  # 'remove' | 'cap' | 'log1p'
    detect_method: str = "IQR",
    z_thresh: float = 3.0,
    iqr_k: float = 1.5,
) -> Tuple[pd.DataFrame, str]:
    try:
        df = df.copy()
        if not columns:
            return df, "No columns selected for outlier handling."

        if method == "remove":
            mask = detect_outliers_mask(df, columns, detect_method, z_thresh, iqr_k)
            removed = int(mask.sum())
            df = df.loc[~mask]
            msg = f"Removed {removed} outlier rows using {detect_method} on {columns}."
        elif method == "cap":
            for c in columns:
                if not pd.api.types.is_numeric_dtype(df[c]):
                    continue
                if detect_method == "IQR":
                    q1, q3 = df[c].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower, upper = q1 - iqr_k * iqr, q3 + iqr_k * iqr
                else:
                    mu, sigma = df[c].mean(), df[c].std(ddof=0)
                    if sigma == 0 or np.isnan(sigma):
                        continue
                    lower, upper = mu - z_thresh * sigma, mu + z_thresh * sigma
                df[c] = df[c].clip(lower=lower, upper=upper)
            msg = f"Capped outliers using {detect_method} thresholds in columns: {columns}."
        else:  # 'log1p'
            for c in columns:
                if pd.api.types.is_numeric_dtype(df[c]):
                    min_val = df[c].min()
                    shift = 1 - min_val if min_val <= 0 else 0
                    df[c] = np.log1p(df[c] + shift)
            msg = f"Applied log1p transform to columns: {columns}."
        return df, msg
    except Exception as e:
        logger.error(f"Error in handle_outliers: {e}")
        return df, f"Error handling outliers: {e}"

def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]], keep: str = "first") -> Tuple[pd.DataFrame, str]:
    try:
        df = df.copy()
        initial = len(df)
        df = df.drop_duplicates(subset=subset if subset else None, keep=keep)
        removed = initial - len(df)
        msg = f"Removed {removed} duplicate rows (keep={keep}) using columns: {subset if subset else 'ALL'}."
        return df, msg
    except Exception as e:
        logger.error(f"Error in remove_duplicates: {e}")
        return df, f"Error removing duplicates: {e}"

def encode_categorical(
    df: pd.DataFrame, columns: List[str], method: str = "onehot"
) -> Tuple[pd.DataFrame, str]:
    try:
        df = df.copy()
        if method == "onehot":
            before_cols = set(df.columns)
            df = pd.get_dummies(df, columns=columns, drop_first=False, dummy_na=False)
            added = list(set(df.columns) - before_cols)
            msg = f"One-hot encoded {len(columns)} columns. New columns: {len(added)}."
        else:  # label
            le_info = []
            for c in columns:
                if c in df.columns:
                    le = LabelEncoder()
                    df[c] = df[c].astype(str).fillna("NaN")
                    df[c] = le.fit_transform(df[c])
                    le_info.append(c)
            msg = f"Label encoded columns: {le_info}"
        return df, msg
    except Exception as e:
        logger.error(f"Error in encode_categorical: {e}")
        return df, f"Error encoding categorical data: {e}"

def scale_features(df: pd.DataFrame, columns: List[str], method: str = "standard") -> Tuple[pd.DataFrame, str]:
    try:
        df = df.copy()
        cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if not cols:
            return df, "No numeric columns selected for scaling."
        scaler = StandardScaler() if method == "standard" else MinMaxScaler()
        df[cols] = scaler.fit_transform(df[cols].values)
        msg = f"Applied {'StandardScaler' if method == 'standard' else 'MinMaxScaler'} to columns: {cols}."
        return df, msg
    except Exception as e:
        logger.error(f"Error in scale_features: {e}")
        return df, f"Error scaling features: {e}"

def rebalance_dataset(
    df: pd.DataFrame, target: str, method: str = "oversample", ratio: float = 1.0, random_state: int = 42
) -> Tuple[pd.DataFrame, str]:
    try:
        if target not in df.columns:
            return df, f"Target column '{target}' not found."

        df = df.copy()
        counts = df[target].value_counts(dropna=False)
        if counts.empty or len(counts) <= 1:
            return df, "Target column has only one class or is empty; skipping rebalancing."

        if method == "oversample":
            majority_count = counts.max()
            desired = int(round(majority_count * ratio))
            dfs = []
            for cls, cnt in counts.items():
                subset = df[df[target] == cls]
                if cnt < desired:
                    add = subset.sample(n=desired - cnt, replace=True, random_state=random_state)
                    dfs.append(pd.concat([subset, add]))
                else:
                    dfs.append(subset)
            df_bal = pd.concat(dfs).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
            msg = f"Oversampled minority classes to ~{desired} rows each (ratio={ratio})."
        else:  # undersample
            minority_count = counts.min()
            desired = int(round(minority_count * ratio))
            dfs = []
            for cls, cnt in counts.items():
                subset = df[df[target] == cls]
                if cnt > desired:
                    dfs.append(subset.sample(n=desired, replace=False, random_state=random_state))
                else:
                    dfs.append(subset)
            df_bal = pd.concat(dfs).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
            msg = f"Undersampled majority classes to ~{desired} rows each (ratio={ratio})."
        return df_bal, msg
    except Exception as e:
        logger.error(f"Error in rebalance_dataset: {e}")
        return df, f"Error rebalancing dataset: {e}"

def type_convert(df: pd.DataFrame, column: str, type: str = "numeric") -> Tuple[pd.DataFrame, str]:
    try:
        df = df.copy()
        if column not in df.columns:
            return df, f"Column {column} not found."
        if type == "numeric":
            df[column] = pd.to_numeric(df[column], errors="coerce")
            msg = f"Converted {column} to numeric type."
        else:
            msg = f"Unsupported type conversion: {type}."
        return df, msg
    except Exception as e:
        logger.error(f"Error in type_convert: {e}")
        return df, f"Error converting type: {e}"

def skewness_transform(df: pd.DataFrame, column: str, transform: str = "log") -> Tuple[pd.DataFrame, str]:
    try:
        df = df.copy()
        if column not in df.columns:
            return df, f"Column {column} not found."
        if not pd.api.types.is_numeric_dtype(df[column]):
            return df, f"Column {column} is not numeric."
        if transform == "log":
            if df[column].min() <= 0:
                return df, f"Cannot apply log transform to {column} due to non-positive values."
            df[column] = np.log(df[column])
            msg = f"Applied log transform to {column}."
        elif transform == "square_root":
            if df[column].min() < 0:
                return df, f"Cannot apply square root transform to {column} due to negative values."
            df[column] = np.sqrt(df[column])
            msg = f"Applied square root transform to {column}."
        else:
            msg = f"Unsupported transform: {transform}."
        return df, msg
    except Exception as e:
        logger.error(f"Error in skewness_transform: {e}")
        return df, f"Error in skewness transform: {e}"

def mask_pii(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, str]:
    try:
        df = df.copy()
        if column not in df.columns:
            return df, f"Column {column} not found."
        pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(\+\d{1,3}[-.\s]?\d{10}|\d{3}[-.\s]\d{3}[-.\s]\d{4})\b',
            'credit_card': r'\b(?:\d[ -]*?){13,16}\b'
        }
        for pii_type, pattern in pii_patterns.items():
            df[column] = df[column].astype(str).str.replace(pattern, f"[MASKED_{pii_type.upper()}]", regex=True)
        msg = f"Masked potential PII in {column}."
        return df, msg
    except Exception as e:
        logger.error(f"Error in mask_pii: {e}")
        return df, f"Error masking PII: {e}"
