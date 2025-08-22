import logging
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import regex as re
from utils.data_utils import dtype_split

logger = logging.getLogger(__name__)

def impute_missing(
    df: pd.DataFrame,
    columns: List[str],
    strategy: str = "mean",
    constant_value: Optional[Any] = None,
) -> Tuple[pd.DataFrame, str]:
    """
    Impute missing values in specified columns using the given strategy.
    Returns (transformed_df, message).
    """
    try:
        df = df.copy()
        if not columns:
            columns = df.columns.tolist()
        valid_cols = [c for c in columns if c in df.columns]
        if not valid_cols:
            return df, "No valid columns selected for imputation."
        num_cols, cat_cols = dtype_split(df)
        imputed_cols = []
        if strategy in ("mean", "median"):
            target_cols = [c for c in valid_cols if c in num_cols]
            if target_cols:
                values = df[target_cols].agg(strategy).to_dict()
                for c in target_cols:
                    if df[c].isna().all():
                        continue
                    imputed_count = df[c].isna().sum()
                    df[c] = df[c].fillna(values[c])
                    imputed_cols.append(f"{c} ({imputed_count} values with {strategy}={values[c]:.2f})")
                msg = f"Imputed {len(target_cols)} numeric columns: {', '.join(imputed_cols)}"
            else:
                msg = "No numeric columns selected for mean/median imputation."
        elif strategy == "mode":
            for c in valid_cols:
                mode = df[c].mode(dropna=True)
                if not mode.empty:
                    imputed_count = df[c].isna().sum()
                    df[c] = df[c].fillna(mode.iloc[0])
                    imputed_cols.append(f"{c} ({imputed_count} values with mode={mode.iloc[0]})")
            msg = f"Imputed {len(valid_cols)} columns: {', '.join(imputed_cols)}"
        elif strategy == "constant":
            val = constant_value if constant_value is not None else 0
            df[valid_cols] = df[valid_cols].fillna(val)
            msg = f"Imputed {len(valid_cols)} columns with constant value {val}."
        elif strategy == "ffill":
            df[valid_cols] = df[valid_cols].fillna(method="ffill")
            msg = f"Forward-filled {len(valid_cols)} columns."
        elif strategy == "bfill":
            df[valid_cols] = df[valid_cols].fillna(method="bfill")
            msg = f"Back-filled {len(valid_cols)} columns."
        else:
            msg = f"Unknown strategy '{strategy}'."
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
    """
    Drop rows or columns with missing values based on axis and threshold.
    Returns (transformed_df, message).
    """
    try:
        df = df.copy()
        valid_cols = columns if columns else df.columns.tolist()
        valid_cols = [c for c in valid_cols if c in df.columns]
        if not valid_cols:
            return df, "No valid columns selected for dropping missing values."
        if axis == "rows":
            initial = len(df)
            if columns:
                to_drop = df[valid_cols].isna().any(axis=1)
                drop_count = to_drop.sum()
                df = df[~to_drop]
                msg = f"Dropped {drop_count} rows with missing values in {valid_cols}."
            elif threshold is not None:
                row_missing_ratio = df[valid_cols].isna().mean(axis=1)
                to_drop = row_missing_ratio >= threshold
                drop_count = to_drop.sum()
                df = df[~to_drop]
                msg = f"Dropped {drop_count} rows with missing ratio ≥ {threshold:.2f}."
            else:
                to_drop = df[valid_cols].isna().any(axis=1)
                drop_count = to_drop.sum()
                df = df[~to_drop]
                msg = f"Dropped {drop_count} rows with any missing values."
        else:
            initial = len(df.columns)
            if threshold is not None:
                col_missing_ratio = df[valid_cols].isna().mean()
                to_drop = col_missing_ratio[col_missing_ratio >= threshold].index.tolist()
                df = df.drop(columns=to_drop)
                msg = f"Dropped {len(to_drop)} columns with missing ratio ≥ {threshold:.2f}: {to_drop}"
            else:
                to_drop = [c for c in valid_cols if df[c].isna().any()]
                df = df.drop(columns=to_drop)
                msg = f"Dropped {len(to_drop)} columns with any missing values: {to_drop}"
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
    remove_special: bool = False,
) -> Tuple[pd.DataFrame, str]:
    """
    Normalize text in specified columns (lowercase, trim, collapse spaces, remove special chars).
    Returns (transformed_df, message).
    """
    try:
        df = df.copy()
        valid_cols = [c for c in columns if c in df.columns]
        if not valid_cols:
            return df, "No valid columns selected for text normalization."
        operations = []
        for c in valid_cols:
            df[c] = df[c].astype(str)
            if trim:
                df[c] = df[c].str.strip()
                operations.append("trimmed")
            if lowercase:
                df[c] = df[c].str.lower()
                operations.append("lowercased")
            if collapse_spaces:
                df[c] = df[c].str.replace(r"\s+", " ", regex=True)
                operations.append("collapsed spaces")
            if remove_special:
                df[c] = df[c].str.replace(r"[^\w\s]", "", regex=True)
                operations.append("removed special characters")
        msg = f"Normalized text for columns {valid_cols}: {', '.join(set(operations))}."
        return df, msg
    except Exception as e:
        logger.error(f"Error in normalize_text: {e}")
        return df, f"Error normalizing text: {e}"

def standardize_dates(
    df: pd.DataFrame, columns: List[str], output_format: str = "%Y-%m-%d"
) -> Tuple[pd.DataFrame, str]:
    """
    Standardize date formats in specified columns.
    Returns (transformed_df, message).
    """
    try:
        df = df.copy()
        valid_cols = [c for c in columns if c in df.columns]
        if not valid_cols:
            return df, "No valid columns selected for date standardization."
        for c in valid_cols:
            dt = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
            df[c] = dt.dt.strftime(output_format)
        msg = f"Standardized date format to {output_format} for columns: {valid_cols}"
        return df, msg
    except Exception as e:
        logger.error(f"Error in standardize_dates: {e}")
        return df, f"Error standardizing dates: {e}"

def unit_convert(
    df: pd.DataFrame, column: str, factor: float, new_name: Optional[str] = None
) -> Tuple[pd.DataFrame, str]:
    """
    Convert units in a numeric column by multiplying by a factor.
    Returns (transformed_df, message).
    """
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

def handle_outliers(
    df: pd.DataFrame, columns: List[str], method: str = "iqr", factor: float = 1.5
) -> Tuple[pd.DataFrame, str]:
    """
    Handle outliers in numeric columns using IQR or z-score.
    Returns (transformed_df, message).
    """
    try:
        df = df.copy()
        valid_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if not valid_cols:
            return df, "No valid numeric columns selected for outlier handling."
        if method == "iqr":
            for c in valid_cols:
                q1 = df[c].quantile(0.25)
                q3 = df[c].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - factor * iqr
                upper = q3 + factor * iqr
                outliers = (df[c] < lower) | (df[c] > upper)
                df.loc[outliers, c] = np.nan
            msg = f"Handled outliers in {len(valid_cols)} columns using IQR (factor={factor})."
        elif method == "zscore":
            for c in valid_cols:
                z_scores = np.abs((df[c] - df[c].mean()) / df[c].std())
                outliers = z_scores > factor
                df.loc[outliers, c] = np.nan
            msg = f"Handled outliers in {len(valid_cols)} columns using z-score (threshold={factor})."
        else:
            msg = f"Unknown outlier method: {method}."
        return df, msg
    except Exception as e:
        logger.error(f"Error in handle_outliers: {e}")
        return df, f"Error handling outliers: {e}"

def remove_duplicates(
    df: pd.DataFrame, subset: Optional[List[str]] = None, keep: str = "first"
) -> Tuple[pd.DataFrame, str]:
    """
    Remove duplicate rows based on all or specified columns.
    Returns (transformed_df, message).
    """
    try:
        df = df.copy()
        initial = len(df)
        valid_subset = [c for c in (subset or df.columns) if c in df.columns]
        if not valid_subset:
            return df, "No valid columns selected for duplicate removal."
        df = df.drop_duplicates(subset=valid_subset, keep=keep)
        removed = initial - len(df)
        msg = f"Removed {removed} duplicate rows based on columns: {valid_subset if subset else 'all'}."
        return df, msg
    except Exception as e:
        logger.error(f"Error in remove_duplicates: {e}")
        return df, f"Error removing duplicates: {e}"

def encode_categorical(
    df: pd.DataFrame, columns: List[str], method: str = "onehot", max_categories: Optional[int] = None
) -> Tuple[pd.DataFrame, str]:
    """
    Encode categorical columns using one-hot, label, or ordinal encoding.
    Returns (transformed_df, message).
    """
    try:
        df = df.copy()
        valid_cols = [c for c in columns if c in df.columns]
        if not valid_cols:
            return df, "No valid columns selected for encoding."
        if method == "onehot":
            for c in valid_cols:
                if max_categories and df[c].nunique() > max_categories:
                    top_cats = df[c].value_counts().index[:max_categories]
                    df[c] = df[c].where(df[c].isin(top_cats), "Other")
            df_encoded = pd.get_dummies(df, columns=valid_cols, prefix=valid_cols)
            new_cols = [col for col in df_encoded.columns if col not in df.columns]
            msg = f"One-hot encoded {len(valid_cols)} columns, added {len(new_cols)} new columns."
            return df_encoded, msg
        elif method == "label":
            le_info = []
            for c in valid_cols:
                if max_categories and df[c].nunique() > max_categories:
                    top_cats = df[c].value_counts().index[:max_categories]
                    df[c] = df[c].where(df[c].isin(top_cats), "Other")
                le = LabelEncoder()
                df[c] = le.fit_transform(df[c].astype(str))
                le_info.append(f"{c} ({len(le.classes_)} classes)")
            msg = f"Label encoded columns: {', '.join(le_info)}. Note: Ensure label encoding suits your model."
            return df, msg
        elif method == "ordinal":
            # Note: Ordinal mapping should be provided via params in a real app
            return df, "Ordinal encoding not implemented (requires mappings)."
        else:
            return df, f"Unknown encoding method: {method}"
    except Exception as e:
        logger.error(f"Error in encode_categorical: {e}")
        return df, f"Error encoding categorical data: {e}"

def scale_features(
    df: pd.DataFrame, columns: List[str], method: str = "standard", keep_original: bool = False
) -> Tuple[pd.DataFrame, str]:
    """
    Scale numeric features using StandardScaler, MinMaxScaler, or RobustScaler.
    Returns (transformed_df, message).
    """
    try:
        df = df.copy()
        cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if not cols:
            return df, "No valid numeric columns selected for scaling."
        if df[cols].isna().any().any():
            return df, "Columns contain missing values; impute first."
        if keep_original:
            new_cols = [f"{c}_scaled" for c in cols]
        else:
            new_cols = cols
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        elif method == "robust":
            scaler = RobustScaler()
        else:
            return df, f"Unknown scaling method: {method}"
        df[new_cols] = scaler.fit_transform(df[cols].values)
        msg = f"Applied {method} scaling to columns: {cols}"
        return df, msg
    except Exception as e:
        logger.error(f"Error in scale_features: {e}")
        return df, f"Error scaling features: {e}"

def rebalance_dataset(
    df: pd.DataFrame, target: str, method: str = "oversample", ratio: float = 1.0, random_state: int = 42
) -> Tuple[pd.DataFrame, str]:
    """
    Rebalance dataset using oversampling (SMOTE) or undersampling.
    Returns (transformed_df, message).
    """
    try:
        if target not in df.columns:
            return df, f"Target column '{target}' not found."
        df = df.copy()
        counts = df[target].value_counts(dropna=False)
        if counts.empty or len(counts) <= 1:
            return df, "Target column has only one class or is empty; skipping rebalancing."
        if pd.api.types.is_numeric_dtype(df[target]) and method != "binning":
            return df, "Numeric target detected; use binning for regression tasks."
        if method == "oversample":
            majority_count = counts.max()
            desired = int(round(majority_count * ratio))
            smote = SMOTE(sampling_strategy={cls: desired for cls in counts.index}, random_state=random_state)
            X, y = df.drop(columns=[target]), df[target]
            X_bal, y_bal = smote.fit_resample(X, y)
            df_bal = pd.concat([X_bal, pd.Series(y_bal, name=target)], axis=1)
            msg = f"SMOTE oversampled minority classes to ~{desired} rows each (ratio={ratio})."
        elif method == "undersample":
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
        else:
            return df, f"Unknown rebalancing method: {method}"
        return df_bal, msg
    except Exception as e:
        logger.error(f"Error in rebalance_dataset: {e}")
        return df, f"Error rebalancing dataset: {e}"

def type_convert(df: pd.DataFrame, column: str, type: str = "numeric") -> Tuple[pd.DataFrame, str]:
    """
    Convert a column to a specified data type.
    Returns (transformed_df, message).
    """
    try:
        df = df.copy()
        if column not in df.columns:
            return df, f"Column {column} not found."
        if type == "numeric":
            df[column] = pd.to_numeric(df[column], errors="coerce")
            msg = f"Converted {column} to numeric type."
        elif type == "string":
            df[column] = df[column].astype(str)
            msg = f"Converted {column} to string type."
        elif type == "category":
            df[column] = df[column].astype("category")
            msg = f"Converted {column} to category type."
        else:
            msg = f"Unsupported type conversion: {type}."
        return df, msg
    except Exception as e:
        logger.error(f"Error in type_convert: {e}")
        return df, f"Error converting type: {e}"

def skewness_transform(df: pd.DataFrame, column: str, transform: str = "log") -> Tuple[pd.DataFrame, str]:
    """
    Apply a transformation to reduce skewness in a numeric column.
    Returns (transformed_df, message).
    """
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
        elif transform == "boxcox":
            from scipy.stats import boxcox
            if df[column].min() <= 0:
                return df, f"Cannot apply Box-Cox transform to {column} due to non-positive values."
            df[column], _ = boxcox(df[column])
            msg = f"Applied Box-Cox transform to {column}."
        else:
            msg = f"Unsupported transform: {transform}."
        return df, msg
    except Exception as e:
        logger.error(f"Error in skewness_transform: {e}")
        return df, f"Error in skewness transform: {e}"

def mask_pii(
    df: pd.DataFrame, column: str, pii_types: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, str]:
    """
    Mask personally identifiable information in a column.
    Returns (transformed_df, message).
    """
    try:
        df = df.copy()
        if column not in df.columns:
            return df, f"Column {column} not found."
        pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,4}|\d{3}[-.\s]\d{3}[-.\s]\d{4})\b',
            'credit_card': r'\b(?:\d[ -]*?){13,16}\b'
        }
        applied = []
        df[column] = df[column].astype(str)
        for pii_type in pii_types or pii_patterns.keys():
            if pii_type in pii_patterns:
                pattern = pii_patterns[pii_type]
                df[column] = df[column].str.replace(pattern, f"[MASKED_{pii_type.upper()}]", regex=True)
                applied.append(pii_type)
        if not applied:
            return df, f"No valid PII types specified for {column}."
        msg = f"Masked PII ({', '.join(applied)}) in {column}."
        return df, msg
    except Exception as e:
        logger.error(f"Error in mask_pii: {e}")
        return df, f"Error masking PII: {e}"
