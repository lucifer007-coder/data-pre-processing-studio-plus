import logging
import time
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Any, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
import regex as re
from scipy.signal import savgol_filter
from pandas.tseries.frequencies import to_offset
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from PIL import Image
import io
import base64
import nltk
from utils.data_utils import dtype_split
import streamlit as st
import urllib.parse
import hashlib

logger = logging.getLogger(__name__)

# Ensure NLTK resources are downloaded
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    logger.error(f"Failed to download NLTK resources: {e}")
    st.error(
        f"Failed to download NLTK resources: {e}\n"
        "Please run the following in Python:\n"
        "import nltk\n"
        "nltk.download('punkt')\n"
        "nltk.download('punkt_tab')\n"
        "nltk.download('stopwords')\n"
        "nltk.download('wordnet')"
    )

def impute_missing(
    df: pd.DataFrame,
    columns: List[str],
    strategy: str = "mean",
    constant_value: Optional[Any] = None,
    n_neighbors: int = 5,
    n_estimators: int = 100,
    preview: bool = False,
) -> Tuple[pd.DataFrame, str]:
    """
    Impute missing values in specified columns using the given strategy.
    Args:
        df: Input DataFrame.
        columns: List of columns to impute.
        strategy: Imputation strategy ('mean', 'median', 'mode', 'constant', 'ffill', 'bfill', 'knn', 'random_forest').
        constant_value: Value to use for constant strategy.
        n_neighbors: Number of neighbors for KNN imputation.
        n_estimators: Number of trees for Random Forest imputation.
        preview: If True, return a copy without modifying the original.
    Returns:
        (transformed_df, message)
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
            for col in valid_cols:
                if col in num_cols:
                    if strategy == "mean":
                        value = df_out[col].mean()
                    else:
                        value = df_out[col].median()
                    df_out[col] = df_out[col].fillna(value)
                    imputed_cols.append(col)
        elif strategy == "mode":
            for col in valid_cols:
                value = df_out[col].mode().iloc[0] if not df_out[col].mode().empty else np.nan
                df_out[col] = df_out[col].fillna(value)
                imputed_cols.append(col)
        elif strategy == "constant":
            if constant_value is None:
                return df, "Constant value must be provided for constant strategy."
            for col in valid_cols:
                df_out[col] = df_out[col].fillna(constant_value)
                imputed_cols.append(col)
        elif strategy in ("ffill", "bfill"):
            for col in valid_cols:
                df_out[col] = df_out[col].fillna(method=strategy)
                imputed_cols.append(col)
        elif strategy == "knn":
            valid_num_cols = [c for c in valid_cols if c in num_cols]
            if not valid_num_cols:
                return df, "KNN imputation requires at least one numeric column."
            if len(df_out) < n_neighbors:
                return df, f"Number of samples ({len(df_out)}) is less than n_neighbors ({n_neighbors})."
            imputer = KNNImputer(n_neighbors=n_neighbors, weights="uniform")
            imputed_data = imputer.fit_transform(df_out[valid_num_cols])
            df_out[valid_num_cols] = pd.DataFrame(imputed_data, columns=valid_num_cols, index=df_out.index)
            imputed_cols.extend(valid_num_cols)
        elif strategy == "random_forest":
            valid_num_cols = [c for c in valid_cols if c in num_cols]
            if not valid_num_cols:
                return df, "Random Forest imputation requires at least one numeric column."
            for col in valid_num_cols:
                if df_out[col].isna().sum() == 0:
                    continue
                # Features: other numeric columns without missing values
                feature_cols = [c for c in num_cols if c != col and df_out[c].isna().sum() == 0]
                if not feature_cols:
                    logger.warning(f"No valid features for Random Forest imputation of {col}.")
                    continue
                mask = df_out[col].isna()
                if mask.sum() == len(df_out):
                    logger.warning(f"All values in {col} are missing.")
                    continue
                # Train on non-missing data
                train_data = df_out[~mask][feature_cols]
                train_target = df_out[~mask][col]
                if len(train_data) < 10:
                    logger.warning(f"Insufficient non-missing data in {col} for Random Forest imputation.")
                    continue
                model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                model.fit(train_data, train_target)
                # Predict missing values
                test_data = df_out[mask][feature_cols]
                if not test_data.empty:
                    df_out.loc[mask, col] = model.predict(test_data)
                imputed_cols.append(col)
        else:
            return df, f"Unsupported imputation strategy: {strategy}"

        msg = f"Imputed missing values in {len(imputed_cols)} columns using {strategy} strategy."
        logger.info(f"{msg} ({time.time() - start_time:.2f} seconds)")
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in impute_missing: {e}")
        return df, f"Error imputing missing values: {e}"

def encode_categorical(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "onehot",
    max_categories: Optional[int] = None,
    group_rare: bool = False,
    ordinal_mappings: Optional[Dict] = None,
    target_column: Optional[str] = None,
    n_components: int = 8,
    preview: bool = False
) -> Tuple[pd.DataFrame, str]:
    """
    Encode categorical columns using one-hot, label, ordinal, target, frequency, or hashing encoding.
    """
    try:
        if not columns:
            return df, "No columns selected for encoding."
        valid_cols = [c for c in columns if c in df.columns]
        if not valid_cols:
            return df, "No valid columns selected for encoding."
        df_out = df if preview else df.copy()

        if method == "target" and (target_column is None or target_column not in df.columns):
            return df, "Target column required for target encoding."
        if method == "ordinal" and not ordinal_mappings:
            return df, "Ordinal mappings required for ordinal encoding."

        processed_cols = []
        for col in valid_cols:
            if max_categories and group_rare:
                top_categories = df_out[col].value_counts().head(max_categories).index
                df_out[col] = df_out[col].where(df_out[col].isin(top_categories), "Rare")

            if method == "onehot":
                if max_categories and not group_rare:
                    top_categories = df_out[col].value_counts().head(max_categories).index
                    df_out[col] = df_out[col].where(df_out[col].isin(top_categories), "Other")
                dummies = pd.get_dummies(df_out[col], prefix=col, dummy_na=False)
                df_out = df_out.drop(columns=[col]).join(dummies)
                processed_cols.append(col)
            elif method == "label":
                le = LabelEncoder()
                df_out[col] = le.fit_transform(df_out[col].astype(str))
                processed_cols.append(col)
            elif method == "ordinal":
                if col in ordinal_mappings:
                    df_out[col] = df_out[col].map(ordinal_mappings[col])
                    processed_cols.append(col)
            elif method == "target":
                target_means = df_out.groupby(col)[target_column].mean()
                df_out[col] = df_out[col].map(target_means)
                processed_cols.append(col)
            elif method == "frequency":
                freq = df_out[col].value_counts(normalize=True)
                df_out[col] = df_out[col].map(freq)
                processed_cols.append(col)
            elif method == "hashing":
                def hash_category(value):
                    if pd.isna(value):
                        return np.nan
                    return int(hashlib.md5(str(value).encode()).hexdigest(), 16) % n_components
                df_out[col] = df_out[col].apply(hash_category)
                processed_cols.append(col)
            else:
                return df, f"Unsupported encoding method: {method}"

        msg = f"Encoded {len(processed_cols)} columns using {method} method" + \
              (f" with {max_categories} max categories" if max_categories else "") + \
              (f" and rare grouping" if group_rare else "") + \
              (f" targeting {target_column}" if target_column else "")
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in encode_categorical: {e}")
        return df, f"Error encoding categorical columns: {e}"

def scale_features(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "standard",
    keep_original: bool = False,
    preview: bool = False
) -> Tuple[pd.DataFrame, str]:
    """
    Scale numeric features using StandardScaler, MinMaxScaler, or RobustScaler.
    """
    try:
        if not columns:
            return df, "No columns selected for scaling."
        valid_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if not valid_cols:
            return df, "No valid numeric columns selected for scaling."
        df_out = df if preview else df.copy()
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        elif method == "robust":
            scaler = RobustScaler()
        else:
            return df, f"Unsupported scaling method: {method}"
        scaled = scaler.fit_transform(df_out[valid_cols])
        if keep_original:
            new_cols = [f"{col}_scaled" for col in valid_cols]
            df_out[new_cols] = pd.DataFrame(scaled, index=df_out.index, columns=new_cols)
        else:
            df_out[valid_cols] = pd.DataFrame(scaled, index=df_out.index, columns=valid_cols)
        msg = f"Scaled {len(valid_cols)} columns using {method} scaler."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in scale_features: {e}")
        return df, f"Error scaling features: {e}"

def extract_domain(df: pd.DataFrame, column: str, new_name: Optional[str] = None, preview: bool = False) -> Tuple[pd.DataFrame, str]:
    """
    Extract domains from URLs in the specified column.
    """
    try:
        if column not in df.columns:
            return df, f"Column {column} not found."
        df_out = df if preview else df.copy()
        url_pattern = r'^(https?://|file://|ftp://)'
        valid_urls = df_out[column].astype(str).str.contains(url_pattern, na=False)
        if not valid_urls.any():
            return df, f"No valid URLs in {column}."

        def get_domain(url):
            try:
                if pd.isna(url) or not isinstance(url, str):
                    return np.nan
                parsed = urllib.parse.urlparse(url)
                return parsed.netloc if parsed.netloc else np.nan
            except Exception:
                return np.nan

        new_col = df_out[column].apply(get_domain)
        target_col = new_name if new_name else column
        df_out[target_col] = new_col
        processed_count = valid_urls.sum()
        msg = f"Extracted domains for {processed_count} URLs in {column} to {target_col}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in extract_domain: {e}")
        return df, f"Error extracting domains: {e}"

def drop_missing(
    df: pd.DataFrame,
    axis: str = "rows",
    threshold: Optional[float] = None,
    columns: Optional[List[str]] = None,
    preview: bool = False
) -> Tuple[pd.DataFrame, str]:
    """
    Drop rows or columns with missing values based on threshold or specific columns.
    """
    try:
        df_out = df if preview else df.copy()
        if axis not in ["rows", "columns"]:
            return df, f"Invalid axis: {axis}. Use 'rows' or 'columns'."
        if columns:
            valid_cols = [c for c in columns if c in df_out.columns]
            if not valid_cols:
                return df, "No valid columns selected for dropping."
            if axis == "rows":
                mask = df_out[valid_cols].isna().any(axis=1)
                initial_rows = len(df_out)
                df_out = df_out[~mask]
                dropped = initial_rows - len(df_out)
                msg = f"Dropped {dropped} rows with missing values in {len(valid_cols)} columns."
            else:
                df_out = df_out.drop(columns=valid_cols)
                msg = f"Dropped {len(valid_cols)} columns: {', '.join(valid_cols)}."
        elif threshold is not None:
            if not 0 <= threshold <= 1:
                return df, "Threshold must be between 0 and 1."
            if axis == "rows":
                mask = df_out.isna().mean(axis=1) <= threshold
                initial_rows = len(df_out)
                df_out = df_out[mask]
                dropped = initial_rows - len(df_out)
                msg = f"Dropped {dropped} rows with missing ratio > {threshold}."
            else:
                mask = df_out.isna().mean() <= threshold
                dropped_cols = df_out.columns[~mask]
                df_out = df_out.loc[:, mask]
                msg = f"Dropped {len(dropped_cols)} columns with missing ratio > {threshold}: {', '.join(dropped_cols)}."
        else:
            initial_shape = df_out.shape
            df_out = df_out.dropna(axis=0 if axis == "rows" else 1)
            dropped = initial_shape[0 if axis == "rows" else 1] - df_out.shape[0 if axis == "rows" else 1]
            msg = f"Dropped {dropped} {axis} with missing values."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in drop_missing: {e}")
        return df, f"Error dropping missing values: {e}"

def normalize_text(
    df: pd.DataFrame,
    columns: List[str],
    lowercase: bool = True,
    trim: bool = True,
    collapse: bool = True,
    remove_special: bool = False,
    preview: bool = False
) -> Tuple[pd.DataFrame, str]:
    """
    Normalize text in specified columns.
    """
    try:
        if not columns:
            return df, "No columns selected for text normalization."
        valid_cols = [c for c in columns if c in df.columns]
        if not valid_cols:
            return df, "No valid columns selected for text normalization."
        df_out = df if preview else df.copy()
        processed_cols = []
        for col in valid_cols:
            series = df_out[col].astype(str)
            if lowercase:
                series = series.str.lower()
            if trim:
                series = series.str.strip()
            if collapse:
                series = series.str.replace(r'\s+', ' ', regex=True)
            if remove_special:
                series = series.str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
            df_out[col] = series
            processed_cols.append(col)
        msg = f"Normalized text in {len(processed_cols)} columns: lowercase={lowercase}, trim={trim}, collapse={collapse}, remove_special={remove_special}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in normalize_text: {e}")
        return df, f"Error normalizing text: {e}"

def standardize_dates(
    df: pd.DataFrame,
    columns: List[str],
    format: str = "%Y-%m-%d",
    preview: bool = False
) -> Tuple[pd.DataFrame, str]:
    """
    Standardize date formats in specified columns.
    """
    try:
        if not columns:
            return df, "No columns selected for date standardization."
        valid_cols = [c for c in columns if c in df.columns]
        if not valid_cols:
            return df, "No valid columns selected for date standardization."
        df_out = df if preview else df.copy()
        processed_cols = []
        for col in valid_cols:
            try:
                df_out[col] = pd.to_datetime(df_out[col], errors='coerce').dt.strftime(format)
                processed_cols.append(col)
            except Exception as e:
                logger.warning(f"Failed to standardize dates in {col}: {e}")
        msg = f"Standardized dates in {len(processed_cols)} columns to format {format}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in standardize_dates: {e}")
        return df, f"Error standardizing dates: {e}"

def unit_convert(
    df: pd.DataFrame,
    column: str,
    factor: float,
    new_name: Optional[str] = None,
    preview: bool = False
) -> Tuple[pd.DataFrame, str]:
    """
    Convert units in a numeric or string column by multiplying by a factor.
    """
    try:
        if column not in df.columns:
            return df, f"Column {column} not found."
        if factor == 0:
            return df, "Conversion factor cannot be zero."
        df_out = df if preview else df.copy()
        target_col = new_name if new_name else column
        if pd.api.types.is_string_dtype(df_out[column]):
            series = df_out[column].astype(str).str.replace(r'[^\d.]', '', regex=True)
            series = pd.to_numeric(series, errors='coerce') * factor
        else:
            series = df_out[column] * factor
        df_out[target_col] = series
        msg = f"Converted units in {column} by factor {factor} to {target_col}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in unit_convert: {e}")
        return df, f"Error converting units: {e}"

def type_convert(
    df: pd.DataFrame,
    column: str,
    type: str,
    preview: bool = False
) -> Tuple[pd.DataFrame, str]:
    """
    Convert the type of a column (e.g., to bool, category).
    """
    try:
        if column not in df.columns:
            return df, f"Column {column} not found."
        df_out = df if preview else df.copy()
        if type == "bool":
            df_out[column] = df_out[column].astype(bool)
            msg = f"Converted {column} to boolean."
        elif type == "category":
            df_out[column] = df_out[column].astype('category')
            msg = f"Converted {column} to category."
        else:
            return df, f"Unsupported type: {type}"
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in type_convert: {e}")
        return df, f"Error converting type: {e}"

def handle_outliers(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "iqr",
    factor: float = 1.5,
    preview: bool = False
) -> Tuple[pd.DataFrame, str]:
    """
    Handle outliers in specified columns using IQR or Z-score.
    """
    try:
        if not columns:
            return df, "No columns selected for outlier handling."
        valid_cols = [c for c in columns if c in df.columns]
        if not valid_cols:
            return df, "No valid columns selected for outlier handling."
        df_out = df if preview else df.copy()
        processed_cols = []
        for col in valid_cols:
            if not pd.api.types.is_numeric_dtype(df_out[col]):
                continue
            if method == "iqr":
                q1 = df_out[col].quantile(0.25)
                q3 = df_out[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - factor * iqr
                upper_bound = q3 + factor * iqr
                df_out[col] = df_out[col].clip(lower=lower_bound, upper=upper_bound)
            elif method == "zscore":
                z_scores = np.abs((df_out[col] - df_out[col].mean()) / df_out[col].std())
                df_out[col] = df_out[col].where(z_scores <= factor, np.nan)
            processed_cols.append(col)
        msg = f"Handled outliers in {len(processed_cols)} columns using {method} method."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in handle_outliers: {e}")
        return df, f"Error handling outliers: {e}"

def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = "first",
    preview: bool = False
) -> Tuple[pd.DataFrame, str]:
    """
    Remove duplicate rows based on subset of columns.
    """
    try:
        df_out = df if preview else df.copy()
        initial_rows = len(df_out)
        valid_subset = [c for c in (subset or df.columns) if c in df.columns]
        if not valid_subset:
            return df, "No valid columns selected for duplicate removal."
        keep = False if keep == "False" else keep
        df_out = df_out.drop_duplicates(subset=valid_subset, keep=keep)
        dropped = initial_rows - len(df_out)
        msg = f"Removed {dropped} duplicate rows based on {len(valid_subset)} columns."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in remove_duplicates: {e}")
        return df, f"Error removing duplicates: {e}"

def rebalance_dataset(
    df: pd.DataFrame,
    target: str,
    method: str = "oversample",
    ratio: float = 1.0,
    preview: bool = False
) -> Tuple[pd.DataFrame, str]:
    """
    Rebalance classification dataset using custom oversampling or undersampling without imblearn.
    """
    try:
        if target not in df.columns:
            return df, f"Target column {target} not found."
        df_out = df if preview else df.copy()
        value_counts = df_out[target].value_counts()
        if len(value_counts) < 2:
            return df, "Target column must have at least two classes."
        
        np.random.seed(42)
        majority_class = value_counts.idxmax()
        majority_count = value_counts.max()
        minority_class = value_counts.idxmin()
        minority_count = value_counts.min()
        
        if method == "oversample":
            target_count = int(majority_count * ratio)
            if target_count < minority_count:
                return df, f"Target count {target_count} is less than minority count {minority_count}."
            minority_df = df_out[df_out[target] == minority_class]
            oversampled_minority = minority_df.sample(n=target_count, replace=True, random_state=42)
            majority_df = df_out[df_out[target] == majority_class]
            df_out = pd.concat([majority_df, oversampled_minority], ignore_index=True)
            msg = f"Oversampled dataset to balance {target} with ratio {ratio} (minority class: {minority_class}, {target_count} samples)."
        elif method == "undersample":
            target_count = int(minority_count * ratio)
            if target_count > majority_count:
                return df, f"Target count {target_count} is greater than majority count {majority_count}."
            majority_df = df_out[df_out[target] == majority_class]
            undersampled_majority = majority_df.sample(n=target_count, replace=False, random_state=42)
            minority_df = df_out[df_out[target] == minority_class]
            df_out = pd.concat([undersampled_majority, minority_df], ignore_index=True)
            msg = f"Undersampled dataset to balance {target} with ratio {ratio} (majority class: {majority_class}, {target_count} samples)."
        else:
            return df, f"Unsupported rebalancing method: {method}"
        
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in rebalance_dataset: {e}")
        return df, f"Error rebalancing dataset: {e}"

def skewness_transform(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "log",
    preview: bool = False
) -> Tuple[pd.DataFrame, str]:
    """
    Apply transformation to reduce skewness in numeric columns.
    """
    try:
        if not columns:
            return df, "No columns selected for skewness transformation."
        valid_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if not valid_cols:
            return df, "No valid numeric columns selected for skewness transformation."
        df_out = df if preview else df.copy()
        processed_cols = []
        for col in valid_cols:
            if method == "log":
                df_out[col] = np.log1p(df_out[col].clip(lower=0))
            elif method == "sqrt":
                df_out[col] = np.sqrt(df_out[col].clip(lower=0))
            else:
                return df, f"Unsupported skewness transformation method: {method}"
            processed_cols.append(col)
        msg = f"Applied {method} transformation to {len(processed_cols)} columns."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in skewness_transform: {e}")
        return df, f"Error applying skewness transformation: {e}"

def mask_pii(
    df: pd.DataFrame,
    columns: List[str],
    patterns: Optional[Dict[str, str]] = None,
    preview: bool = False
) -> Tuple[pd.DataFrame, str]:
    """
    Mask personally identifiable information (PII) in specified columns.
    """
    try:
        if not columns:
            return df, "No columns selected for PII masking."
        valid_cols = [c for c in columns if c in df.columns]
        if not valid_cols:
            return df, "No valid columns selected for PII masking."
        df_out = df if preview else df.copy()
        default_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}-\d{3}-\d{4}\b|\b\d{10}\b',
            'credit_card': r'\b\d{4}-\d{4}-\d{4}-\d{4}\b'
        }
        patterns = patterns or default_patterns
        processed_cols = []
        for col in valid_cols:
            series = df_out[col].astype(str)
            for name, pattern in patterns.items():
                series = series.str.replace(pattern, f'[MASKED_{name.upper()}]', regex=True)
            df_out[col] = series
            processed_cols.append(col)
        msg = f"Masked PII in {len(processed_cols)} columns."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in mask_pii: {e}")
        return df, f"Error masking PII: {e}"

def smooth_time_series(
    df: pd.DataFrame,
    column: str,
    method: str = "moving_average",
    window: int = 5,
    interpolate: str = "none",
    preview: bool = False
) -> Tuple[pd.DataFrame, str]:
    """
    Smooth time-series data in a numeric column.
    """
    try:
        if column not in df.columns:
            return df, f"Column {column} not found."
        if not pd.api.types.is_numeric_dtype(df[column]):
            return df, f"Column {column} is not numeric."
        df_out = df if preview else df.copy()
        if interpolate != "none":
            if interpolate in ["linear", "ffill", "bfill"]:
                df_out[column] = df_out[column].interpolate(method=interpolate)
            else:
                return df, f"Unsupported interpolation method: {interpolate}"
        if method == "moving_average":
            df_out[column] = df_out[column].rolling(window=window, min_periods=1).mean()
        elif method == "savitzky_golay":
            if window % 2 == 0:
                window += 1
            df_out[column] = savgol_filter(df_out[column].fillna(method='ffill'), window_length=window, polyorder=2)
        else:
            return df, f"Unsupported smoothing method: {method}"
        msg = f"Smoothed {column} using {method} with window {window}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in smooth_time_series: {e}")
        return df, f"Error smoothing time-series: {e}"

def resample_time_series(
    df: pd.DataFrame,
    time_column: str,
    freq: str,
    agg_func: str = "mean",
    preview: bool = False
) -> Tuple[pd.DataFrame, str]:
    """
    Resample time-series data to a specified frequency.
    """
    try:
        if time_column not in df.columns:
            return df, f"Time column {time_column} not found."
        df_out = df if preview else df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_out[time_column]):
            df_out[time_column] = pd.to_datetime(df_out[time_column], errors='coerce')
        df_out = df_out.set_index(time_column)
        df_out = df_out.resample(freq)
        if agg_func == "mean":
            df_out = df_out.mean()
        elif agg_func == "sum":
            df_out = df_out.sum()
        elif agg_func == "first":
            df_out = df_out.first()
        elif agg_func == "last":
            df_out = df_out.last()
        else:
            return df, f"Unsupported aggregation function: {agg_func}"
        df_out = df_out.reset_index()
        msg = f"Resampled time-series on {time_column} to frequency {freq} using {agg_func}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in resample_time_series: {e}")
        return df, f"Error resampling time-series: {e}"

def clean_text(
    df: pd.DataFrame,
    column: str,
    remove_stopwords: bool = False,
    lemmatize: bool = False,
    preview: bool = False
) -> Tuple[pd.DataFrame, str]:
    """
    Clean text in a column by removing stopwords and/or lemmatizing.
    """
    try:
        if column not in df.columns:
            return df, f"Column {column} not found."
        df_out = df if preview else df.copy()
        stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        lemmatizer = WordNetLemmatizer() if lemmatize else None
        def process_text(text):
            if pd.isna(text):
                return text
            text = str(text).lower()
            tokens = word_tokenize(text)
            if remove_stopwords:
                tokens = [t for t in tokens if t not in stop_words]
            if lemmatize:
                tokens = [lemmatizer.lemmatize(t) for t in tokens]
            return ' '.join(tokens)
        df_out[column] = df_out[column].apply(process_text)
        msg = f"Cleaned text in {column}: remove_stopwords={remove_stopwords}, lemmatize={lemmatize}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in clean_text: {e}")
        return df, f"Error cleaning text: {e}"

def extract_tfidf(
    df: pd.DataFrame,
    column: str,
    max_features: int = 100,
    preview: bool = False
) -> Tuple[pd.DataFrame, str]:
    """
    Extract TF-IDF features from a text column.
    """
    try:
        if column not in df.columns:
            return df, f"Column {column} not found."
        df_out = df if preview else df.copy()
        tfidf = TfidfVectorizer(max_features=max_features, stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df_out[column].astype(str).fillna(''))
        feature_names = [f"tfidf_{col}" for col in tfidf.get_feature_names_out()]
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names, index=df_out.index)
        df_out = df_out.drop(columns=[column]).join(tfidf_df)
        msg = f"Extracted {len(feature_names)} TF-IDF features from {column}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in extract_tfidf: {e}")
        return df, f"Error extracting TF-IDF features: {e}"

def resize_image(
    df: pd.DataFrame,
    column: str,
    width: int,
    height: int,
    preview: bool = False
) -> Tuple[pd.DataFrame, str]:
    """
    Resize images in a column to specified dimensions.
    """
    try:
        if column not in df.columns:
            return df, f"Column {column} not found."
        valid_images = df[column].astype(str).str.contains(r'\.(png|jpg|jpeg)$|data:image', regex=True, na=False)
        if not valid_images.any():
            return df, f"No valid image paths or base64 strings in {column}."
        df_out = df if preview else df.copy()
        processed_count = 0
        for idx, img_data in df_out[column].items():
            if not valid_images.loc[idx]:
                continue
            try:
                if img_data.startswith("data:image"):
                    format_str, encoded = img_data.split(",", 1)
                    format_type = format_str.split(";")[0].split("/")[-1].upper()
                    img_data = base64.b64decode(encoded)
                    img = Image.open(io.BytesIO(img_data))
                else:
                    img = Image.open(img_data)
                img = img.resize((width, height))
                buf = io.BytesIO()
                img.save(buf, format=format_type if format_type in ["PNG", "JPEG"] else "PNG")
                df_out.at[idx, column] = f"data:image/{format_type.lower()};base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
                processed_count += 1
            except Exception as e:
                logger.warning(f"Failed to resize image at index {idx}: {e}")
        msg = f"Resized {processed_count} images in {column} to {width}x{height}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in resize_image: {e}")
        return df, f"Error resizing images: {e}"

def normalize_image(
    df: pd.DataFrame,
    column: str,
    preview: bool = False
) -> Tuple[pd.DataFrame, str]:
    """
    Normalize pixel values in images to [0,1].
    """
    try:
        if column not in df.columns:
            return df, f"Column {column} not found."
        valid_images = df[column].astype(str).str.contains(r'\.(png|jpg|jpeg)$|data:image', regex=True, na=False)
        if not valid_images.any():
            return df, f"No valid image paths or base64 strings in {column}."
        df_out = df if preview else df.copy()
        processed_count = 0
        for idx, img_data in df_out[column].items():
            if not valid_images.loc[idx]:
                continue
            try:
                if img_data.startswith("data:image"):
                    format_str, encoded = img_data.split(",", 1)
                    format_type = format_str.split(";")[0].split("/")[-1].upper()
                    img_data = base64.b64decode(encoded)
                    img = Image.open(io.BytesIO(img_data))
                else:
                    img = Image.open(img_data)
                img_array = np.array(img) / 255.0
                img = Image.fromarray((img_array * 255).astype(np.uint8))
                buf = io.BytesIO()
                img.save(buf, format=format_type if format_type in ["PNG", "JPEG"] else "PNG")
                df_out.at[idx, column] = f"data:image/{format_type.lower()};base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
                processed_count += 1
            except Exception as e:
                logger.warning(f"Failed to normalize image at index {idx}: {e}")
        msg = f"Normalized {processed_count} images in {column} to [0,1]."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in normalize_image: {e}")
        return df, f"Error normalizing images: {e}"
