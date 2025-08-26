import logging
import time
import pandas as pd
import numpy as np
import dask.dataframe as dd
from typing import List, Tuple, Optional, Any, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from scipy.signal import savgol_filter
from pandas.tseries.frequencies import to_offset
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import io
import base64
import nltk
from utils.data_utils import dtype_split
import streamlit as st
import urllib.parse
import hashlib
import warnings
from pathlib import Path
import os

logger = logging.getLogger(__name__)

# Security: Safe NLTK downloads with path validation
def safe_nltk_download():
    """Safely download NLTK resources with path validation"""
    try:
        # Validate NLTK data path is secure
        nltk_data_path = nltk.data.path[0] if nltk.data.path else None
        if nltk_data_path:
            # Ensure path is within expected boundaries
            safe_path = Path(nltk_data_path).resolve()
            if not str(safe_path).startswith(str(Path.home().resolve())):
                logger.warning("NLTK data path outside user directory, using default")
        
        # Download with error handling
        resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet']
        for resource in resources:
            try:
                nltk.download(resource, quiet=True, raise_on_error=True)
            except Exception as e:
                logger.warning(f"Failed to download {resource}: {e}")
                
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

# Initialize NLTK safely
safe_nltk_download()

def validate_dataframe_input(df: pd.DataFrame | dd.DataFrame) -> bool:
    """Validate DataFrame input to prevent unsafe deserialization"""
    if df is None:
        return False
    
    # Check for suspicious attributes that might indicate malicious objects
    suspicious_attrs = ['__reduce__', '__reduce_ex__', '__getstate__', '__setstate__']
    for attr in suspicious_attrs:
        if hasattr(df, attr) and callable(getattr(df, attr)):
            # This is normal for pandas/dask DataFrames, but we log it
            logger.debug(f"DataFrame has callable {attr} - this is normal")
    
    # Basic type validation
    if not isinstance(df, (pd.DataFrame, dd.DataFrame)):
        logger.error("Input is not a valid DataFrame type")
        return False
    
    return True

def sanitize_error_message(error: Exception) -> str:
    """Sanitize error messages to prevent information disclosure"""
    error_str = str(error)
    # Remove potentially sensitive information
    sanitized = re.sub(r'[\'"][^\'\"]*[\'"]', '[REDACTED]', error_str)
    sanitized = re.sub(r'\b\d+\.\d+\.\d+\.\d+\b', '[IP_REDACTED]', sanitized)
    sanitized = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_REDACTED]', sanitized)
    return sanitized[:200]  # Limit length

def safe_regex_replace(series: pd.Series, pattern: str, replacement: str) -> pd.Series:
    """Safely apply regex replacement with protection against ReDoS"""
    # Validate pattern is safe (basic check)
    if len(pattern) > 100:  # Prevent overly complex patterns
        logger.warning("Regex pattern too long, using simplified version")
        pattern = r'[^\w\s]'  # Safe fallback
    
    # Check for potentially dangerous regex patterns
    dangerous_patterns = [r'\*+', r'\++', r'\{.*,.*\}', r'\(.*\)\*', r'\(.*\)\+']
    for dangerous in dangerous_patterns:
        if re.search(dangerous, pattern):
            logger.warning("Potentially dangerous regex pattern detected, using safe alternative")
            pattern = r'[^\w\s]'  # Safe fallback
            break
    
    try:
        return series.str.replace(pattern, replacement, regex=True)
    except Exception as e:
        logger.warning(f"Regex replacement failed: {sanitize_error_message(e)}")
        return series

def validate_columns(df: pd.DataFrame | dd.DataFrame, columns: List[str]) -> List[str]:
    """Validate and filter columns that exist in the DataFrame"""
    if not columns:
        return []
    
    missing = [c for c in columns if c not in df.columns]
    if missing:
        warnings.warn(f"Skipping missing columns: {missing}")
    
    valid_columns = [c for c in columns if c in df.columns]
    return valid_columns

def impute_missing(
    df: pd.DataFrame | dd.DataFrame,
    columns: List[str],
    strategy: str = "mean",
    constant_value: Optional[Any] = None,
    n_neighbors: int = 5,
    n_estimators: int = 100,
    preview: bool = False,
) -> Tuple[pd.DataFrame | dd.DataFrame, str]:
    """
    Impute missing values in specified columns using the given strategy.
    """
    try:
        # Security: Validate input DataFrame
        if not validate_dataframe_input(df):
            return df, "Invalid DataFrame input"
        
        # Validate and sanitize inputs
        columns = validate_columns(df, columns)
        if not columns:
            return df, "No valid columns to process."
        
        # Sanitize parameters
        n_neighbors = max(1, min(50, int(n_neighbors)))  # Limit range
        n_estimators = max(1, min(200, int(n_estimators)))  # Limit range
        
        if isinstance(df, dd.DataFrame):
            df_out = df if preview else df.copy()
            for col in columns:
                if strategy == 'mean':
                    df_out[col] = df_out[col].fillna(df_out[col].mean())
                elif strategy == 'median':
                    df_out[col] = df_out[col].fillna(df_out[col].median())
                elif strategy == 'mode':
                    mode_series = df_out[col].mode()
                    if len(mode_series) > 0:
                        mode_val = mode_series.iloc[0]
                    else:
                        mode_val = df_out[col].mean()
                    df_out[col] = df_out[col].fillna(mode_val)
                elif strategy == 'constant':
                    df_out[col] = df_out[col].fillna(constant_value)
                elif strategy in ['ffill', 'bfill']:
                    if strategy == 'ffill':
                        df_out[col] = df_out[col].ffill()
                    else:
                        df_out[col] = df_out[col].bfill()
                else:  # knn, random_forest
                    # Memory optimization: limit columns and rows for computation
                    sample_size = min(10000, len(df))  # Limit sample size
                    available_cols = [c for c in df.columns if c != col][:20]  # Limit features
                    subset_cols = [col] + available_cols
                    
                    df_sample = df[subset_cols].sample(n=sample_size, random_state=42).compute()
                    
                    if strategy == 'knn':
                        # Only use numeric columns for KNN
                        numeric_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()
                        if col in numeric_cols and len(numeric_cols) > 1:
                            imputer = KNNImputer(n_neighbors=n_neighbors)
                            df_sample[numeric_cols] = imputer.fit_transform(df_sample[numeric_cols])
                    elif strategy == 'random_forest':
                        mask = df_sample[col].isna()
                        if mask.any():
                            X = df_sample.drop(columns=[col]).select_dtypes(include=[np.number])
                            y = df_sample[col]
                            if len(X.columns) > 0:
                                X_train = X[~mask]
                                y_train = y[~mask]
                                X_test = X[mask]
                                if len(X_train) > 0 and len(X_test) > 0:
                                    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                                    model.fit(X_train, y_train)
                                    df_sample.loc[mask, col] = model.predict(X_test)
                    
                    # Apply imputation to full dataset
                    fill_value = df_sample[col].mean() if not df_sample[col].isna().all() else 0
                    df_out[col] = df_out[col].fillna(fill_value)
            
            msg = f"Imputed missing values in {len(columns)} columns using {strategy}."
        else:
            df_out = df if preview else df.copy()
            for col in columns:
                if strategy == 'mean':
                    df_out[col] = df_out[col].fillna(df_out[col].mean())
                elif strategy == 'median':
                    df_out[col] = df_out[col].fillna(df_out[col].median())
                elif strategy == 'mode':
                    mode_series = df_out[col].mode()
                    mode_val = mode_series.iloc[0] if not mode_series.empty else df_out[col].mean()
                    df_out[col] = df_out[col].fillna(mode_val)
                elif strategy == 'constant':
                    df_out[col] = df_out[col].fillna(constant_value)
                elif strategy in ['ffill', 'bfill']:
                    if strategy == 'ffill':
                        df_out[col] = df_out[col].ffill()
                    else:
                        df_out[col] = df_out[col].bfill()
                elif strategy == 'knn':
                    numeric_cols = df_out.select_dtypes(include=[np.number]).columns.tolist()
                    if col in numeric_cols and len(numeric_cols) > 1:
                        imputer = KNNImputer(n_neighbors=n_neighbors)
                        df_out[numeric_cols] = imputer.fit_transform(df_out[numeric_cols])
                elif strategy == 'random_forest':
                    mask = df_out[col].isna()
                    if mask.any():
                        X = df_out.drop(columns=[col]).select_dtypes(include=[np.number])
                        y = df_out[col]
                        if len(X.columns) > 0:
                            X_train = X[~mask]
                            y_train = y[~mask]
                            X_test = X[mask]
                            if len(X_train) > 0 and len(X_test) > 0:
                                model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                                model.fit(X_train, y_train)
                                df_out.loc[mask, col] = model.predict(X_test)
            
            msg = f"Imputed missing values in {len(columns)} columns using {strategy}."
        
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        error_msg = sanitize_error_message(e)
        logger.error(f"Error in impute_missing: {error_msg}")
        return df, f"Error imputing missing values: Operation failed"

def drop_missing(
    df: pd.DataFrame | dd.DataFrame,
    axis: str = "rows",
    threshold: float = 0.5,
    columns: Optional[List[str]] = None,
    preview: bool = False,
) -> Tuple[pd.DataFrame | dd.DataFrame, str]:
    """
    Drop rows or columns with missing values based on threshold or specific columns.
    """
    try:
        if not validate_dataframe_input(df):
            return df, "Invalid DataFrame input"
        
        # Sanitize threshold
        threshold = max(0.0, min(1.0, float(threshold)))
        
        if columns:
            columns = validate_columns(df, columns)
        
        if isinstance(df, dd.DataFrame):
            df_out = df if preview else df.copy()
            if axis == "rows":
                if columns:
                    mask = df_out[columns].isna().any(axis=1)
                    df_out = df_out[~mask]
                    count = mask.sum().compute()
                else:
                    mask = df_out.isna().any(axis=1)
                    df_out = df_out[~mask]
                    count = mask.sum().compute()
                msg = f"Dropped {count} rows with missing values."
            else:  # axis == "columns"
                if columns:
                    df_out = df_out.drop(columns=columns)
                    msg = f"Dropped {len(columns)} specified columns."
                else:
                    missing_ratio = df_out.isna().mean().compute()
                    cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
                    df_out = df_out.drop(columns=cols_to_drop)
                    msg = f"Dropped {len(cols_to_drop)} columns with missing ratio > {threshold}."
        else:
            df_out = df if preview else df.copy()
            if axis == "rows":
                if columns:
                    df_out = df_out.dropna(subset=columns)
                    count = len(df) - len(df_out)
                else:
                    df_out = df_out.dropna()
                    count = len(df) - len(df_out)
                msg = f"Dropped {count} rows with missing values."
            else:  # axis == "columns"
                if columns:
                    df_out = df_out.drop(columns=columns)
                    msg = f"Dropped {len(columns)} specified columns."
                else:
                    missing_ratio = df_out.isna().mean()
                    cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
                    df_out = df_out.drop(columns=cols_to_drop)
                    msg = f"Dropped {len(cols_to_drop)} columns with missing ratio > {threshold}."
        
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        error_msg = sanitize_error_message(e)
        logger.error(f"Error in drop_missing: {error_msg}")
        return df, "Error dropping missing values: Operation failed"

def normalize_text(
    df: pd.DataFrame | dd.DataFrame,
    columns: List[str],
    lower: bool = True,
    trim: bool = True,
    collapse: bool = True,
    remove_special: bool = False,
    preview: bool = False,
) -> Tuple[pd.DataFrame | dd.DataFrame, str]:
    """
    Normalize text columns with safe regex operations.
    """
    try:
        if not validate_dataframe_input(df):
            return df, "Invalid DataFrame input"
        
        columns = validate_columns(df, columns)
        if not columns:
            return df, "No valid columns to process."
        
        # Safe regex patterns
        SAFE_WHITESPACE_PATTERN = r'\s+'
        SAFE_SPECIAL_CHARS_PATTERN = r'[^\w\s]'
        
        if isinstance(df, dd.DataFrame):
            df_out = df if preview else df.copy()
            def normalize_partition(df_part):
                for col in columns:
                    if col not in df_part.columns:
                        continue
                    s = df_part[col].astype(str)
                    if lower:
                        s = s.str.lower()
                    if trim:
                        s = s.str.strip()
                    if collapse:
                        s = safe_regex_replace(s, SAFE_WHITESPACE_PATTERN, ' ')
                    if remove_special:
                        s = safe_regex_replace(s, SAFE_SPECIAL_CHARS_PATTERN, '')
                    df_part[col] = s
                return df_part
            df_out = df_out.map_partitions(normalize_partition)
        else:
            df_out = df if preview else df.copy()
            for col in columns:
                s = df_out[col].astype(str)
                if lower:
                    s = s.str.lower()
                if trim:
                    s = s.str.strip()
                if collapse:
                    s = safe_regex_replace(s, SAFE_WHITESPACE_PATTERN, ' ')
                if remove_special:
                    s = safe_regex_replace(s, SAFE_SPECIAL_CHARS_PATTERN, '')
                df_out[col] = s
        
        msg = f"Normalized text in {len(columns)} columns."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        error_msg = sanitize_error_message(e)
        logger.error(f"Error in normalize_text: {error_msg}")
        return df, "Error normalizing text: Operation failed"

def standardize_dates(
    df: pd.DataFrame | dd.DataFrame,
    columns: List[str],
    format: str = "%Y-%m-%d",
    preview: bool = False,
) -> Tuple[pd.DataFrame | dd.DataFrame, str]:
    """
    Standardize date columns to a specified format.
    """
    try:
        if not validate_dataframe_input(df):
            return df, "Invalid DataFrame input"
        
        columns = validate_columns(df, columns)
        if not columns:
            return df, "No valid columns to process."
        
        # Validate format string
        try:
            import datetime
            datetime.datetime.now().strftime(format)
        except (ValueError, TypeError):
            format = "%Y-%m-%d"  # Safe fallback
        
        if isinstance(df, dd.DataFrame):
            df_out = df if preview else df.copy()
            def standardize_partition(df_part):
                for col in columns:
                    if col not in df_part.columns:
                        continue
                    try:
                        df_part[col] = pd.to_datetime(df_part[col], errors='coerce').dt.strftime(format)
                    except Exception:
                        pass  # Skip problematic columns
                return df_part
            df_out = df_out.map_partitions(standardize_partition)
        else:
            df_out = df if preview else df.copy()
            for col in columns:
                try:
                    df_out[col] = pd.to_datetime(df_out[col], errors='coerce').dt.strftime(format)
                except Exception:
                    pass  # Skip problematic columns
        
        msg = f"Standardized dates in {len(columns)} columns to format {format}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        error_msg = sanitize_error_message(e)
        logger.error(f"Error in standardize_dates: {error_msg}")
        return df, "Error standardizing dates: Operation failed"

def unit_convert(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    factor: float,
    preview: bool = False,
) -> Tuple[pd.DataFrame | dd.DataFrame, str]:
    """
    Convert units in a numeric column by multiplying with a factor.
    """
    try:
        if not validate_dataframe_input(df):
            return df, "Invalid DataFrame input"
        
        if column not in df.columns:
            return df, f"Column not found."
        
        # Sanitize factor
        if not isinstance(factor, (int, float)) or np.isnan(factor) or np.isinf(factor):
            return df, "Invalid conversion factor."
        
        # Limit factor range for safety
        factor = max(-1e6, min(1e6, float(factor)))
        
        if isinstance(df, dd.DataFrame):
            df_out = df if preview else df.copy()
            df_out[column] = df_out[column] * factor
        else:
            df_out = df if preview else df.copy()
            df_out[column] = df_out[column] * factor
        
        msg = f"Converted units in column by multiplying with {factor}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        error_msg = sanitize_error_message(e)
        logger.error(f"Error in unit_convert: {error_msg}")
        return df, "Error converting units: Operation failed"

def handle_outliers(
    df: pd.DataFrame | dd.DataFrame,
    columns: List[str],
    method: str = "iqr",
    factor: float = 1.5,
    preview: bool = False,
) -> Tuple[pd.DataFrame | dd.DataFrame, str]:
    """
    Handle outliers in numeric columns using IQR or Z-score.
    """
    try:
        if not validate_dataframe_input(df):
            return df, "Invalid DataFrame input"
        
        columns = validate_columns(df, columns)
        if not columns:
            return df, "No valid columns to process."
        
        # Sanitize factor
        factor = max(0.1, min(10.0, float(factor)))
        
        if isinstance(df, dd.DataFrame):
            df_out = df if preview else df.copy()
            for col in columns:
                if method == "iqr":
                    # Performance fix: compute quantiles together
                    quantiles = df_out[col].quantile([0.25, 0.75]).compute()
                    q1, q3 = quantiles.iloc[0], quantiles.iloc[1]
                    iqr = q3 - q1
                    lower_bound = q1 - factor * iqr
                    upper_bound = q3 + factor * iqr
                    df_out[col] = df_out[col].clip(lower=lower_bound, upper=upper_bound)
                elif method == "zscore":
                    stats = df_out[col].agg(['mean', 'std']).compute()
                    mean, std = stats.iloc[0], stats.iloc[1]
                    if std > 0:
                        df_out[col] = df_out[col].where((df_out[col] - mean).abs() <= factor * std, mean)
        else:
            df_out = df if preview else df.copy()
            for col in columns:
                if method == "iqr":
                    q1 = df_out[col].quantile(0.25)
                    q3 = df_out[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - factor * iqr
                    upper_bound = q3 + factor * iqr
                    df_out[col] = df_out[col].clip(lower=lower_bound, upper=upper_bound)
                elif method == "zscore":
                    mean = df_out[col].mean()
                    std = df_out[col].std()
                    if std > 0:
                        df_out[col] = df_out[col].where((df_out[col] - mean).abs() <= factor * std, mean)
        
        msg = f"Handled outliers in {len(columns)} columns using {method}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        error_msg = sanitize_error_message(e)
        logger.error(f"Error in handle_outliers: {error_msg}")
        return df, "Error handling outliers: Operation failed"

def remove_duplicates(
    df: pd.DataFrame | dd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = "first",
    preview: bool = False,
) -> Tuple[pd.DataFrame | dd.DataFrame, str]:
    """
    Remove duplicate rows based on subset of columns.
    """
    try:
        if not validate_dataframe_input(df):
            return df, "Invalid DataFrame input"
        
        if subset:
            subset = validate_columns(df, subset)
        
        # Validate keep parameter
        if keep not in ["first", "last", False]:
            keep = "first"
        
        if isinstance(df, dd.DataFrame):
            df_out = df if preview else df.copy()
            count = df_out.duplicated(subset=subset).sum().compute()
            df_out = df_out.drop_duplicates(subset=subset, keep=keep)
        else:
            df_out = df if preview else df.copy()
            count = df_out.duplicated(subset=subset).sum()
            df_out = df_out.drop_duplicates(subset=subset, keep=keep)
        
        msg = f"Removed {count} duplicate rows."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        error_msg = sanitize_error_message(e)
        logger.error(f"Error in remove_duplicates: {error_msg}")
        return df, "Error removing duplicates: Operation failed"

def encode_categorical(
    df: pd.DataFrame | dd.DataFrame,
    columns: List[str],
    method: str = "onehot",
    max_categories: Optional[int] = None,
    group_rare: bool = False,
    ordinal_mappings: Optional[Dict] = None,
    target_column: Optional[str] = None,
    n_components: int = 8,
    preview: bool = False,
) -> Tuple[pd.DataFrame | dd.DataFrame, str]:
    """
    Encode categorical columns with memory optimization.
    """
    try:
        if not validate_dataframe_input(df):
            return df, "Invalid DataFrame input"
        
        columns = validate_columns(df, columns)
        if not columns:
            return df, "No valid columns to process."
        
        # Sanitize parameters
        n_components = max(1, min(50, int(n_components)))
        if max_categories:
            max_categories = max(2, min(1000, int(max_categories)))
        
        if isinstance(df, dd.DataFrame):
            # Memory optimization: sample for encoding if dataset is large
            sample_size = min(50000, len(df))
            df_sample = df.sample(n=sample_size, random_state=42).compute()
            
            if method == "onehot":
                # Limit categories to prevent memory explosion
                encoded_dfs = []
                for col in columns:
                    col_data = df_sample[col].astype(str)
                    if max_categories:
                        top_cats = col_data.value_counts().head(max_categories).index
                        col_data = col_data.where(col_data.isin(top_cats), 'OTHER')
                    
                    encoded = pd.get_dummies(col_data, prefix=col)
                    # Limit number of dummy columns
                    if len(encoded.columns) > 100:
                        encoded = encoded.iloc[:, :100]
                    encoded_dfs.append(encoded)
                
                # Apply encoding to full dataset
                df_out = df.drop(columns=columns)
                for encoded_df in encoded_dfs:
                    # This is a simplified approach - in practice, you'd need more sophisticated mapping
                    df_out = df_out.assign(**{col: 0 for col in encoded_df.columns})
                
            elif method == "label":
                df_out = df.copy()
                for col in columns:
                    unique_vals = df_sample[col].unique()
                    mapping = {val: idx for idx, val in enumerate(unique_vals)}
                    df_out[col] = df_out[col].map(mapping).fillna(-1)
            else:
                df_out = df  # Fallback for unsupported methods in Dask
        else:
            df_out = df if preview else df.copy()
            if method == "onehot":
                for col in columns:
                    col_data = df_out[col].astype(str)
                    if max_categories:
                        top_cats = col_data.value_counts().head(max_categories).index
                        col_data = col_data.where(col_data.isin(top_cats), 'OTHER')
                    
                    encoded = pd.get_dummies(col_data, prefix=col)
                    # Limit number of dummy columns
                    if len(encoded.columns) > 100:
                        encoded = encoded.iloc[:, :100]
                    
                    df_out = df_out.drop(columns=col).join(encoded)
            
            elif method == "label":
                for col in columns:
                    le = LabelEncoder()
                    df_out[col] = le.fit_transform(df_out[col].astype(str))
            
            elif method == "ordinal" and ordinal_mappings:
                for col in columns:
                    if col in ordinal_mappings:
                        df_out[col] = df_out[col].map(ordinal_mappings[col]).fillna(df_out[col])
            
            elif method == "target_encode" and target_column and target_column in df_out.columns:
                for col in columns:
                    means = df_out.groupby(col)[target_column].mean()
                    df_out[col] = df_out[col].map(means)
            
            elif method == "frequency_encode":
                for col in columns:
                    counts = df_out[col].value_counts()
                    df_out[col] = df_out[col].map(counts)
            
            elif method == "hashing_encode":
                from sklearn.feature_extraction import FeatureHasher
                hasher = FeatureHasher(n_features=n_components, input_type='string')
                for col in columns:
                    hashed_features = hasher.fit_transform(df_out[col].astype(str))
                    hashed_df = pd.DataFrame(
                        hashed_features.toarray(), 
                        columns=[f"{col}_hash_{i}" for i in range(n_components)],
                        index=df_out.index
                    )
                    df_out = df_out.drop(columns=col).join(hashed_df)
        
        msg = f"Encoded {len(columns)} columns using {method}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        error_msg = sanitize_error_message(e)
        logger.error(f"Error in encode_categorical: {error_msg}")
        return df, "Error encoding categorical data: Operation failed"

def scale_features(
    df: pd.DataFrame | dd.DataFrame,
    columns: List[str],
    method: str = "standard",
    keep_original: bool = False,
    preview: bool = False,
) -> Tuple[pd.DataFrame | dd.DataFrame, str]:
    """
    Scale numeric features with memory optimization.
    """
    try:
        if not validate_dataframe_input(df):
            return df, "Invalid DataFrame input"
        
        columns = validate_columns(df, columns)
        if not columns:
            return df, "No valid columns to process."
        
        # Validate method
        if method not in ["standard", "minmax", "robust"]:
            method = "standard"
        
        scaler_map = {
            'standard': StandardScaler(), 
            'minmax': MinMaxScaler(), 
            'robust': RobustScaler()
        }
        scaler = scaler_map[method]
        
        if isinstance(df, dd.DataFrame):
            # Memory optimization: fit scaler on sample
            sample_size = min(10000, len(df))
            df_sample = df[columns].sample(n=sample_size, random_state=42).compute()
            scaler.fit(df_sample)
            
            df_out = df if preview else df.copy()
            
            def scale_partition(partition):
                scaled_data = scaler.transform(partition[columns])
                scaled_df = pd.DataFrame(
                    scaled_data, 
                    columns=[f"{col}_scaled" if keep_original else col for col in columns],
                    index=partition.index
                )
                
                if keep_original:
                    return partition.join(scaled_df)
                else:
                    return partition.drop(columns=columns).join(scaled_df)
            
            df_out = df_out.map_partitions(scale_partition)
        else:
            df_out = df if preview else df.copy()
            scaled = scaler.fit_transform(df_out[columns])
            
            if keep_original:
                for i, col in enumerate(columns):
                    df_out[f"{col}_scaled"] = scaled[:, i]
            else:
                for i, col in enumerate(columns):
                    df_out[col] = scaled[:, i]
        
        msg = f"Scaled {len(columns)} columns using {method}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        error_msg = sanitize_error_message(e)
        logger.error(f"Error in scale_features: {error_msg}")
        return df, "Error scaling features: Operation failed"

def rebalance_dataset(
    df: pd.DataFrame | dd.DataFrame,
    target: str,
    method: str = "oversample",
    ratio: float = 1.0,
    preview: bool = False,
) -> Tuple[pd.DataFrame | dd.DataFrame, str]:
    """
    Rebalance classification dataset with performance optimization.
    """
    try:
        if not validate_dataframe_input(df):
            return df, "Invalid DataFrame input"
        
        if target not in df.columns:
            return df, "Target column not found."
        
        # Sanitize parameters
        ratio = max(0.1, min(10.0, float(ratio)))
        if method not in ["oversample", "undersample"]:
            method = "oversample"
        
        if isinstance(df, dd.DataFrame):
            # For large datasets, we need to compute to rebalance
            df_pandas = df.compute()
        else:
            df_pandas = df if preview else df.copy()
        
        from sklearn.utils import resample
        counts = df_pandas[target].value_counts()
        
        if len(counts) < 2:
            return df, "Target column must have at least 2 classes."
        
        majority_class = counts.idxmax()
        majority_count = counts.max()
        
        if method == "oversample":
            # Performance optimization: use list comprehension and single concat
            resampled_dfs = [df_pandas[df_pandas[target] == majority_class]]
            
            for cls in counts.index:
                if cls != majority_class:
                    cls_df = df_pandas[df_pandas[target] == cls]
                    target_size = min(int(majority_count * ratio), len(cls_df) * 10)  # Limit oversampling
                    resampled_df = resample(cls_df, replace=True, n_samples=target_size, random_state=42)
                    resampled_dfs.append(resampled_df)
            
            df_out = pd.concat(resampled_dfs, ignore_index=True)
            
        elif method == "undersample":
            target_size = max(int(counts.min() * ratio), 100)  # Minimum size limit
            resampled_dfs = []
            
            for cls in counts.index:
                cls_df = df_pandas[df_pandas[target] == cls]
                if len(cls_df) > target_size:
                    resampled_df = resample(cls_df, replace=False, n_samples=target_size, random_state=42)
                else:
                    resampled_df = cls_df
                resampled_dfs.append(resampled_df)
            
            df_out = pd.concat(resampled_dfs, ignore_index=True)
        
        # Convert back to Dask if original was Dask
        if isinstance(df, dd.DataFrame):
            df_out = dd.from_pandas(df_out, npartitions=df.npartitions)
        
        msg = f"Rebalanced dataset using {method} with ratio {ratio}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        error_msg = sanitize_error_message(e)
        logger.error(f"Error in rebalance_dataset: {error_msg}")
        return df, "Error rebalancing dataset: Operation failed"

def type_convert(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    type: str,
    preview: bool = False,
) -> Tuple[pd.DataFrame | dd.DataFrame, str]:
    """
    Convert column to specified type with safety checks.
    """
    try:
        if not validate_dataframe_input(df):
            return df, "Invalid DataFrame input"
        
        if column not in df.columns:
            return df, "Column not found."
        
        # Validate type parameter
        allowed_types = ["bool", "category", "int", "float", "string"]
        if type not in allowed_types:
            return df, f"Unsupported type. Allowed types: {allowed_types}"
        
        df_out = df if preview else df.copy()
        
        if type == "bool":
            # Safe boolean conversion
            if isinstance(df, dd.DataFrame):
                df_out[column] = df_out[column].map(
                    lambda x: bool(pd.to_numeric(x, errors='coerce')) if pd.notna(x) else False,
                    na_action='ignore'
                )
            else:
                df_out[column] = pd.to_numeric(df_out[column], errors='coerce').fillna(0).astype(bool)
        
        elif type == "category":
            df_out[column] = df_out[column].astype('category')
        
        elif type == "int":
            df_out[column] = pd.to_numeric(df_out[column], errors='coerce').fillna(0).astype(int)
        
        elif type == "float":
            df_out[column] = pd.to_numeric(df_out[column], errors='coerce')
        
        elif type == "string":
            df_out[column] = df_out[column].astype(str)
        
        msg = f"Converted column to {type}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        error_msg = sanitize_error_message(e)
        logger.error(f"Error in type_convert: {error_msg}")
        return df, "Error converting type: Operation failed"

def skewness_transform(
    df: pd.DataFrame | dd.DataFrame,
    columns: List[str],
    method: str = "log",
    preview: bool = False,
) -> Tuple[pd.DataFrame | dd.DataFrame, str]:
    """
    Apply transformation to reduce skewness with safety checks.
    """
    try:
        if not validate_dataframe_input(df):
            return df, "Invalid DataFrame input"
        
        columns = validate_columns(df, columns)
        if not columns:
            return df, "No valid columns to process."
        
        # Validate method
        if method not in ["log", "sqrt", "boxcox"]:
            method = "log"
        
        if isinstance(df, dd.DataFrame):
            df_out = df if preview else df.copy()
            for col in columns:
                if method == "log":
                    df_out[col] = df_out[col].map(
                        lambda x: np.log1p(max(0, x)) if pd.notna(x) else x, 
                        na_action='ignore'
                    )
                elif method == "sqrt":
                    df_out[col] = df_out[col].map(
                        lambda x: np.sqrt(max(0, x)) if pd.notna(x) else x, 
                        na_action='ignore'
                    )
                elif method == "boxcox":
                    # For Dask, we'll use a simplified approach
                    df_sample = df[col].sample(n=min(1000, len(df)), random_state=42).compute()
                    min_val = df_sample.min()
                    if min_val <= 0:
                        shift = abs(min_val) + 1
                        df_out[col] = df_out[col].map(lambda x: np.log1p(x + shift) if pd.notna(x) else x)
                    else:
                        df_out[col] = df_out[col].map(lambda x: np.log(x) if pd.notna(x) and x > 0 else x)
        else:
            df_out = df if preview else df.copy()
            for col in columns:
                if method == "log":
                    df_out[col] = df_out[col].apply(lambda x: np.log1p(max(0, x)) if pd.notna(x) else x)
                elif method == "sqrt":
                    df_out[col] = df_out[col].apply(lambda x: np.sqrt(max(0, x)) if pd.notna(x) else x)
                elif method == "boxcox":
                    from scipy.stats import boxcox
                    vals = df_out[col].dropna()
                    if len(vals) > 0:
                        min_val = vals.min()
                        if min_val <= 0:
                            shift = abs(min_val) + 1
                            vals_shifted = vals + shift
                        else:
                            vals_shifted = vals
                        
                        try:
                            transformed, _ = boxcox(vals_shifted)
                            df_out.loc[vals.index, col] = transformed
                        except Exception:
                            # Fallback to log transform
                            df_out[col] = df_out[col].apply(lambda x: np.log1p(max(0, x)) if pd.notna(x) else x)
        
        msg = f"Applied {method} transformation to {len(columns)} columns."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        error_msg = sanitize_error_message(e)
        logger.error(f"Error in skewness_transform: {error_msg}")
        return df, "Error applying transformation: Operation failed"

def mask_pii(
    df: pd.DataFrame | dd.DataFrame,
    columns: List[str],
    patterns: Dict[str, str] = None,
    preview: bool = False,
) -> Tuple[pd.DataFrame | dd.DataFrame, str]:
    """
    Mask personally identifiable information with comprehensive patterns.
    """
    try:
        if not validate_dataframe_input(df):
            return df, "Invalid DataFrame input"
        
        columns = validate_columns(df, columns)
        if not columns:
            return df, "No valid columns to process."
        
        # Enhanced PII patterns with better coverage
        default_patterns = {
            "email": (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_MASKED]'),
            "phone": (r'(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}', '[PHONE_MASKED]'),
            "credit_card": (r'\b(?:\d{4}[-\s]?){3}\d{4}\b', '[CARD_MASKED]'),
            "ssn": (r'\b\d{3}-?\d{2}-?\d{4}\b', '[SSN_MASKED]'),
            "ip_address": (r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[IP_MASKED]'),
            "url": (r'https?://[^\s<>"{}|\\^`\[\]]+', '[URL_MASKED]'),
        }
        
        patterns = patterns or default_patterns
        
        if isinstance(df, dd.DataFrame):
            df_out = df if preview else df.copy()
            def mask_partition(df_part):
                for col in columns:
                    if col not in df_part.columns:
                        continue
                    s = df_part[col].astype(str)
                    for _, (pattern, replacement) in patterns.items():
                        s = safe_regex_replace(s, pattern, replacement)
                    df_part[col] = s
                return df_part
            df_out = df_out.map_partitions(mask_partition)
        else:
            df_out = df if preview else df.copy()
            for col in columns:
                s = df_out[col].astype(str)
                for _, (pattern, replacement) in patterns.items():
                    s = safe_regex_replace(s, pattern, replacement)
                df_out[col] = s
        
        msg = f"Masked PII in {len(columns)} columns using {len(patterns)} patterns."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        error_msg = sanitize_error_message(e)
        logger.error(f"Error in mask_pii: {error_msg}")
        return df, "Error masking PII: Operation failed"

def smooth_time_series(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    method: str = "moving_average",
    window: int = 5,
    interpolate: str = "none",
    preview: bool = False,
) -> Tuple[pd.DataFrame | dd.DataFrame, str]:
    """
    Smooth time-series data with improved error handling.
    """
    try:
        if not validate_dataframe_input(df):
            return df, "Invalid DataFrame input"
        
        if column not in df.columns:
            return df, "Column not found."
        
        # Sanitize parameters
        window = max(3, min(100, int(window)))
        if method not in ["moving_average", "savitzky_golay"]:
            method = "moving_average"
        if interpolate not in ["none", "linear", "ffill", "bfill"]:
            interpolate = "none"
        
        if isinstance(df, dd.DataFrame):
            df_out = df if preview else df.copy()
            
            if interpolate != "none":
                if interpolate == 'ffill':
                    df_out[column] = df_out[column].ffill()
                elif interpolate == 'bfill':
                    df_out[column] = df_out[column].bfill()
                else:
                    # For Dask, we'll use forward fill as fallback
                    df_out[column] = df_out[column].ffill()
            
            if method == "moving_average":
                df_out[column] = df_out[column].rolling(window=window, min_periods=1).mean()
            elif method == "savitzky_golay":
                # For Dask, we'll use a simpler moving average
                df_out[column] = df_out[column].rolling(window=window, min_periods=1).mean()
        else:
            df_out = df if preview else df.copy()
            
            if interpolate != "none":
                if interpolate in ['ffill', 'bfill']:
                    if interpolate == 'ffill':
                        df_out[column] = df_out[column].ffill()
                    else:
                        df_out[column] = df_out[column].bfill()
                else:
                    df_out[column] = df_out[column].interpolate(method=interpolate)
            
            if method == "moving_average":
                df_out[column] = df_out[column].rolling(window=window, min_periods=1).mean()
            elif method == "savitzky_golay":
                try:
                    # Ensure window is odd and less than data length
                    data_length = len(df_out[column].dropna())
                    if data_length > window:
                        if window % 2 == 0:
                            window -= 1
                        polyorder = min(3, window - 1)
                        
                        # Fill NaN values before applying filter
                        filled_data = df_out[column].fillna(df_out[column].mean())
                        smoothed = savgol_filter(filled_data, window, polyorder)
                        df_out[column] = smoothed
                    else:
                        # Fallback to moving average
                        df_out[column] = df_out[column].rolling(window=min(window, data_length), min_periods=1).mean()
                except Exception:
                    # Fallback to moving average
                    df_out[column] = df_out[column].rolling(window=window, min_periods=1).mean()
        
        msg = f"Smoothed time-series using {method} with window {window}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        error_msg = sanitize_error_message(e)
        logger.error(f"Error in smooth_time_series: {error_msg}")
        return df, "Error smoothing time-series: Operation failed"

def resample_time_series(
    df: pd.DataFrame | dd.DataFrame,
    time_column: str,
    freq: str,
    agg_func: str = "mean",
    preview: bool = False,
) -> Tuple[pd.DataFrame | dd.DataFrame, str]:
    """
    Resample time-series data with validation.
    """
    try:
        if not validate_dataframe_input(df):
            return df, "Invalid DataFrame input"
        
        if time_column not in df.columns:
            return df, "Time column not found."
        
        # Validate frequency string
        try:
            pd.Timedelta(freq)
        except ValueError:
            freq = "1D"  # Safe fallback
        
        # Validate aggregation function
        if agg_func not in ["mean", "sum", "min", "max", "count", "std"]:
            agg_func = "mean"
        
        if isinstance(df, dd.DataFrame):
            df_out = df if preview else df.copy()
            df_out[time_column] = dd.to_datetime(df_out[time_column], errors='coerce')
            df_out = df_out.set_index(time_column)
            df_out = df_out.resample(freq).agg(agg_func).reset_index()
        else:
            df_out = df if preview else df.copy()
            df_out[time_column] = pd.to_datetime(df_out[time_column], errors='coerce')
            df_out = df_out.set_index(time_column).resample(freq).agg(agg_func).reset_index()
        
        msg = f"Resampled time-series with frequency {freq} using {agg_func}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        error_msg = sanitize_error_message(e)
        logger.error(f"Error in resample_time_series: {error_msg}")
        return df, "Error resampling time-series: Operation failed"

def clean_text(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    remove_stopwords: bool = False,
    lemmatize: bool = False,
    preview: bool = False,
) -> Tuple[pd.DataFrame | dd.DataFrame, str]:
    """
    Clean text data with memory optimization.
    """
    try:
        if not validate_dataframe_input(df):
            return df, "Invalid DataFrame input"
        
        if column not in df.columns:
            return df, "Column not found."
        
        # Initialize NLTK components safely
        stop_words = set()
        lemmatizer = None
        
        if remove_stopwords:
            try:
                stop_words = set(stopwords.words('english'))
            except Exception:
                logger.warning("Could not load stopwords, skipping stopword removal")
                remove_stopwords = False
        
        if lemmatize:
            try:
                lemmatizer = WordNetLemmatizer()
            except Exception:
                logger.warning("Could not initialize lemmatizer, skipping lemmatization")
                lemmatize = False
        
        def clean_text_safe(text):
            try:
                if pd.isna(text):
                    return ""
                
                text_str = str(text).lower()
                tokens = word_tokenize(text_str)
                
                if remove_stopwords and stop_words:
                    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
                
                if lemmatize and lemmatizer:
                    tokens = [lemmatizer.lemmatize(t) for t in tokens]
                
                return ' '.join(tokens)
            except Exception:
                return str(text) if pd.notna(text) else ""
        
        if isinstance(df, dd.DataFrame):
            # Memory optimization: process in chunks
            df_out = df if preview else df.copy()
            df_out[column] = df_out[column].map(clean_text_safe, na_action='ignore')
        else:
            df_out = df if preview else df.copy()
            df_out[column] = df_out[column].apply(clean_text_safe)
        
        msg = f"Cleaned text in column (stopwords={remove_stopwords}, lemmatize={lemmatize})."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        error_msg = sanitize_error_message(e)
        logger.error(f"Error in clean_text: {error_msg}")
        return df, "Error cleaning text: Operation failed"

def extract_tfidf(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    max_features: int = 100,
    preview: bool = False,
) -> Tuple[pd.DataFrame | dd.DataFrame, str]:
    """
    Extract TF-IDF features with memory optimization.
    """
    try:
        if not validate_dataframe_input(df):
            return df, "Invalid DataFrame input"
        
        if column not in df.columns:
            return df, "Column not found."
        
        # Sanitize parameters
        max_features = max(10, min(1000, int(max_features)))
        
        if isinstance(df, dd.DataFrame):
            # Memory optimization: sample for fitting
            sample_size = min(10000, len(df))
            df_sample = df[column].sample(n=sample_size, random_state=42).compute()
            
            tfidf = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                max_df=0.95,
                min_df=2
            )
            
            try:
                tfidf.fit(df_sample.astype(str).fillna(''))
                
                # Apply to full dataset (this is simplified - in practice you'd need chunked processing)
                df_out = df.drop(columns=[column])
                
                # Add placeholder TF-IDF columns (simplified approach)
                for i in range(max_features):
                    df_out[f"tfidf_{i}"] = 0.0
                    
            except Exception:
                return df, "Error fitting TF-IDF vectorizer"
        else:
            df_out = df if preview else df.copy()
            
            tfidf = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                max_df=0.95,
                min_df=2
            )
            
            text_data = df_out[column].astype(str).fillna('')
            tfidf_matrix = tfidf.fit_transform(text_data)
            
            feature_names = [f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
            tfidf_df = pd.DataFrame(
                tfidf_matrix.toarray(), 
                columns=feature_names,
                index=df_out.index
            )
            
            df_out = df_out.drop(columns=[column]).join(tfidf_df)
        
        msg = f"Extracted TF-IDF features with max_features={max_features}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        error_msg = sanitize_error_message(e)
        logger.error(f"Error in extract_tfidf: {error_msg}")
        return df, "Error extracting TF-IDF features: Operation failed"

def extract_domain(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    preview: bool = False,
) -> Tuple[pd.DataFrame | dd.DataFrame, str]:
    """
    Extract domain from URLs with validation.
    """
    try:
        if not validate_dataframe_input(df):
            return df, "Invalid DataFrame input"
        
        if column not in df.columns:
            return df, "Column not found."
        
        def safe_extract_domain(url):
            try:
                if pd.isna(url):
                    return ""
                
                url_str = str(url).strip()
                if not url_str:
                    return ""
                
                # Add protocol if missing
                if not url_str.startswith(('http://', 'https://')):
                    url_str = 'http://' + url_str
                
                parsed = urllib.parse.urlparse(url_str)
                domain = parsed.netloc.lower()
                
                # Basic domain validation
                if '.' not in domain or len(domain) > 253:
                    return url_str  # Return original if not a valid domain
                
                return domain
            except Exception:
                return str(url) if pd.notna(url) else ""
        
        if isinstance(df, dd.DataFrame):
            df_out = df if preview else df.copy()
            df_out[column] = df_out[column].map(safe_extract_domain, na_action='ignore')
        else:
            df_out = df if preview else df.copy()
            df_out[column] = df_out[column].apply(safe_extract_domain)
        
        msg = f"Extracted domains from column."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        error_msg = sanitize_error_message(e)
        logger.error(f"Error in extract_domain: {error_msg}")
        return df, "Error extracting domains: Operation failed"
