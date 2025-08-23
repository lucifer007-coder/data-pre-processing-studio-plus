import logging
import time
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Any, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
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
                msg = f"Imputed {len(valid_cols)} columns with mode: {', '.join(imputed_cols)}"
        elif strategy == "constant":
            for c in valid_cols:
                imputed_count = df_out[c].isna().sum()
                df_out[c] = df_out[c].fillna(constant_value)
                imputed_cols.append(f"{c} ({imputed_count} values with constant={constant_value})")
            msg = f"Imputed {len(valid_cols)} columns with constant value: {', '.join(imputed_cols)}"
        elif strategy in ("ffill", "bfill"):
            for c in valid_cols:
                imputed_count = df_out[c].isna().sum()
                df_out[c] = df_out[c].ffill() if strategy == "ffill" else df_out[c].bfill()
                imputed_cols.append(f"{c} ({imputed_count} values with {strategy})")
            msg = f"Imputed {len(valid_cols)} columns with {strategy}: {', '.join(imputed_cols)}"
        else:
            return df, f"Unsupported strategy: {strategy}."
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
    preview: bool = False
) -> Tuple[pd.DataFrame, str]:
    """
    Drop rows or columns with missing values based on threshold or specific columns.
    """
    try:
        df_out = df if preview else df.copy()
        if axis == "rows":
            if threshold:
                missing_ratio = df_out.isna().mean(axis=1)
                drop_count = len(df_out[missing_ratio > threshold])
                df_out = df_out[missing_ratio <= threshold]
                msg = f"Dropped {drop_count} rows with missing ratio > {threshold}."
            else:
                columns = columns or df_out.columns.tolist()
                drop_count = len(df_out) - len(df_out.dropna(subset=columns))
                df_out = df_out.dropna(subset=columns)
                msg = f"Dropped {drop_count} rows with missing values in {', '.join(columns)}."
        else:  # axis == "columns"
            if threshold:
                missing_ratio = df_out.isna().mean()
                cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
                df_out = df_out.drop(columns=cols_to_drop)
                msg = f"Dropped {len(cols_to_drop)} columns with missing ratio > {threshold}: {', '.join(cols_to_drop)}."
            else:
                columns = columns or df_out.columns.tolist()
                df_out = df_out.drop(columns=columns)
                msg = f"Dropped {len(columns)} columns: {', '.join(columns)}."
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
    collapse_spaces: bool = True,
    remove_special: bool = False,
    preview: bool = False
) -> Tuple[pd.DataFrame, str]:
    """
    Normalize text columns.
    """
    try:
        valid_cols = [c for c in columns if c in df.columns]
        if not valid_cols:
            return df, "No valid columns selected for text normalization."
        df_out = df if preview else df.copy()
        for col in valid_cols:
            series = df_out[col].astype(str)
            if lowercase:
                series = series.str.lower()
            if trim:
                series = series.str.strip()
            if collapse_spaces:
                series = series.str.replace(r'\s+', ' ', regex=True)
            if remove_special:
                series = series.str.replace(r'[^\w\s]', '', regex=True)
            df_out[col] = series
        msg = f"Normalized text in {', '.join(valid_cols)} with options: lowercase={lowercase}, trim={trim}, collapse_spaces={collapse_spaces}, remove_special={remove_special}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in normalize_text: {e}")
        return df, f"Error normalizing text: {e}"

def standardize_dates(
    df: pd.DataFrame,
    columns: List[str],
    output_format: str = "YYYY-MM-DD",
    preview: bool = False
) -> Tuple[pd.DataFrame, str]:
    """
    Standardize date columns to a specified format.
    """
    try:
        valid_cols = [c for c in columns if c in df.columns]
        if not valid_cols:
            return df, "No valid columns selected for date standardization."
        df_out = df if preview else df.copy()
        for col in valid_cols:
            df_out[col] = pd.to_datetime(df_out[col], errors='coerce').dt.strftime(output_format)
        msg = f"Standardized dates in {', '.join(valid_cols)} to format {output_format}."
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
    Convert units in a numeric column by multiplying by a factor.
    """
    try:
        if column not in df.columns:
            return df, f"Column {column} not found."
        if not pd.api.types.is_numeric_dtype(df[column]):
            return df, f"Column {column} is not numeric."
        df_out = df if preview else df.copy()
        target_col = new_name or column
        df_out[target_col] = df_out[column] * factor
        if new_name and new_name != column:
            df_out = df_out.drop(columns=[column])
        msg = f"Converted units in {column} by factor {factor} {'to new column ' + new_name if new_name else ''}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in unit_convert: {e}")
        return df, f"Error in unit conversion: {e}"

def handle_outliers(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "iqr",
    factor: float = 1.5,
    preview: bool = False
) -> Tuple[pd.DataFrame, str]:
    """
    Handle outliers using IQR or Z-score method.
    """
    try:
        valid_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if not valid_cols:
            return df, "No valid numeric columns selected for outlier handling."
        df_out = df if preview else df.copy()
        for col in valid_cols:
            if method == "iqr":
                q1, q3 = df_out[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                lower, upper = q1 - factor * iqr, q3 + factor * iqr
                outliers = (df_out[col] < lower) | (df_out[col] > upper)
                df_out.loc[outliers, col] = df_out[col].clip(lower, upper)
                msg = f"Handled outliers in {', '.join(valid_cols)} using IQR with factor {factor}."
            elif method == "zscore":
                z_scores = np.abs((df_out[col] - df_out[col].mean()) / df_out[col].std())
                outliers = z_scores > factor
                df_out.loc[outliers, col] = df_out[col].clip(df_out[col].mean() - factor * df_out[col].std(), df_out[col].mean() + factor * df_out[col].std())
                msg = f"Handled outliers in {', '.join(valid_cols)} using Z-score with threshold {factor}."
            else:
                return df, f"Unsupported outlier method: {method}."
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
        subset = subset or df_out.columns.tolist()
        valid_cols = [c for c in subset if c in df_out.columns]
        if not valid_cols:
            return df, "No valid columns selected for duplicate removal."
        drop_count = len(df_out) - len(df_out.drop_duplicates(subset=valid_cols, keep=keep))
        df_out = df_out.drop_duplicates(subset=valid_cols, keep=keep)
        msg = f"Removed {drop_count} duplicate rows based on {', '.join(valid_cols)} (keep={keep})."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in remove_duplicates: {e}")
        return df, f"Error removing duplicates: {e}"

def encode_categorical(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "onehot",
    max_categories: Optional[int] = None,
    ordinal_mappings: Optional[Dict[str, Dict]] = None,
    preview: bool = False
) -> Tuple[pd.DataFrame, str]:
    """
    Encode categorical columns using one-hot, label, or ordinal encoding.
    """
    try:
        valid_cols = [c for c in columns if c in df.columns]
        if not valid_cols:
            return df, "No valid columns selected for encoding."
        df_out = df if preview else df.copy()
        if method == "onehot":
            for col in valid_cols:
                if max_categories:
                    top_cats = df_out[col].value_counts().nlargest(max_categories).index
                    df_out[col] = df_out[col].where(df_out[col].isin(top_cats), "Other")
                dummies = pd.get_dummies(df_out[col], prefix=col, drop_first=True)
                df_out = pd.concat([df_out.drop(columns=[col]), dummies], axis=1)
            msg = f"One-hot encoded {', '.join(valid_cols)} {'with max_categories=' + str(max_categories) if max_categories else ''}."
        elif method == "label":
            for col in valid_cols:
                le = LabelEncoder()
                df_out[col] = le.fit_transform(df_out[col].astype(str))
            msg = f"Label encoded {', '.join(valid_cols)}."
        elif method == "ordinal":
            if not ordinal_mappings:
                return df, "Ordinal encoding requires mappings."
            for col in valid_cols:
                if col in ordinal_mappings:
                    df_out[col] = df_out[col].map(ordinal_mappings[col]).fillna(df_out[col])
            msg = f"Ordinal encoded {', '.join(valid_cols)}."
        else:
            return df, f"Unsupported encoding method: {method}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in encode_categorical: {e}")
        return df, f"Error encoding categorical data: {e}"

def scale_features(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "standard",
    keep_original: bool = False,
    preview: bool = False
) -> Tuple[pd.DataFrame, str]:
    """
    Scale numeric features using standard, minmax, or robust scaling.
    """
    try:
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
            return df, f"Unsupported scaling method: {method}."
        scaled = scaler.fit_transform(df_out[valid_cols])
        new_cols = [f"{col}_scaled" for col in valid_cols] if keep_original else valid_cols
        df_out[new_cols] = pd.DataFrame(scaled, index=df_out.index, columns=new_cols)
        if not keep_original:
            df_out = df_out.drop(columns=valid_cols)
        msg = f"Scaled {', '.join(valid_cols)} using {method} scaling {'(kept original columns)' if keep_original else ''}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in scale_features: {e}")
        return df, f"Error scaling features: {e}"

def rebalance_dataset(
    df: pd.DataFrame,
    target: str,
    method: str = "oversample",
    ratio: float = 1.0,
    preview: bool = False
) -> Tuple[pd.DataFrame, str]:
    """
    Rebalance a dataset for classification using oversampling or undersampling.
    """
    try:
        if target not in df.columns:
            return df, f"Target column {target} not found."
        df_out = df if preview else df.copy()
        value_counts = df_out[target].value_counts()
        majority_count = value_counts.max()
        minority_count = value_counts.min()
        if method == "oversample":
            target_count = int(majority_count * ratio)
            for cls in value_counts.index:
                cls_count = value_counts[cls]
                if cls_count < target_count:
                    samples = df_out[df_out[target] == cls].sample(n=target_count - cls_count, replace=True, random_state=42)
                    df_out = pd.concat([df_out, samples])
            msg = f"Oversampled {target} to achieve ratio {ratio}."
        elif method == "undersample":
            target_count = int(minority_count * ratio)
            df_out = pd.concat([
                df_out[df_out[target] == cls].sample(n=min(target_count, value_counts[cls]), random_state=42)
                for cls in value_counts.index
            ])
            msg = f"Undersampled {target} to achieve ratio {ratio}."
        else:
            return df, f"Unsupported rebalancing method: {method}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in rebalance_dataset: {e}")
        return df, f"Error rebalancing dataset: {e}"

def type_convert(
    df: pd.DataFrame,
    column: str,
    type: str,
    preview: bool = False
) -> Tuple[pd.DataFrame, str]:
    """
    Convert a column to a specified data type.
    """
    try:
        if column not in df.columns:
            return df, f"Column {column} not found."
        df_out = df if preview else df.copy()
        if type == "int":
            df_out[column] = pd.to_numeric(df_out[column], errors='coerce').astype('Int64')
        elif type == "float":
            df_out[column] = pd.to_numeric(df_out[column], errors='coerce').astype(float)
        elif type == "str":
            df_out[column] = df_out[column].astype(str)
        elif type == "datetime":
            df_out[column] = pd.to_datetime(df_out[column], errors='coerce')
        else:
            return df, f"Unsupported type: {type}."
        msg = f"Converted {column} to {type}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in type_convert: {e}")
        return df, f"Error converting type: {e}"

def skewness_transform(
    df: pd.DataFrame,
    column: str,
    transform: str = "log",
    preview: bool = False
) -> Tuple[pd.DataFrame, str]:
    """
    Apply a transformation to reduce skewness in a numeric column.
    """
    try:
        if column not in df.columns:
            return df, f"Column {column} not found."
        if not pd.api.types.is_numeric_dtype(df[column]):
            return df, f"Column {column} is not numeric."
        df_out = df if preview else df.copy()
        if transform == "log":
            if df_out[column].min() <= 0:
                return df, f"Cannot apply log transform to {column} due to non-positive values."
            df_out[column] = np.log1p(df_out[column])
            msg = f"Applied log transform to {column}."
        elif transform == "sqrt":
            if df_out[column].min() < 0:
                return df, f"Cannot apply sqrt transform to {column} due to negative values."
            df_out[column] = np.sqrt(df_out[column])
            msg = f"Applied sqrt transform to {column}."
        else:
            return df, f"Unsupported transform: {transform}."
        logger.info(msg)
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

def smooth_time_series(
    df: pd.DataFrame,
    column: str,
    window: int = 5,
    method: str = "moving_average",
    interpolate: str = "linear",
    preview: bool = False
) -> Tuple[pd.DataFrame, str]:
    """
    Smooth a time-series column using a specified method with optional interpolation.
    Args:
        df: Input DataFrame.
        column: Time-series column (numeric).
        window: Window size for smoothing.
        method: Smoothing method ("moving_average" or "savitzky_golay").
        interpolate: Interpolation method for missing values ("linear", "ffill", "bfill", None).
        preview: If True, return unchanged df for preview.
    Returns:
        (transformed_df, message).
    """
    try:
        if column not in df.columns:
            return df, f"Column {column} not found."
        # Validate numeric column
        try:
            df[column] = pd.to_numeric(df[column], errors='coerce')
        except Exception as e:
            return df, f"Column {column} contains non-numeric values: {e}"
        if df[column].isna().all():
            return df, f"Column {column} contains only missing values."
        df_out = df if preview else df.copy()
        if interpolate and df_out[column].isna().any():
            df_out[column] = df_out[column].interpolate(method=interpolate, limit_direction='both')
        if method == "moving_average":
            df_out[column] = df_out[column].rolling(window=window, min_periods=1).mean()
            msg = f"Applied moving average smoothing to {column} with window {window} and interpolation {interpolate}."
        elif method == "savitzky_golay":
            if window % 2 == 0:
                return df, f"Savitzky-Golay requires an odd window size, got {window}."
            if df_out[column].isna().any():
                return df, f"Savitzky-Golay cannot handle missing values in {column}."
            df_out[column] = savgol_filter(df_out[column], window_length=window, polyorder=2)
            msg = f"Applied Savitzky-Golay smoothing to {column} with window {window}."
        else:
            return df, f"Unsupported smoothing method: {method}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in smooth_time_series: {e}")
        return df, f"Error smoothing time series: {e}"

def resample_time_series(
    df: pd.DataFrame,
    time_column: str,
    freq: str = "1H",
    agg_func: str = "mean",
    preview: bool = False
) -> Tuple[pd.DataFrame, str]:
    """
    Resample a time-series DataFrame to a specified frequency.
    Args:
        df: Input DataFrame with a datetime index or column.
        time_column: Datetime column to set as index.
        freq: Resampling frequency (e.g., '1H', '1D').
        agg_func: Aggregation function ("mean", "sum", "last").
        preview: If True, return unchanged df for preview.
    Returns:
        (transformed_df, message).
    """
    try:
        if time_column not in df.columns:
            return df, f"Time column {time_column} not found."
        df_out = df if preview else df.copy()
        # Convert to datetime if not already
        df_out[time_column] = pd.to_datetime(df_out[time_column], errors='coerce')
        if df_out[time_column].isna().all():
            return df, f"Column {time_column} contains no valid datetime values."
        df_out = df_out.set_index(time_column)
        try:
            df_out = df_out.resample(freq).agg(agg_func).reset_index()
        except ValueError as e:
            return df, f"Invalid frequency {freq}: {e}"
        msg = f"Resampled time series on {time_column} to {freq} using {agg_func}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in resample_time_series: {e}")
        return df, f"Error resampling time series: {e}"

def clean_text(
    df: pd.DataFrame,
    column: str,
    remove_stopwords: bool = False,
    lemmatize: bool = False,
    preview: bool = False
) -> Tuple[pd.DataFrame, str]:
    """
    Clean text data by removing stopwords and optionally lemmatizing.
    """
    try:
        if column not in df.columns:
            return df, f"Column {column} not found."
        df_out = df if preview else df.copy()
        # Check for empty or whitespace-only text
        if df_out[column].astype(str).str.strip().eq('').all():
            return df, f"Column {column} contains only empty or whitespace text."
        df_out[column] = df_out[column].astype(str).str.lower().str.replace(r'[^\w\s]', '', regex=True).str.strip()
        if remove_stopwords:
            stop_words = set(stopwords.words('english'))
            df_out[column] = df_out[column].apply(
                lambda x: ' '.join(word for word in word_tokenize(x) if word not in stop_words)
            )
        if lemmatize:
            lemmatizer = WordNetLemmatizer()
            df_out[column] = df_out[column].apply(
                lambda x: ' '.join(lemmatizer.lemmatize(word) for word in word_tokenize(x))
            )
        msg = f"Cleaned text in {column} {'with stopword removal' if remove_stopwords else ''}{' and lemmatization' if lemmatize else ''}."
        logger.info(msg)
        return df_out, msg
    except LookupError as e:
        logger.error(f"NLTK resource error in clean_text: {e}")
        st.error(
            f"NLTK resource error: {e}\n"
            "Please run the following in Python:\n"
            "import nltk\n"
            "nltk.download('punkt')\n"
            "nltk.download('punkt_tab')\n"
            "nltk.download('stopwords')\n"
            "nltk.download('wordnet')"
        )
        return df, f"Error cleaning text: {e}"
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
        # Check for empty or whitespace-only text
        if df[column].astype(str).str.strip().eq('').all():
            return df, f"Column {column} contains only empty or whitespace text."
        df_out = df if preview else df.copy()
        vectorizer = TfidfVectorizer(max_features=max_features)
        tfidf_matrix = vectorizer.fit_transform(df_out[column].astype(str))
        feature_names = vectorizer.get_feature_names_out()
        # Adjust max_features to actual number of features to avoid shape mismatch
        actual_features = min(max_features, len(feature_names))
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray()[:, :actual_features],
            columns=[f"{column}_tfidf_{i}" for i in range(actual_features)],
            index=df_out.index
        )
        df_out = pd.concat([df_out, tfidf_df], axis=1).drop(columns=[column])
        msg = f"Extracted {actual_features} TF-IDF features from {column}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in extract_tfidf: {e}")
        return df, f"Error extracting TF-IDF: {e}"

def resize_image(
    df: pd.DataFrame,
    column: str,
    width: int = 224,
    height: int = 224,
    preview: bool = False
) -> Tuple[pd.DataFrame, str]:
    """
    Resize images referenced in a DataFrame column.
    Args:
        df: DataFrame with image file paths or base64 strings.
        column: Column containing image paths/URLs.
        width, height: Target dimensions.
        preview: If True, return unchanged df for preview.
    Returns:
        (transformed_df, message).
    """
    try:
        if column not in df.columns:
            return df, f"Column {column} not found."
        # Validate image data
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
        # Validate image data
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
