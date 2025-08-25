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
import regex as re
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
        if isinstance(df, dd.DataFrame):
            df_out = df if preview else df.copy()
            for col in columns:
                if col not in df_out.columns:
                    return df, f"Column {col} not found."
                if strategy == 'mean':
                    df_out[col] = df_out[col].fillna(df_out[col].mean().compute())
                elif strategy == 'median':
                    df_out[col] = df_out[col].fillna(df_out[col].median().compute())
                elif strategy == 'mode':
                    mode_val = df_out[col].mode().compute().iloc[0] if not df_out[col].mode().compute().empty else None
                    df_out[col] = df_out[col].fillna(mode_val)
                elif strategy == 'constant':
                    df_out[col] = df_out[col].fillna(constant_value)
                elif strategy in ['ffill', 'bfill']:
                    df_out[col] = df_out[col].fillna(method=strategy)
                else:  # knn, random_forest
                    df_pandas = df.compute()
                    if strategy == 'knn':
                        imputer = KNNImputer(n_neighbors=n_neighbors)
                        df_pandas[columns] = imputer.fit_transform(df_pandas[columns])
                    elif strategy == 'random_forest':
                        for col in columns:
                            mask = df_pandas[col].isna()
                            if mask.any():
                                X = df_pandas.drop(columns=col)
                                y = df_pandas[col]
                                X_train = X[~mask]
                                y_train = y[~mask]
                                X_test = X[mask]
                                model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                                model.fit(X_train, y_train)
                                df_pandas.loc[mask, col] = model.predict(X_test)
                    df_out = dd.from_pandas(df_pandas, npartitions=df.npartitions)
            msg = f"Imputed missing values in {columns} using {strategy}."
        else:
            df_out = df if preview else df.copy()
            for col in columns:
                if col not in df_out.columns:
                    return df, f"Column {col} not found."
                if strategy == 'mean':
                    df_out[col] = df_out[col].fillna(df_out[col].mean())
                elif strategy == 'median':
                    df_out[col] = df_out[col].fillna(df_out[col].median())
                elif strategy == 'mode':
                    df_out[col] = df_out[col].fillna(df_out[col].mode().iloc[0])
                elif strategy == 'constant':
                    df_out[col] = df_out[col].fillna(constant_value)
                elif strategy in ['ffill', 'bfill']:
                    df_out[col] = df_out[col].fillna(method=strategy)
                elif strategy == 'knn':
                    imputer = KNNImputer(n_neighbors=n_neighbors)
                    df_out[columns] = imputer.fit_transform(df_out[columns])
                elif strategy == 'random_forest':
                    for col in columns:
                        mask = df_out[col].isna()
                        if mask.any():
                            X = df_out.drop(columns=col)
                            y = df_out[col]
                            X_train = X[~mask]
                            y_train = y[~mask]
                            X_test = X[mask]
                            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                            model.fit(X_train, y_train)
                            df_out.loc[mask, col] = model.predict(X_test)
            msg = f"Imputed missing values in {columns} using {strategy}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in impute_missing: {e}")
        return df, f"Error imputing missing values: {e}"

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
                    msg = f"Dropped columns {columns}."
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
                    msg = f"Dropped columns {columns}."
                else:
                    missing_ratio = df_out.isna().mean()
                    cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
                    df_out = df_out.drop(columns=cols_to_drop)
                    msg = f"Dropped {len(cols_to_drop)} columns with missing ratio > {threshold}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in drop_missing: {e}")
        return df, f"Error dropping missing values: {e}"

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
    Normalize text columns.
    """
    try:
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
                        s = s.str.replace(r'\s+', ' ', regex=True)
                    if remove_special:
                        s = s.str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
                    df_part[col] = s
                return df_part
            df_out = df_out.map_partitions(normalize_partition)
            msg = f"Normalized text in {columns} (lower={lower}, trim={trim}, collapse={collapse}, remove_special={remove_special})."
        else:
            df_out = df if preview else df.copy()
            for col in columns:
                if col not in df_out.columns:
                    return df, f"Column {col} not found."
                s = df_out[col].astype(str)
                if lower:
                    s = s.str.lower()
                if trim:
                    s = s.str.strip()
                if collapse:
                    s = s.str.replace(r'\s+', ' ', regex=True)
                if remove_special:
                    s = s.str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
                df_out[col] = s
            msg = f"Normalized text in {columns} (lower={lower}, trim={trim}, collapse={collapse}, remove_special={remove_special})."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in normalize_text: {e}")
        return df, f"Error normalizing text: {e}"

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
        if isinstance(df, dd.DataFrame):
            df_out = df if preview else df.copy()
            def standardize_partition(df_part):
                for col in columns:
                    if col not in df_part.columns:
                        continue
                    df_part[col] = pd.to_datetime(df_part[col], errors='coerce').dt.strftime(format)
                return df_part
            df_out = df_out.map_partitions(standardize_partition)
            msg = f"Standardized dates in {columns} to format {format}."
        else:
            df_out = df if preview else df.copy()
            for col in columns:
                if col not in df_out.columns:
                    return df, f"Column {col} not found."
                df_out[col] = pd.to_datetime(df_out[col], errors='coerce').dt.strftime(format)
            msg = f"Standardized dates in {columns} to format {format}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in standardize_dates: {e}")
        return df, f"Error standardizing dates: {e}"

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
        if column not in df.columns:
            return df, f"Column {column} not found."
        if isinstance(df, dd.DataFrame):
            df_out = df if preview else df.copy()
            df_out[column] = df_out[column] * factor
            msg = f"Converted units in {column} by multiplying with {factor}."
        else:
            df_out = df if preview else df.copy()
            df_out[column] = df_out[column] * factor
            msg = f"Converted units in {column} by multiplying with {factor}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in unit_convert: {e}")
        return df, f"Error converting units: {e}"

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
        if isinstance(df, dd.DataFrame):
            df_out = df if preview else df.copy()
            for col in columns:
                if col not in df_out.columns:
                    continue
                if method == "iqr":
                    q1 = df_out[col].quantile(0.25).compute()
                    q3 = df_out[col].quantile(0.75).compute()
                    iqr = q3 - q1
                    lower_bound = q1 - factor * iqr
                    upper_bound = q3 + factor * iqr
                    df_out[col] = df_out[col].clip(lower=lower_bound, upper=upper_bound)
                elif method == "zscore":
                    mean = df_out[col].mean().compute()
                    std = df_out[col].std().compute()
                    df_out[col] = df_out[col].where((df_out[col] - mean).abs() <= factor * std, mean)
            msg = f"Handled outliers in {columns} using {method} with factor {factor}."
        else:
            df_out = df if preview else df.copy()
            for col in columns:
                if col not in df_out.columns:
                    return df, f"Column {col} not found."
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
                    df_out[col] = df_out[col].where((df_out[col] - mean).abs() <= factor * std, mean)
            msg = f"Handled outliers in {columns} using {method} with factor {factor}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in handle_outliers: {e}")
        return df, f"Error handling outliers: {e}"

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
        if isinstance(df, dd.DataFrame):
            df_out = df if preview else df.copy()
            count = df_out.duplicated(subset=subset).sum().compute()
            df_out = df_out.drop_duplicates(subset=subset, keep=keep)
            msg = f"Removed {count} duplicate rows."
        else:
            df_out = df if preview else df.copy()
            count = df_out.duplicated(subset=subset).sum()
            df_out = df_out.drop_duplicates(subset=subset, keep=keep)
            msg = f"Removed {count} duplicate rows."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in remove_duplicates: {e}")
        return df, f"Error removing duplicates: {e}"

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
    Encode categorical columns.
    """
    try:
        if isinstance(df, dd.DataFrame):
            df_pandas = df.compute()  # Scikit-learn requires Pandas
            if method == "onehot":
                encoded = pd.get_dummies(df_pandas[columns], prefix=columns)
                df_out = df_pandas.drop(columns=columns).join(encoded)
            elif method == "label":
                for col in columns:
                    le = LabelEncoder()
                    df_pandas[col] = le.fit_transform(df_pandas[col].astype(str))
                df_out = df_pandas
            elif method == "ordinal" and ordinal_mappings:
                for col in columns:
                    if col in ordinal_mappings:
                        df_pandas[col] = df_pandas[col].map(ordinal_mappings[col]).fillna(df_pandas[col])
                df_out = df_pandas
            elif method == "target_encode" and target_column:
                for col in columns:
                    means = df_pandas.groupby(col)[target_column].mean()
                    df_pandas[col] = df_pandas[col].map(means)
                df_out = df_pandas
            elif method == "frequency_encode":
                for col in columns:
                    counts = df_pandas[col].value_counts()
                    df_pandas[col] = df_pandas[col].map(counts)
                df_out = df_pandas
            elif method == "hashing_encode":
                from sklearn.feature_extraction import FeatureHasher
                hasher = FeatureHasher(n_features=n_components, input_type='string')
                for col in columns:
                    hashed_features = hasher.fit_transform(df_pandas[col].astype(str))
                    hashed_df = pd.DataFrame(hashed_features.toarray(), columns=[f"{col}_hash_{i}" for i in range(n_components)])
                    df_pandas = df_pandas.drop(columns=col).join(hashed_df)
                df_out = df_pandas
            else:
                df_out = df_pandas
            df_out = dd.from_pandas(df_out, npartitions=df.npartitions)
            msg = f"Encoded {columns} using {method}."
        else:
            df_out = df if preview else df.copy()
            if method == "onehot":
                encoded = pd.get_dummies(df_out[columns], prefix=columns)
                df_out = df_out.drop(columns=columns).join(encoded)
            elif method == "label":
                for col in columns:
                    le = LabelEncoder()
                    df_out[col] = le.fit_transform(df_out[col].astype(str))
            elif method == "ordinal" and ordinal_mappings:
                for col in columns:
                    if col in ordinal_mappings:
                        df_out[col] = df_out[col].map(ordinal_mappings[col]).fillna(df_out[col])
            elif method == "target_encode" and target_column:
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
                    hashed_df = pd.DataFrame(hashed_features.toarray(), columns=[f"{col}_hash_{i}" for i in range(n_components)])
                    df_out = df_out.drop(columns=col).join(hashed_df)
            msg = f"Encoded {columns} using {method}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in encode_categorical: {e}")
        return df, f"Error encoding categorical data: {e}"

def scale_features(
    df: pd.DataFrame | dd.DataFrame,
    columns: List[str],
    method: str = "standard",
    keep_original: bool = False,
    preview: bool = False,
) -> Tuple[pd.DataFrame | dd.DataFrame, str]:
    """
    Scale numeric features.
    """
    try:
        if isinstance(df, dd.DataFrame):
            df_pandas = df[columns].compute()
            scaler = {'standard': StandardScaler(), 'minmax': MinMaxScaler(), 'robust': RobustScaler()}[method]
            scaled = scaler.fit_transform(df_pandas)
            scaled_df = pd.DataFrame(scaled, columns=[f"{col}_scaled" if keep_original else col for col in columns], index=df_pandas.index)
            df_out = df if preview else df.copy()
            if keep_original:
                df_out = df_out.join(dd.from_pandas(scaled_df, npartitions=df.npartitions))
            else:
                df_out = df_out.drop(columns=columns).join(dd.from_pandas(scaled_df, npartitions=df.npartitions))
            msg = f"Scaled {columns} using {method}."
        else:
            df_out = df if preview else df.copy()
            scaler = {'standard': StandardScaler(), 'minmax': MinMaxScaler(), 'robust': RobustScaler()}[method]
            scaled = scaler.fit_transform(df_out[columns])
            if keep_original:
                for i, col in enumerate(columns):
                    df_out[f"{col}_scaled"] = scaled[:, i]
            else:
                for i, col in enumerate(columns):
                    df_out[col] = scaled[:, i]
            msg = f"Scaled {columns} using {method}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in scale_features: {e}")
        return df, f"Error scaling features: {e}"

def rebalance_dataset(
    df: pd.DataFrame | dd.DataFrame,
    target: str,
    method: str = "oversample",
    ratio: float = 1.0,
    preview: bool = False,
) -> Tuple[pd.DataFrame | dd.DataFrame, str]:
    """
    Rebalance classification dataset.
    """
    try:
        if isinstance(df, dd.DataFrame):
            df_pandas = df.compute()  # Sampling-based rebalancing requires Pandas
            from sklearn.utils import resample
            counts = df_pandas[target].value_counts()
            majority_class = counts.idxmax()
            majority_count = counts.max()
            df_out = df_pandas
            if method == "oversample":
                minority_dfs = []
                for cls in counts.index:
                    if cls != majority_class:
                        cls_df = df_pandas[df_pandas[target] == cls]
                        target_size = int(majority_count * ratio)
                        minority_dfs.append(resample(cls_df, replace=True, n_samples=target_size, random_state=42))
                df_out = pd.concat([df_pandas[df_pandas[target] == majority_class]] + minority_dfs)
            elif method == "undersample":
                majority_df = df_pandas[df_pandas[target] == majority_class]
                target_size = int(counts.min() * ratio)
                majority_df = resample(majority_df, replace=False, n_samples=target_size, random_state=42)
                df_out = pd.concat([majority_df] + [df_pandas[df_pandas[target] == cls] for cls in counts.index if cls != majority_class])
            df_out = dd.from_pandas(df_out, npartitions=df.npartitions)
            msg = f"Rebalanced {target} using {method} with ratio {ratio}."
        else:
            df_out = df if preview else df.copy()
            from sklearn.utils import resample
            counts = df_out[target].value_counts()
            majority_class = counts.idxmax()
            majority_count = counts.max()
            if method == "oversample":
                minority_dfs = []
                for cls in counts.index:
                    if cls != majority_class:
                        cls_df = df_out[df_out[target] == cls]
                        target_size = int(majority_count * ratio)
                        minority_dfs.append(resample(cls_df, replace=True, n_samples=target_size, random_state=42))
                df_out = pd.concat([df_out[df_out[target] == majority_class]] + minority_dfs)
            elif method == "undersample":
                majority_df = df_out[df_out[target] == majority_class]
                target_size = int(counts.min() * ratio)
                majority_df = resample(majority_df, replace=False, n_samples=target_size, random_state=42)
                df_out = pd.concat([majority_df] + [df_out[df_out[target] == cls] for cls in counts.index if cls != majority_class])
            msg = f"Rebalanced {target} using {method} with ratio {ratio}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in rebalance_dataset: {e}")
        return df, f"Error rebalancing dataset: {e}"

def type_convert(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    type: str,
    preview: bool = False,
) -> Tuple[pd.DataFrame | dd.DataFrame, str]:
    """
    Convert column to specified type.
    """
    try:
        if isinstance(df, dd.DataFrame):
            df_out = df if preview else df.copy()
            if type == "bool":
                df_out[column] = df_out[column].map({0: False, 1: True, '0': False, '1': True}, na_action='ignore').astype(bool)
            elif type == "category":
                df_out[column] = df_out[column].astype('category')
            msg = f"Converted {column} to {type}."
        else:
            df_out = df if preview else df.copy()
            if type == "bool":
                df_out[column] = df_out[column].map({0: False, 1: True, '0': False, '1': True}, na_action='ignore').astype(bool)
            elif type == "category":
                df_out[column] = df_out[column].astype('category')
            msg = f"Converted {column} to {type}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in type_convert: {e}")
        return df, f"Error converting type: {e}"

def skewness_transform(
    df: pd.DataFrame | dd.DataFrame,
    columns: List[str],
    method: str = "log",
    preview: bool = False,
) -> Tuple[pd.DataFrame | dd.DataFrame, str]:
    """
    Apply transformation to reduce skewness.
    """
    try:
        if isinstance(df, dd.DataFrame):
            df_out = df if preview else df.copy()
            for col in columns:
                if method == "log":
                    df_out[col] = df_out[col].map(lambda x: np.log1p(x) if x > 0 else x, na_action='ignore')
                elif method == "sqrt":
                    df_out[col] = df_out[col].map(lambda x: np.sqrt(x) if x >= 0 else x, na_action='ignore')
                elif method == "boxcox":
                    df_pandas = df[columns].compute()
                    from scipy.stats import boxcox
                    df_pandas[col], _ = boxcox(df_pandas[col] + 1)
                    df_out = df_out.drop(columns=col).join(dd.from_pandas(df_pandas[[col]], npartitions=df.npartitions))
            msg = f"Applied {method} transformation to {columns}."
        else:
            df_out = df if preview else df.copy()
            for col in columns:
                if method == "log":
                    df_out[col] = df_out[col].apply(lambda x: np.log1p(x) if x > 0 else x)
                elif method == "sqrt":
                    df_out[col] = df_out[col].apply(lambda x: np.sqrt(x) if x >= 0 else x)
                elif method == "boxcox":
                    from scipy.stats import boxcox
                    df_out[col], _ = boxcox(df_out[col] + 1)
            msg = f"Applied {method} transformation to {columns}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in skewness_transform: {e}")
        return df, f"Error applying skewness transformation: {e}"

def mask_pii(
    df: pd.DataFrame | dd.DataFrame,
    columns: List[str],
    patterns: Dict[str, str] = None,
    preview: bool = False,
) -> Tuple[pd.DataFrame | dd.DataFrame, str]:
    """
    Mask personally identifiable information.
    """
    try:
        default_patterns = {
            "email": (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),
            "phone": (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]'),
            "credit_card": (r'\b\d{4}-\d{4}-\d{4}-\d{4}\b', '[CREDIT_CARD]')
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
                        s = s.str.replace(pattern, replacement, regex=True)
                    df_part[col] = s
                return df_part
            df_out = df_out.map_partitions(mask_partition)
            msg = f"Masked PII in {columns}."
        else:
            df_out = df if preview else df.copy()
            for col in columns:
                if col not in df_out.columns:
                    return df, f"Column {col} not found."
                s = df_out[col].astype(str)
                for _, (pattern, replacement) in patterns.items():
                    s = s.str.replace(pattern, replacement, regex=True)
                df_out[col] = s
            msg = f"Masked PII in {columns}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in mask_pii: {e}")
        return df, f"Error masking PII: {e}"

def smooth_time_series(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    method: str = "moving_average",
    window: int = 5,
    interpolate: str = "none",
    preview: bool = False,
) -> Tuple[pd.DataFrame | dd.DataFrame, str]:
    """
    Smooth time-series data.
    """
    try:
        if isinstance(df, dd.DataFrame):
            df_out = df if preview else df.copy()
            if interpolate != "none":
                df_out[column] = df_out[column].fillna(method={'linear': 'linear', 'ffill': 'ffill', 'bfill': 'bfill'}[interpolate])
            if method == "moving_average":
                df_out[column] = df_out[column].rolling(window=window, min_periods=1).mean()
            elif method == "savitzky_golay":
                df_pandas = df[[column]].compute()
                df_pandas[column] = savgol_filter(df_pandas[column].fillna(df_pandas[column].mean()), window, 3)
                df_out = df_out.drop(columns=column).join(dd.from_pandas(df_pandas[[column]], npartitions=df.npartitions))
            msg = f"Smoothed {column} using {method} with window {window}."
        else:
            df_out = df if preview else df.copy()
            if interpolate != "none":
                df_out[column] = df_out[column].interpolate(method=interpolate)
            if method == "moving_average":
                df_out[column] = df_out[column].rolling(window=window, min_periods=1).mean()
            elif method == "savitzky_golay":
                df_out[column] = savgol_filter(df_out[column].fillna(df_out[column].mean()), window, 3)
            msg = f"Smoothed {column} using {method} with window {window}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in smooth_time_series: {e}")
        return df, f"Error smoothing time-series: {e}"

def resample_time_series(
    df: pd.DataFrame | dd.DataFrame,
    time_column: str,
    freq: str,
    agg_func: str = "mean",
    preview: bool = False,
) -> Tuple[pd.DataFrame | dd.DataFrame, str]:
    """
    Resample time-series data.
    """
    try:
        if isinstance(df, dd.DataFrame):
            df_out = df if preview else df.copy()
            df_out = df_out.set_index(time_column)
            df_out = df_out.resample(freq).agg(agg_func).reset_index()
            msg = f"Resampled time-series on {time_column} with frequency {freq} using {agg_func}."
        else:
            df_out = df if preview else df.copy()
            df_out[time_column] = pd.to_datetime(df_out[time_column])
            df_out = df_out.set_index(time_column).resample(freq).agg(agg_func).reset_index()
            msg = f"Resampled time-series on {time_column} with frequency {freq} using {agg_func}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in resample_time_series: {e}")
        return df, f"Error resampling time-series: {e}"

def clean_text(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    remove_stopwords: bool = False,
    lemmatize: bool = False,
    preview: bool = False,
) -> Tuple[pd.DataFrame | dd.DataFrame, str]:
    """
    Clean text data by removing stopwords and lemmatizing.
    """
    try:
        if isinstance(df, dd.DataFrame):
            df_pandas = df[[column]].compute()
            stop_words = set(stopwords.words('english')) if remove_stopwords else set()
            lemmatizer = WordNetLemmatizer() if lemmatize else None
            def clean(s):
                tokens = word_tokenize(s.lower())
                if remove_stopwords:
                    tokens = [t for t in tokens if t not in stop_words]
                if lemmatize:
                    tokens = [lemmatizer.lemmatize(t) for t in tokens]
                return ' '.join(tokens)
            df_pandas[column] = df_pandas[column].astype(str).apply(clean)
            df_out = df if preview else df.copy()
            df_out = df_out.drop(columns=column).join(dd.from_pandas(df_pandas[[column]], npartitions=df.npartitions))
            msg = f"Cleaned text in {column} (stopwords={remove_stopwords}, lemmatize={lemmatize})."
        else:
            df_out = df if preview else df.copy()
            stop_words = set(stopwords.words('english')) if remove_stopwords else set()
            lemmatizer = WordNetLemmatizer() if lemmatize else None
            def clean(s):
                tokens = word_tokenize(s.lower())
                if remove_stopwords:
                    tokens = [t for t in tokens if t not in stop_words]
                if lemmatize:
                    tokens = [lemmatizer.lemmatize(t) for t in tokens]
                return ' '.join(tokens)
            df_out[column] = df_out[column].astype(str).apply(clean)
            msg = f"Cleaned text in {column} (stopwords={remove_stopwords}, lemmatize={lemmatize})."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in clean_text: {e}")
        return df, f"Error cleaning text: {e}"

def extract_tfidf(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    max_features: int = 100,
    preview: bool = False,
) -> Tuple[pd.DataFrame | dd.DataFrame, str]:
    """
    Extract TF-IDF features from a text column.
    """
    try:
        if isinstance(df, dd.DataFrame):
            df_pandas = df[[column]].compute()
            tfidf = TfidfVectorizer(max_features=max_features)
            tfidf_matrix = tfidf.fit_transform(df_pandas[column].astype(str))
            feature_names = [f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
            tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names, index=df_pandas.index)
            df_out = df.drop(columns=column).join(dd.from_pandas(tfidf_df, npartitions=df.npartitions))
            msg = f"Extracted {len(feature_names)} TF-IDF features from {column}."
        else:
            df_out = df if preview else df.copy()
            tfidf = TfidfVectorizer(max_features=max_features)
            tfidf_matrix = tfidf.fit_transform(df_out[column].astype(str))
            feature_names = [f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
            tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
            df_out = df_out.drop(columns=column).join(tfidf_df)
            msg = f"Extracted {len(feature_names)} TF-IDF features from {column}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in extract_tfidf: {e}")
        return df, f"Error extracting TF-IDF features: {e}"

def extract_domain(
    df: pd.DataFrame | dd.DataFrame,
    column: str,
    preview: bool = False,
) -> Tuple[pd.DataFrame | dd.DataFrame, str]:
    """
    Extract domain from URLs in a column.
    """
    try:
        if isinstance(df, dd.DataFrame):
            df_out = df if preview else df.copy()
            def extract_partition(df_part):
                if column not in df_part.columns:
                    return df_part
                df_part[column] = df_part[column].astype(str).apply(lambda x: urllib.parse.urlparse(x).netloc if x.startswith(('http://', 'https://')) else x)
                return df_part
            df_out = df_out.map_partitions(extract_partition)
            msg = f"Extracted domains from {column}."
        else:
            df_out = df if preview else df.copy()
            if column not in df_out.columns:
                return df, f"Column {column} not found."
            df_out[column] = df_out[column].astype(str).apply(lambda x: urllib.parse.urlparse(x).netloc if x.startswith(('http://', 'https://')) else x)
            msg = f"Extracted domains from {column}."
        logger.info(msg)
        return df_out, msg
    except Exception as e:
        logger.error(f"Error in extract_domain: {e}")
        return df, f"Error extracting domains: {e}"
