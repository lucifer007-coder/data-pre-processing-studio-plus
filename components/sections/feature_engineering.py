import streamlit as st
import pandas as pd
import dask.dataframe as dd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_regression, mutual_info_classif, f_regression, f_classif
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from dask_ml.preprocessing import PolynomialFeatures as DaskPolynomialFeatures
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, is_string_dtype
import altair as alt
import logging
from typing import List, Tuple, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import threading
import uuid
import numexpr as ne
import io
import warnings
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import re
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)

# Enhanced feature engineering configuration
FEATURE_CONFIG = {
    'max_polynomial_degree': 3,
    'max_interaction_features': 20,
    'min_variance_threshold': 1e-6,
    'correlation_threshold': 0.95,
    'mutual_info_threshold': 0.01,
    'max_categorical_cardinality': 50,
    'text_feature_min_length': 3,
    'datetime_feature_combinations': True,
    'enable_clustering_features': True,
    'enable_pca_features': True,
    'enable_statistical_features': True,
    'enable_domain_features': True
}

# Thread lock for session state updates
session_lock = threading.Lock()

# Placeholder implementations for external utilities
def dtype_split(df: Union[pd.DataFrame, dd.DataFrame]) -> Tuple[List[str], List[str]]:
    """Split columns into numeric and non-numeric."""
    try:
        num_cols = [col for col in df.columns if is_numeric_dtype(df[col])]
        cat_cols = [col for col in df.columns if col not in num_cols]
        return num_cols, cat_cols
    except Exception as e:
        logger.error(f"Error in dtype_split: {str(e)}")
        return [], []

def sample_for_preview(df: Union[pd.DataFrame, dd.DataFrame], n: int = 1000) -> pd.DataFrame:
    """Sample DataFrame for preview."""
    try:
        if isinstance(df, dd.DataFrame):
            total_rows = len(df)
            if hasattr(total_rows, 'compute'):
                total_rows = total_rows.compute()
            if total_rows == 0:
                return pd.DataFrame()
            frac = min(1.0, n / max(total_rows, 1))
            return df.sample(frac=frac).compute()
        return df.sample(n=min(n, len(df))) if len(df) > 0 else df
    except Exception as e:
        logger.error(f"Error in sample_for_preview: {str(e)}")
        return pd.DataFrame()

def alt_histogram(series: pd.Series, title: str) -> alt.Chart:
    """Create a histogram using Altair."""
    try:
        if series.empty:
            return alt.Chart(pd.DataFrame()).mark_text(text="No data available")
        
        data = pd.DataFrame({title: series.dropna()})
        if data.empty:
            return alt.Chart(pd.DataFrame()).mark_text(text="No valid data")
            
        return alt.Chart(data).mark_bar().encode(
            x=alt.X(f"{title}:Q", bin=True),
            y='count()'
        )
    except Exception as e:
        logger.error(f"Error in alt_histogram: {str(e)}")
        return alt.Chart(pd.DataFrame()).mark_text(text="Error creating histogram")

def compute_basic_stats(df: Union[pd.DataFrame, dd.DataFrame]) -> Dict[str, Any]:
    """Compute basic statistics."""
    try:
        if df is None:
            return {'n_rows': 0, 'n_columns': 0, 'columns': []}
            
        if isinstance(df, dd.DataFrame):
            df_computed = df.compute()
        else:
            df_computed = df
            
        return {
            'n_rows': len(df_computed),
            'n_columns': len(df_computed.columns),
            'columns': list(df_computed.columns)
        }
    except Exception as e:
        logger.error(f"Error in compute_basic_stats: {str(e)}")
        return {'n_rows': 0, 'n_columns': 0, 'columns': []}

def compare_stats(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """Compare stats before and after."""
    try:
        before_cols = set(before.get('columns', []))
        after_cols = set(after.get('columns', []))
        
        return {
            'n_columns_before': before.get('n_columns', 0),
            'n_columns_after': after.get('n_columns', 0),
            'added_columns': list(after_cols - before_cols),
            'removed_columns': list(before_cols - after_cols)
        }
    except Exception as e:
        logger.error(f"Error in compare_stats: {str(e)}")
        return {
            'n_columns_before': 0,
            'n_columns_after': 0,
            'added_columns': [],
            'removed_columns': []
        }

def push_history(message: str):
    """Push message to session state history."""
    try:
        with session_lock:
            if 'history' not in st.session_state:
                st.session_state.history = []
            st.session_state.history.append({
                'message': str(message)[:500],  # Limit message length
                'timestamp': datetime.now().isoformat()
            })
            # Limit history size to prevent memory issues
            if len(st.session_state.history) > 100:
                st.session_state.history = st.session_state.history[-100:]
    except Exception as e:
        logger.error(f"Error in push_history: {str(e)}")

def validate_step_function(func):
    """Validate a step function (placeholder)."""
    return func

# Local feature engineering step registry
FEATURE_STEP_REGISTRY = {
    "create_polynomial_features": {
        "func": validate_step_function(lambda df, **kwargs: create_polynomial_features(df, **kwargs)),
        "depends_on": []
    },
    "extract_datetime_features": {
        "func": validate_step_function(lambda df, **kwargs: extract_datetime_features(df, **kwargs)),
        "depends_on": []
    },
    "bin_features": {
        "func": validate_step_function(lambda df, **kwargs: bin_features(df, **kwargs)),
        "depends_on": []
    },
    "select_features_correlation": {
        "func": validate_step_function(lambda df, **kwargs: select_features_correlation(df, **kwargs)),
        "depends_on": []
    },
    "automated_feature_engineering": {
        "func": validate_step_function(lambda df, **kwargs: automated_feature_engineering(df, **kwargs)),
        "depends_on": []
    }
}

def analyze_dataset_advanced(df: Union[pd.DataFrame, dd.DataFrame], target_col: Optional[str] = None, sample_size: int = 10000) -> Dict[str, Any]:
    """Advanced dataset analysis with feature importance and statistical insights."""
    try:
        if df is None or df.empty:
            return {}
            
        # Sample for analysis if dataset is large
        if isinstance(df, dd.DataFrame):
            total_rows = len(df)
            if hasattr(total_rows, 'compute'):
                total_rows = total_rows.compute()
            if total_rows > sample_size:
                df_sample = sample_for_preview(df, n=sample_size)
            else:
                df_sample = df.compute()
        else:
            if len(df) > sample_size:
                df_sample = sample_for_preview(df, n=sample_size)
            else:
                df_sample = df
        
        if df_sample.empty:
            return {}
        
        analysis = {
            'numeric_cols': [],
            'categorical_cols': [],
            'datetime_cols': [],
            'text_cols': [],
            'binary_cols': [],
            'high_cardinality_cols': [],
            'stats': {},
            'feature_importance': {},
            'data_quality': {},
            'patterns': {}
        }
        
        total_rows = len(df_sample)
        
        for col in df.columns:
            try:
                # Basic statistics
                unique_count = df_sample[col].nunique()
                missing_count = df_sample[col].isna().sum()
                missing_rate = missing_count / max(total_rows, 1)
                
                analysis['stats'][col] = {
                    'unique_count': unique_count,
                    'missing_rate': missing_rate,
                    'variance': None,
                    'skewness': None,
                    'kurtosis': None,
                    'type': None,
                    'cardinality_ratio': unique_count / max(total_rows, 1)
                }
                
                # Data quality assessment
                analysis['data_quality'][col] = {
                    'completeness': 1 - missing_rate,
                    'uniqueness': unique_count / max(total_rows, 1),
                    'consistency': 1.0  # Placeholder for consistency checks
                }
                
                # Determine column type with enhanced logic
                if is_numeric_dtype(df_sample[col]):
                    analysis['numeric_cols'].append(col)
                    analysis['stats'][col]['type'] = 'numeric'
                    
                    # Calculate statistical measures
                    try:
                        col_data = df_sample[col].dropna()
                        if len(col_data) > 1:
                            analysis['stats'][col]['variance'] = col_data.var()
                            analysis['stats'][col]['skewness'] = stats.skew(col_data)
                            analysis['stats'][col]['kurtosis'] = stats.kurtosis(col_data)
                            
                            # Check if binary numeric
                            if unique_count == 2:
                                analysis['binary_cols'].append(col)
                    except Exception:
                        pass
                        
                elif is_datetime64_any_dtype(df_sample[col]):
                    analysis['datetime_cols'].append(col)
                    analysis['stats'][col]['type'] = 'datetime'
                    
                    # Analyze datetime patterns
                    try:
                        dt_data = df_sample[col].dropna()
                        if len(dt_data) > 1:
                            date_range = dt_data.max() - dt_data.min()
                            analysis['patterns'][col] = {
                                'date_range_days': date_range.days if hasattr(date_range, 'days') else None,
                                'has_time_component': dt_data.dt.hour.nunique() > 1,
                                'frequency_pattern': 'irregular'  # Could be enhanced with frequency detection
                            }
                    except Exception:
                        pass
                        
                elif is_string_dtype(df_sample[col]) or df_sample[col].dtype == 'object':
                    # Enhanced text/categorical classification
                    unique_ratio = unique_count / max(total_rows, 1)
                    avg_length = df_sample[col].astype(str).str.len().mean()
                    
                    if unique_count > FEATURE_CONFIG['max_categorical_cardinality']:
                        analysis['high_cardinality_cols'].append(col)
                    
                    if unique_ratio > 0.8 or avg_length > 20:  # Likely text
                        analysis['text_cols'].append(col)
                        analysis['stats'][col]['type'] = 'text'
                        
                        # Text analysis
                        analysis['patterns'][col] = {
                            'avg_length': avg_length,
                            'has_numbers': df_sample[col].astype(str).str.contains(r'\d').any(),
                            'has_special_chars': df_sample[col].astype(str).str.contains(r'[^a-zA-Z0-9\s]').any(),
                            'word_count_avg': df_sample[col].astype(str).str.split().str.len().mean()
                        }
                    else:  # Categorical
                        analysis['categorical_cols'].append(col)
                        analysis['stats'][col]['type'] = 'categorical'
                        
                        # Check if binary categorical
                        if unique_count == 2:
                            analysis['binary_cols'].append(col)
                else:
                    analysis['categorical_cols'].append(col)
                    analysis['stats'][col]['type'] = 'other'
                    
            except Exception as e:
                logger.warning(f"Error analyzing column {col}: {str(e)}")
                continue
        
        # Feature importance analysis if target is provided
        if target_col and target_col in df_sample.columns:
            try:
                analysis['feature_importance'] = calculate_feature_importance(
                    df_sample, target_col, analysis
                )
            except Exception as e:
                logger.warning(f"Error calculating feature importance: {str(e)}")
        
        # Correlation analysis for numeric columns
        if len(analysis['numeric_cols']) > 1:
            try:
                numeric_df = df_sample[analysis['numeric_cols']].select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    corr_matrix = numeric_df.corr()
                    analysis['correlations'] = corr_matrix.abs()
                    
                    # Find highly correlated pairs
                    high_corr_pairs = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr_val = abs(corr_matrix.iloc[i, j])
                            if corr_val > 0.7:
                                high_corr_pairs.append({
                                    'feature1': corr_matrix.columns[i],
                                    'feature2': corr_matrix.columns[j],
                                    'correlation': corr_val
                                })
                    analysis['high_correlations'] = high_corr_pairs
            except Exception as e:
                logger.warning(f"Error computing correlations: {str(e)}")
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error in analyze_dataset_advanced: {str(e)}")
        return {}

def calculate_feature_importance(df: pd.DataFrame, target_col: str, analysis: Dict) -> Dict[str, float]:
    """Calculate feature importance using multiple methods."""
    try:
        if target_col not in df.columns:
            return {}
        
        target = df[target_col].dropna()
        if len(target) == 0:
            return {}
        
        # Determine if regression or classification
        is_classification = (
            target_col in analysis['categorical_cols'] or 
            target_col in analysis['binary_cols'] or
            target.nunique() < 20
        )
        
        importance_scores = {}
        
        # Prepare features
        numeric_features = [col for col in analysis['numeric_cols'] if col != target_col]
        
        if numeric_features:
            X = df[numeric_features].fillna(df[numeric_features].median())
            y = target
            
            # Align X and y indices
            common_idx = X.index.intersection(y.index)
            X = X.loc[common_idx]
            y = y.loc[common_idx]
            
            if len(X) > 10:  # Minimum samples for meaningful analysis
                try:
                    # Mutual information
                    if is_classification:
                        mi_scores = mutual_info_classif(X, y, random_state=42)
                    else:
                        mi_scores = mutual_info_regression(X, y, random_state=42)
                    
                    for i, col in enumerate(numeric_features):
                        importance_scores[col] = {
                            'mutual_info': mi_scores[i],
                            'type': 'mutual_info'
                        }
                except Exception as e:
                    logger.warning(f"Error calculating mutual information: {str(e)}")
                
                try:
                    # Statistical tests
                    if is_classification:
                        f_scores, _ = f_classif(X, y)
                    else:
                        f_scores, _ = f_regression(X, y)
                    
                    for i, col in enumerate(numeric_features):
                        if col not in importance_scores:
                            importance_scores[col] = {}
                        importance_scores[col]['f_score'] = f_scores[i]
                except Exception as e:
                    logger.warning(f"Error calculating F-scores: {str(e)}")
                
                try:
                    # Random Forest importance
                    if is_classification:
                        rf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
                    else:
                        rf = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
                    
                    rf.fit(X, y)
                    rf_importance = rf.feature_importances_
                    
                    for i, col in enumerate(numeric_features):
                        if col not in importance_scores:
                            importance_scores[col] = {}
                        importance_scores[col]['rf_importance'] = rf_importance[i]
                except Exception as e:
                    logger.warning(f"Error calculating RF importance: {str(e)}")
        
        return importance_scores
        
    except Exception as e:
        logger.error(f"Error in calculate_feature_importance: {str(e)}")
        return {}

def create_advanced_numeric_features(df: Union[pd.DataFrame, dd.DataFrame], analysis: Dict, max_features: int = 20) -> Tuple[Union[pd.DataFrame, dd.DataFrame], List[str]]:
    """Create advanced numeric features including statistical and mathematical transformations."""
    try:
        df_out = df.copy()
        new_features = []
        feature_count = 0
        
        numeric_cols = analysis['numeric_cols']
        high_var_cols = [
            col for col in numeric_cols 
            if col in analysis['stats'] and 
            analysis['stats'][col]['variance'] is not None and 
            analysis['stats'][col]['variance'] > FEATURE_CONFIG['min_variance_threshold']
        ]
        
        # 1. Mathematical transformations
        for col in high_var_cols[:min(5, len(high_var_cols))]:
            if feature_count >= max_features:
                break
            
            try:
                col_data = df_out[col]
                
                # Log transformation (for positive skewed data)
                if analysis['stats'][col].get('skewness', 0) > 1:
                    new_col = f"log_{col}"
                    if isinstance(df_out, dd.DataFrame):
                        df_out[new_col] = col_data.map_partitions(
                            lambda s: np.log1p(np.maximum(s, 0)), 
                            meta=(new_col, 'float64')
                        )
                    else:
                        df_out[new_col] = np.log1p(np.maximum(col_data, 0))
                    new_features.append(new_col)
                    feature_count += 1
                
                # Square root for moderate skewness
                if 0.5 < analysis['stats'][col].get('skewness', 0) <= 1 and feature_count < max_features:
                    new_col = f"sqrt_{col}"
                    if isinstance(df_out, dd.DataFrame):
                        df_out[new_col] = col_data.map_partitions(
                            lambda s: np.sqrt(np.maximum(s, 0)), 
                            meta=(new_col, 'float64')
                        )
                    else:
                        df_out[new_col] = np.sqrt(np.maximum(col_data, 0))
                    new_features.append(new_col)
                    feature_count += 1
                
                # Reciprocal transformation
                if feature_count < max_features:
                    new_col = f"reciprocal_{col}"
                    if isinstance(df_out, dd.DataFrame):
                        df_out[new_col] = col_data.map_partitions(
                            lambda s: 1 / (s + 1e-8), 
                            meta=(new_col, 'float64')
                        )
                    else:
                        df_out[new_col] = 1 / (col_data + 1e-8)
                    new_features.append(new_col)
                    feature_count += 1
                    
            except Exception as e:
                logger.warning(f"Error creating transformations for {col}: {str(e)}")
        
        # 2. Statistical features (rolling windows for time series-like data)
        if len(high_var_cols) >= 2 and feature_count < max_features:
            try:
                # Create statistical aggregations
                for col in high_var_cols[:3]:
                    if feature_count >= max_features:
                        break
                    
                    # Z-score (standardized values)
                    new_col = f"zscore_{col}"
                    if isinstance(df_out, dd.DataFrame):
                        col_mean = df_out[col].mean().compute()
                        col_std = df_out[col].std().compute()
                        df_out[new_col] = (df_out[col] - col_mean) / (col_std + 1e-8)
                    else:
                        col_mean = df_out[col].mean()
                        col_std = df_out[col].std()
                        df_out[new_col] = (df_out[col] - col_mean) / (col_std + 1e-8)
                    new_features.append(new_col)
                    feature_count += 1
                    
            except Exception as e:
                logger.warning(f"Error creating statistical features: {str(e)}")
        
        # 3. Interaction features based on correlation
        if 'correlations' in analysis and feature_count < max_features:
            try:
                corr_matrix = analysis['correlations']
                interaction_count = 0
                
                for i, col1 in enumerate(high_var_cols):
                    if feature_count >= max_features or interaction_count >= FEATURE_CONFIG['max_interaction_features']:
                        break
                    for col2 in high_var_cols[i+1:]:
                        if feature_count >= max_features or interaction_count >= FEATURE_CONFIG['max_interaction_features']:
                            break
                        
                        if col1 in corr_matrix.index and col2 in corr_matrix.columns:
                            corr_val = corr_matrix.loc[col1, col2]
                            if 0.3 <= corr_val <= 0.8:  # Moderate correlation
                                # Multiplication
                                new_col = f"{col1}_x_{col2}"
                                df_out[new_col] = df_out[col1] * df_out[col2]
                                new_features.append(new_col)
                                feature_count += 1
                                interaction_count += 1
                                
                                # Division (if denominator is not zero)
                                if feature_count < max_features:
                                    new_col = f"{col1}_div_{col2}"
                                    if isinstance(df_out, dd.DataFrame):
                                        df_out[new_col] = df_out[col1] / (df_out[col2] + 1e-8)
                                    else:
                                        df_out[new_col] = df_out[col1] / (df_out[col2] + 1e-8)
                                    new_features.append(new_col)
                                    feature_count += 1
                                    interaction_count += 1
                                    
            except Exception as e:
                logger.warning(f"Error creating interaction features: {str(e)}")
        
        return df_out, new_features
        
    except Exception as e:
        logger.error(f"Error in create_advanced_numeric_features: {str(e)}")
        return df, []

def create_advanced_categorical_features(df: Union[pd.DataFrame, dd.DataFrame], analysis: Dict, max_features: int = 15) -> Tuple[Union[pd.DataFrame, dd.DataFrame], List[str]]:
    """Create advanced categorical features including encoding and aggregations."""
    try:
        df_out = df.copy()
        new_features = []
        feature_count = 0
        
        categorical_cols = [
            col for col in analysis['categorical_cols'] 
            if col not in analysis['high_cardinality_cols']
        ]
        
        for col in categorical_cols[:min(5, len(categorical_cols))]:
            if feature_count >= max_features:
                break
            
            try:
                # 1. Frequency encoding
                new_col = f"freq_{col}"
                if isinstance(df_out, dd.DataFrame):
                    freq_map = df_out[col].value_counts().compute().to_dict()
                    df_out[new_col] = df_out[col].map(freq_map, meta=(new_col, 'int64'))
                else:
                    freq_map = df_out[col].value_counts().to_dict()
                    df_out[new_col] = df_out[col].map(freq_map)
                new_features.append(new_col)
                feature_count += 1
                
                # 2. Rank encoding
                if feature_count < max_features:
                    new_col = f"rank_{col}"
                    if isinstance(df_out, dd.DataFrame):
                        rank_map = df_out[col].value_counts().compute().rank(ascending=False).to_dict()
                        df_out[new_col] = df_out[col].map(rank_map, meta=(new_col, 'float64'))
                    else:
                        rank_map = df_out[col].value_counts().rank(ascending=False).to_dict()
                        df_out[new_col] = df_out[col].map(rank_map)
                    new_features.append(new_col)
                    feature_count += 1
                
                # 3. Binary encoding for categories with moderate cardinality
                unique_count = analysis['stats'][col]['unique_count']
                if 3 <= unique_count <= 10 and feature_count < max_features:
                    # Create binary features for top categories
                    if isinstance(df_out, dd.DataFrame):
                        top_categories = df_out[col].value_counts().compute().head(3).index
                    else:
                        top_categories = df_out[col].value_counts().head(3).index
                    
                    for category in top_categories:
                        if feature_count >= max_features:
                            break
                        new_col = f"is_{col}_{str(category)[:10]}"  # Limit name length
                        df_out[new_col] = (df_out[col] == category).astype(int)
                        new_features.append(new_col)
                        feature_count += 1
                        
            except Exception as e:
                logger.warning(f"Error creating categorical features for {col}: {str(e)}")
        
        return df_out, new_features
        
    except Exception as e:
        logger.error(f"Error in create_advanced_categorical_features: {str(e)}")
        return df, []

def create_advanced_datetime_features(df: Union[pd.DataFrame, dd.DataFrame], analysis: Dict, max_features: int = 20) -> Tuple[Union[pd.DataFrame, dd.DataFrame], List[str]]:
    """Create comprehensive datetime features including cyclical and business features."""
    try:
        df_out = df.copy()
        new_features = []
        feature_count = 0
        
        datetime_cols = analysis['datetime_cols']
        
        for col in datetime_cols[:min(3, len(datetime_cols))]:
            if feature_count >= max_features:
                break
            
            try:
                # Ensure datetime type
                if not is_datetime64_any_dtype(df_out[col]):
                    if isinstance(df_out, dd.DataFrame):
                        df_out[col] = dd.to_datetime(df_out[col], errors='coerce')
                    else:
                        df_out[col] = pd.to_datetime(df_out[col], errors='coerce')
                
                # Basic datetime features
                basic_features = ["year", "month", "day", "dayofweek", "hour", "quarter"]
                for feature in basic_features:
                    if feature_count >= max_features:
                        break
                    try:
                        new_col = f"{col}_{feature}"
                        if isinstance(df_out, dd.DataFrame):
                            df_out[new_col] = getattr(df_out[col].dt, feature)
                        else:
                            df_out[new_col] = getattr(df_out[col].dt, feature)
                        new_features.append(new_col)
                        feature_count += 1
                    except AttributeError:
                        continue
                
                # Advanced datetime features
                if feature_count < max_features:
                    # Is weekend
                    new_col = f"{col}_is_weekend"
                    if isinstance(df_out, dd.DataFrame):
                        df_out[new_col] = (df_out[col].dt.dayofweek >= 5).astype(int)
                    else:
                        df_out[new_col] = (df_out[col].dt.dayofweek >= 5).astype(int)
                    new_features.append(new_col)
                    feature_count += 1
                
                if feature_count < max_features:
                    # Is month end
                    new_col = f"{col}_is_month_end"
                    if isinstance(df_out, dd.DataFrame):
                        df_out[new_col] = df_out[col].dt.is_month_end.astype(int)
                    else:
                        df_out[new_col] = df_out[col].dt.is_month_end.astype(int)
                    new_features.append(new_col)
                    feature_count += 1
                
                # Cyclical features (sin/cos encoding)
                if feature_count < max_features - 1:
                    # Month cyclical
                    new_col_sin = f"{col}_month_sin"
                    new_col_cos = f"{col}_month_cos"
                    if isinstance(df_out, dd.DataFrame):
                        df_out[new_col_sin] = df_out[col].dt.month.map_partitions(
                            lambda s: np.sin(2 * np.pi * s / 12), 
                            meta=(new_col_sin, 'float64')
                        )
                        df_out[new_col_cos] = df_out[col].dt.month.map_partitions(
                            lambda s: np.cos(2 * np.pi * s / 12), 
                            meta=(new_col_cos, 'float64')
                        )
                    else:
                        df_out[new_col_sin] = np.sin(2 * np.pi * df_out[col].dt.month / 12)
                        df_out[new_col_cos] = np.cos(2 * np.pi * df_out[col].dt.month / 12)
                    new_features.extend([new_col_sin, new_col_cos])
                    feature_count += 2
                
                # Time since epoch (for trend analysis)
                if feature_count < max_features:
                    new_col = f"{col}_timestamp"
                    if isinstance(df_out, dd.DataFrame):
                        df_out[new_col] = df_out[col].map_partitions(
                            lambda s: s.astype('int64') // 10**9, 
                            meta=(new_col, 'int64')
                        )
                    else:
                        df_out[new_col] = df_out[col].astype('int64') // 10**9
                    new_features.append(new_col)
                    feature_count += 1
                    
            except Exception as e:
                logger.warning(f"Error creating datetime features for {col}: {str(e)}")
        
        return df_out, new_features
        
    except Exception as e:
        logger.error(f"Error in create_advanced_datetime_features: {str(e)}")
        return df, []

def create_advanced_text_features(df: Union[pd.DataFrame, dd.DataFrame], analysis: Dict, max_features: int = 10) -> Tuple[Union[pd.DataFrame, dd.DataFrame], List[str]]:
    """Create advanced text features including NLP-inspired features."""
    try:
        df_out = df.copy()
        new_features = []
        feature_count = 0
        
        text_cols = analysis['text_cols'][:min(2, len(analysis['text_cols']))]
        
        for col in text_cols:
            if feature_count >= max_features:
                break
            
            try:
                # Basic text features
                new_col = f"len_{col}"
                if isinstance(df_out, dd.DataFrame):
                    df_out[new_col] = df_out[col].map_partitions(
                        lambda s: s.astype(str).str.len(), 
                        meta=(new_col, 'int64')
                    )
                else:
                    df_out[new_col] = df_out[col].astype(str).str.len()
                new_features.append(new_col)
                feature_count += 1
                
                # Word count
                if feature_count < max_features:
                    new_col = f"word_count_{col}"
                    if isinstance(df_out, dd.DataFrame):
                        df_out[new_col] = df_out[col].map_partitions(
                            lambda s: s.astype(str).str.split().str.len(), 
                            meta=(new_col, 'int64')
                        )
                    else:
                        df_out[new_col] = df_out[col].astype(str).str.split().str.len()
                    new_features.append(new_col)
                    feature_count += 1
                
                # Number of digits
                if feature_count < max_features:
                    new_col = f"digit_count_{col}"
                    if isinstance(df_out, dd.DataFrame):
                        df_out[new_col] = df_out[col].map_partitions(
                            lambda s: s.astype(str).str.count(r'\d'), 
                            meta=(new_col, 'int64')
                        )
                    else:
                        df_out[new_col] = df_out[col].astype(str).str.count(r'\d')
                    new_features.append(new_col)
                    feature_count += 1
                
                # Number of special characters
                if feature_count < max_features:
                    new_col = f"special_char_count_{col}"
                    if isinstance(df_out, dd.DataFrame):
                        df_out[new_col] = df_out[col].map_partitions(
                            lambda s: s.astype(str).str.count(r'[^a-zA-Z0-9\s]'), 
                            meta=(new_col, 'int64')
                        )
                    else:
                        df_out[new_col] = df_out[col].astype(str).str.count(r'[^a-zA-Z0-9\s]')
                    new_features.append(new_col)
                    feature_count += 1
                
                # Average word length
                if feature_count < max_features:
                    new_col = f"avg_word_len_{col}"
                    if isinstance(df_out, dd.DataFrame):
                        df_out[new_col] = df_out[col].map_partitions(
                            lambda s: s.astype(str).str.split().apply(
                                lambda words: np.mean([len(w) for w in words]) if words else 0
                            ), 
                            meta=(new_col, 'float64')
                        )
                    else:
                        df_out[new_col] = df_out[col].astype(str).str.split().apply(
                            lambda words: np.mean([len(w) for w in words]) if words else 0
                        )
                    new_features.append(new_col)
                    feature_count += 1
                    
            except Exception as e:
                logger.warning(f"Error creating text features for {col}: {str(e)}")
        
        return df_out, new_features
        
    except Exception as e:
        logger.error(f"Error in create_advanced_text_features: {str(e)}")
        return df, []

def create_clustering_features(df: Union[pd.DataFrame, dd.DataFrame], analysis: Dict, max_features: int = 5) -> Tuple[Union[pd.DataFrame, dd.DataFrame], List[str]]:
    """Create clustering-based features for unsupervised pattern discovery."""
    try:
        if not FEATURE_CONFIG['enable_clustering_features']:
            return df, []
        
        df_out = df.copy()
        new_features = []
        
        numeric_cols = [
            col for col in analysis['numeric_cols'] 
            if col in analysis['stats'] and 
            analysis['stats'][col]['variance'] is not None and 
            analysis['stats'][col]['variance'] > FEATURE_CONFIG['min_variance_threshold']
        ]
        
        if len(numeric_cols) < 2:
            return df, []
        
        # Sample data for clustering if too large
        if isinstance(df_out, dd.DataFrame):
            df_sample = sample_for_preview(df_out[numeric_cols], n=10000)
        else:
            df_sample = df_out[numeric_cols] if len(df_out) <= 10000 else sample_for_preview(df_out[numeric_cols], n=10000)
        
        if df_sample.empty:
            return df, []
        
        # Fill missing values
        df_sample_filled = df_sample.fillna(df_sample.median())
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_sample_filled)
        
        # K-means clustering
        n_clusters = min(5, max(2, len(df_sample) // 1000))  # Adaptive number of clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Create cluster feature
        new_col = "cluster_kmeans"
        if isinstance(df_out, dd.DataFrame):
            # For Dask, we need to apply clustering to the full dataset
            df_full_filled = df_out[numeric_cols].fillna(df_out[numeric_cols].median())
            X_full_scaled = df_full_filled.map_partitions(
                lambda x: pd.DataFrame(scaler.transform(x), index=x.index, columns=x.columns),
                meta=df_full_filled
            )
            # Apply clustering (this is approximate for Dask)
            df_out[new_col] = X_full_scaled.map_partitions(
                lambda x: pd.Series(kmeans.predict(x), index=x.index),
                meta=(new_col, 'int64')
            )
        else:
            df_full_filled = df_out[numeric_cols].fillna(df_out[numeric_cols].median())
            X_full_scaled = scaler.transform(df_full_filled)
            df_out[new_col] = kmeans.predict(X_full_scaled)
        
        new_features.append(new_col)
        
        # Distance to cluster centers
        if len(new_features) < max_features:
            new_col = "cluster_distance"
            if isinstance(df_out, dd.DataFrame):
                # Simplified distance calculation for Dask
                df_out[new_col] = X_full_scaled.map_partitions(
                    lambda x: pd.Series([np.min(np.linalg.norm(x.values - kmeans.cluster_centers_, axis=1)) for _ in range(len(x))], index=x.index),
                    meta=(new_col, 'float64')
                )
            else:
                distances = np.array([np.min(np.linalg.norm(X_full_scaled - kmeans.cluster_centers_, axis=1))])
                df_out[new_col] = np.min(np.linalg.norm(X_full_scaled[:, np.newaxis] - kmeans.cluster_centers_.T[np.newaxis, :], axis=2), axis=1)
            new_features.append(new_col)
        
        return df_out, new_features
        
    except Exception as e:
        logger.warning(f"Error creating clustering features: {str(e)}")
        return df, []

def create_pca_features(df: Union[pd.DataFrame, dd.DataFrame], analysis: Dict, max_features: int = 5) -> Tuple[Union[pd.DataFrame, dd.DataFrame], List[str]]:
    """Create PCA-based features for dimensionality reduction insights."""
    try:
        if not FEATURE_CONFIG['enable_pca_features']:
            return df, []
        
        df_out = df.copy()
        new_features = []
        
        numeric_cols = [
            col for col in analysis['numeric_cols'] 
            if col in analysis['stats'] and 
            analysis['stats'][col]['variance'] is not None and 
            analysis['stats'][col]['variance'] > FEATURE_CONFIG['min_variance_threshold']
        ]
        
        if len(numeric_cols) < 3:  # Need at least 3 features for meaningful PCA
            return df, []
        
        # Sample data for PCA if too large
        if isinstance(df_out, dd.DataFrame):
            df_sample = sample_for_preview(df_out[numeric_cols], n=10000)
        else:
            df_sample = df_out[numeric_cols] if len(df_out) <= 10000 else sample_for_preview(df_out[numeric_cols], n=10000)
        
        if df_sample.empty:
            return df, []
        
        # Fill missing values and standardize
        df_sample_filled = df_sample.fillna(df_sample.median())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_sample_filled)
        
        # Apply PCA
        n_components = min(max_features, len(numeric_cols), 5)
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create PCA features
        for i in range(n_components):
            new_col = f"pca_component_{i+1}"
            if isinstance(df_out, dd.DataFrame):
                # Apply PCA transformation to full dataset
                df_full_filled = df_out[numeric_cols].fillna(df_out[numeric_cols].median())
                X_full_scaled = df_full_filled.map_partitions(
                    lambda x: pd.DataFrame(scaler.transform(x), index=x.index, columns=x.columns),
                    meta=df_full_filled
                )
                df_out[new_col] = X_full_scaled.map_partitions(
                    lambda x: pd.Series(pca.transform(x)[:, i], index=x.index),
                    meta=(new_col, 'float64')
                )
            else:
                df_full_filled = df_out[numeric_cols].fillna(df_out[numeric_cols].median())
                X_full_scaled = scaler.transform(df_full_filled)
                X_full_pca = pca.transform(X_full_scaled)
                df_out[new_col] = X_full_pca[:, i]
            
            new_features.append(new_col)
        
        return df_out, new_features
        
    except Exception as e:
        logger.warning(f"Error creating PCA features: {str(e)}")
        return df, []

def create_polynomial_features(df: Union[pd.DataFrame, dd.DataFrame], columns: List[str], degree: int = 2, preview: bool = False) -> Tuple[Union[pd.DataFrame, dd.DataFrame], str]:
    """Create polynomial and interaction features for specified numeric columns."""
    try:
        if not isinstance(df, (pd.DataFrame, dd.DataFrame)):
            return df, "Invalid DataFrame input"
        
        if df.empty:
            return df, "DataFrame is empty"
            
        # Validate and filter columns
        valid_columns = []
        for col in columns:
            if col not in df.columns:
                continue
            if not is_numeric_dtype(df[col]):
                continue
            # Check for sufficient variance
            col_var = df[col].var()
            if isinstance(df, dd.DataFrame):
                col_var = col_var.compute()
            if pd.isna(col_var) or col_var == 0:
                continue
            valid_columns.append(col)
        
        if not valid_columns:
            return df, "No valid numeric columns with sufficient variance selected for polynomial features"
        
        # Validate degree
        degree = max(1, min(5, int(degree)))
        
        # Check for memory constraints
        if len(valid_columns) > 10 and degree > 2:
            return df, "Too many columns for high-degree polynomial features (memory constraint)"
        
        df_out = df if preview else df.copy()
        
        with st.spinner(f"Generating polynomial features (degree={degree})..."):
            try:
                if isinstance(df_out, dd.DataFrame):
                    # Use Dask for large datasets
                    poly = DaskPolynomialFeatures(degree=degree, include_bias=False)
                    selected_data = df_out[valid_columns].fillna(0)  # Handle NaN values
                    poly_features = poly.fit_transform(selected_data)
                    feature_names = poly.get_feature_names_out(valid_columns)
                    
                    # Create new DataFrame with polynomial features
                    poly_df = dd.from_array(poly_features, columns=feature_names)
                    # Remove original columns from polynomial features to avoid duplication
                    new_feature_names = [name for name in feature_names if name not in valid_columns]
                    if new_feature_names:
                        poly_df_new = poly_df[new_feature_names]
                        df_out = dd.concat([df_out, poly_df_new], axis=1)
                else:
                    # Use sklearn for regular pandas DataFrames
                    poly = PolynomialFeatures(degree=degree, include_bias=False)
                    selected_data = df_out[valid_columns].fillna(0)  # Handle NaN values
                    poly_features = poly.fit_transform(selected_data)
                    feature_names = poly.get_feature_names_out(valid_columns)
                    
                    # Create DataFrame with polynomial features
                    poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df_out.index)
                    # Remove original columns from polynomial features to avoid duplication
                    new_feature_names = [name for name in feature_names if name not in valid_columns]
                    if new_feature_names:
                        for name in new_feature_names:
                            df_out[name] = poly_df[name]
                
                msg = f"Created polynomial features (degree={degree}) for columns: {', '.join(valid_columns)}"
                logger.info(msg)
                
                if not preview:
                    with session_lock:
                        if 'pipeline' not in st.session_state:
                            st.session_state.pipeline = []
                        st.session_state.pipeline.append({
                            "kind": "create_polynomial_features", 
                            "params": {"columns": valid_columns, "degree": degree}
                        })
                
                return df_out, msg
                
            except MemoryError:
                logger.error("MemoryError in create_polynomial_features")
                return df, "Error: Dataset too large for polynomial feature creation"
                
    except ValueError as e:
        logger.error(f"ValueError in create_polynomial_features: {str(e)}")
        return df, f"Error creating polynomial features: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error in create_polynomial_features: {str(e)}")
        return df, f"Unexpected error: {str(e)}"

def extract_datetime_features(df: Union[pd.DataFrame, dd.DataFrame], columns: List[str], features: List[str], preview: bool = False) -> Tuple[Union[pd.DataFrame, dd.DataFrame], str]:
    """Extract datetime features (e.g., year, month, day) from specified columns."""
    try:
        if not isinstance(df, (pd.DataFrame, dd.DataFrame)):
            return df, "Invalid DataFrame input"
        
        if df.empty:
            return df, "DataFrame is empty"
            
        # Validate columns
        valid_columns = [c for c in columns if c in df.columns]
        if not valid_columns:
            return df, "No valid columns selected for datetime features"
        
        # Validate features
        valid_features_list = ["year", "month", "day", "hour", "minute", "second", "dayofweek", "quarter"]
        valid_features = [f for f in features if f in valid_features_list]
        if not valid_features:
            return df, "No valid datetime features selected"
        
        df_out = df if preview else df.copy()
        invalid_counts = {}
        
        with st.spinner(f"Extracting datetime features ({', '.join(valid_features)})"):
            for col in valid_columns:
                try:
                    # Convert to datetime if not already
                    if not is_datetime64_any_dtype(df_out[col]):
                        if isinstance(df_out, dd.DataFrame):
                            df_out[col] = dd.to_datetime(df_out[col], errors='coerce')
                        else:
                            df_out[col] = pd.to_datetime(df_out[col], errors='coerce')
                    
                    # Count invalid dates
                    invalid_count = df_out[col].isna().sum()
                    if isinstance(df_out, dd.DataFrame):
                        invalid_count = invalid_count.compute()
                    if invalid_count > 0:
                        invalid_counts[col] = invalid_count
                    
                    # Extract features
                    for feature in valid_features:
                        new_col = f"{col}_{feature}"
                        if new_col not in df_out.columns:  # Avoid overwriting existing columns
                            try:
                                if isinstance(df_out, dd.DataFrame):
                                    df_out[new_col] = getattr(df_out[col].dt, feature)
                                else:
                                    df_out[new_col] = getattr(df_out[col].dt, feature)
                            except AttributeError:
                                logger.warning(f"Feature {feature} not available for column {col}")
                                continue
                                
                except Exception as e:
                    logger.error(f"Error processing column {col}: {str(e)}")
                    continue
        
        msg = f"Extracted datetime features ({', '.join(valid_features)}) for columns: {', '.join(valid_columns)}"
        if invalid_counts:
            msg += f". Warning: Invalid datetimes found in {', '.join(f'{col} ({count} invalid)' for col, count in invalid_counts.items())}"
        
        logger.info(msg)
        
        if not preview:
            with session_lock:
                if 'pipeline' not in st.session_state:
                    st.session_state.pipeline = []
                st.session_state.pipeline.append({
                    "kind": "extract_datetime_features", 
                    "params": {"columns": valid_columns, "features": valid_features}
                })
        
        return df_out, msg
        
    except ValueError as e:
        logger.error(f"ValueError in extract_datetime_features: {str(e)}")
        return df, f"Error extracting datetime features: {str(e)}"
    except MemoryError:
        logger.error("MemoryError in extract_datetime_features")
        return df, "Error: Dataset too large for datetime extraction"
    except Exception as e:
        logger.error(f"Unexpected error in extract_datetime_features: {str(e)}")
        return df, f"Unexpected error: {str(e)}"

def bin_features(df: Union[pd.DataFrame, dd.DataFrame], columns: List[str], bins: int = 10, preview: bool = False) -> Tuple[Union[pd.DataFrame, dd.DataFrame], str]:
    """Bin numeric features into discrete intervals."""
    try:
        if not isinstance(df, (pd.DataFrame, dd.DataFrame)):
            return df, "Invalid DataFrame input"
        
        if df.empty:
            return df, "DataFrame is empty"
            
        # Validate columns
        valid_columns = []
        for col in columns:
            if col not in df.columns:
                continue
            if not is_numeric_dtype(df[col]):
                continue
            
            # Check for sufficient unique values
            unique_count = df[col].nunique()
            if isinstance(df, dd.DataFrame):
                unique_count = unique_count.compute()
            if unique_count < 2:
                continue
                
            valid_columns.append(col)
        
        if not valid_columns:
            return df, "No valid numeric columns with sufficient unique values selected for binning"
        
        # Validate bins parameter
        bins = max(2, min(50, int(bins)))
        df_out = df if preview else df.copy()
        
        with st.spinner(f"Binning columns into {bins} bins"):
            for col in valid_columns:
                new_col = f"{col}_binned"
                try:
                    if isinstance(df_out, dd.DataFrame):
                        # Use quantile-based binning for Dask
                        quantiles = np.linspace(0, 1, bins + 1)
                        bin_edges = df_out[col].quantile(quantiles).compute()
                        bin_edges = bin_edges.drop_duplicates().sort_values()
                        
                        if len(bin_edges) < 2:
                            logger.warning(f"Insufficient unique values for binning column {col}")
                            continue
                            
                        df_out[new_col] = df_out[col].map_partitions(
                            lambda s: pd.cut(s, bins=bin_edges, labels=False, include_lowest=True, duplicates='drop'),
                            meta=(new_col, 'int64')
                        )
                    else:
                        # Use pandas qcut for regular DataFrames
                        try:
                            df_out[new_col] = pd.qcut(df_out[col], q=bins, labels=False, duplicates='drop')
                        except ValueError:
                            # Fallback to regular cut if qcut fails
                            df_out[new_col] = pd.cut(df_out[col], bins=bins, labels=False, include_lowest=True, duplicates='drop')
                            
                except ValueError as e:
                    logger.warning(f"Skipping binning for {col}: {str(e)}")
                    continue
        
        msg = f"Binned columns ({bins} bins): {', '.join(valid_columns)}"
        logger.info(msg)
        
        if not preview:
            with session_lock:
                if 'pipeline' not in st.session_state:
                    st.session_state.pipeline = []
                st.session_state.pipeline.append({
                    "kind": "bin_features", 
                    "params": {"columns": valid_columns, "bins": bins}
                })
        
        return df_out, msg
        
    except ValueError as e:
        logger.error(f"ValueError in bin_features: {str(e)}")
        return df, f"Error binning features: {str(e)}"
    except MemoryError:
        logger.error("MemoryError in bin_features")
        return df, "Error: Dataset too large for binning"
    except Exception as e:
        logger.error(f"Unexpected error in bin_features: {str(e)}")
        return df, f"Unexpected error: {str(e)}"

def select_features_correlation(df: Union[pd.DataFrame, dd.DataFrame], threshold: float = 0.8, preview: bool = False) -> Tuple[Union[pd.DataFrame, dd.DataFrame], str]:
    """Select features based on correlation analysis."""
    try:
        if not isinstance(df, (pd.DataFrame, dd.DataFrame)):
            return df, "Invalid DataFrame input"
        
        if df.empty:
            return df, "DataFrame is empty"
            
        # Validate threshold
        threshold = max(0.1, min(0.99, float(threshold)))
        
        # Get numeric columns
        num_cols, _ = dtype_split(df)
        if not num_cols:
            return df, "No numeric columns available for correlation analysis"
        
        # Filter out constant columns
        valid_num_cols = []
        for col in num_cols:
            try:
                col_var = df[col].var()
                if isinstance(df, dd.DataFrame):
                    col_var = col_var.compute()
                if not pd.isna(col_var) and col_var > 1e-10:  # Not constant
                    valid_num_cols.append(col)
            except Exception:
                continue
        
        if len(valid_num_cols) < 2:
            return df, "Insufficient numeric columns for correlation analysis"
        
        with st.spinner("Computing correlation matrix..."):
            try:
                if isinstance(df, dd.DataFrame):
                    # Sample for correlation computation if dataset is too large
                    if len(df) > 50000:
                        df_sample = sample_for_preview(df[valid_num_cols], n=50000)
                        corr_matrix = df_sample.corr()
                    else:
                        corr_matrix = df[valid_num_cols].corr().compute()
                else:
                    corr_matrix = df[valid_num_cols].corr()
                
                # Handle NaN values in correlation matrix
                corr_matrix = corr_matrix.fillna(0)
                
                # Find highly correlated features
                corr_matrix_abs = corr_matrix.abs()
                upper_triangle = np.triu(np.ones(corr_matrix_abs.shape), k=1).astype(bool)
                upper_corr = corr_matrix_abs.where(upper_triangle)
                
                # Find columns to drop
                to_drop = []
                for column in upper_corr.columns:
                    if any(upper_corr[column] > threshold):
                        to_drop.append(column)
                
                df_out = df if preview else df.copy()
                
                if to_drop:
                    # Ensure we don't drop all columns
                    remaining_cols = [col for col in df_out.columns if col not in to_drop]
                    if len(remaining_cols) == 0:
                        return df, "Cannot drop all columns due to correlation threshold"
                    
                    df_out = df_out.drop(columns=to_drop)
                    msg = f"Dropped highly correlated columns (threshold={threshold}): {', '.join(to_drop)}"
                else:
                    msg = f"No columns dropped (correlation threshold={threshold})"
                
                logger.info(msg)
                
                if not preview:
                    with session_lock:
                        if 'pipeline' not in st.session_state:
                            st.session_state.pipeline = []
                        st.session_state.pipeline.append({
                            "kind": "select_features_correlation", 
                            "params": {"threshold": threshold}
                        })
                
                return df_out, msg
                
            except Exception as e:
                logger.error(f"Error computing correlation: {str(e)}")
                return df, f"Error computing correlation matrix: {str(e)}"
        
    except ValueError as e:
        logger.error(f"ValueError in select_features_correlation: {str(e)}")
        return df, f"Error in correlation-based feature selection: {str(e)}"
    except MemoryError:
        logger.error("MemoryError in select_features_correlation")
        return df, "Error: Dataset too large for correlation analysis"
    except Exception as e:
        logger.error(f"Unexpected error in select_features_correlation: {str(e)}")
        return df, f"Unexpected error: {str(e)}"

def analyze_dataset(df: Union[pd.DataFrame, dd.DataFrame], sample_size: int = 10000) -> Dict[str, Any]:
    """Analyze dataset to identify column types and statistical properties."""
    try:
        if df is None or df.empty:
            return {}
            
        # Sample for analysis if dataset is large
        if isinstance(df, dd.DataFrame):
            total_rows = len(df)
            if hasattr(total_rows, 'compute'):
                total_rows = total_rows.compute()
            if total_rows > sample_size:
                df_sample = sample_for_preview(df, n=sample_size)
            else:
                df_sample = df.compute()
        else:
            if len(df) > sample_size:
                df_sample = sample_for_preview(df, n=sample_size)
            else:
                df_sample = df
        
        if df_sample.empty:
            return {}
        
        analysis = {
            'numeric_cols': [],
            'categorical_cols': [],
            'datetime_cols': [],
            'text_cols': [],
            'stats': {}
        }
        
        for col in df.columns:
            try:
                # Compute statistics safely
                unique_count = df[col].nunique()
                missing_count = df[col].isna().sum()
                total_count = len(df)
                
                if isinstance(df, dd.DataFrame):
                    unique_count = unique_count.compute()
                    missing_count = missing_count.compute()
                    total_count = total_count.compute() if hasattr(total_count, 'compute') else total_count
                
                missing_rate = missing_count / max(total_count, 1)
                
                analysis['stats'][col] = {
                    'unique_count': unique_count,
                    'missing_rate': missing_rate,
                    'variance': None,
                    'type': None
                }
                
                # Determine column type
                if is_numeric_dtype(df[col]):
                    analysis['numeric_cols'].append(col)
                    analysis['stats'][col]['type'] = 'numeric'
                    try:
                        col_var = df[col].var()
                        if isinstance(df, dd.DataFrame):
                            col_var = col_var.compute()
                        analysis['stats'][col]['variance'] = col_var
                    except Exception:
                        analysis['stats'][col]['variance'] = None
                        
                elif is_datetime64_any_dtype(df[col]):
                    analysis['datetime_cols'].append(col)
                    analysis['stats'][col]['type'] = 'datetime'
                    
                elif is_string_dtype(df[col]):
                    # Distinguish between categorical and text based on unique ratio
                    unique_ratio = unique_count / max(total_count, 1)
                    if unique_ratio < 0.5:  # Less than 50% unique values
                        analysis['categorical_cols'].append(col)
                        analysis['stats'][col]['type'] = 'categorical'
                    else:
                        analysis['text_cols'].append(col)
                        analysis['stats'][col]['type'] = 'text'
                else:
                    # Default to categorical for other types
                    analysis['categorical_cols'].append(col)
                    analysis['stats'][col]['type'] = 'other'
                    
            except Exception as e:
                logger.warning(f"Error analyzing column {col}: {str(e)}")
                continue
        
        # Compute correlations for numeric columns
        if len(analysis['numeric_cols']) > 1:
            try:
                numeric_df = df_sample[analysis['numeric_cols']].select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    corr_matrix = numeric_df.corr()
                    analysis['correlations'] = corr_matrix.abs()
            except Exception as e:
                logger.warning(f"Error computing correlations: {str(e)}")
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error in analyze_dataset: {str(e)}")
        return {}

def automated_feature_engineering(df: Union[pd.DataFrame, dd.DataFrame], max_features: int = 50, preview: bool = False, target_col: Optional[str] = None) -> Tuple[Union[pd.DataFrame, dd.DataFrame], str]:
    """Enhanced automated feature engineering with advanced techniques and intelligent feature selection."""
    try:
        if not isinstance(df, (pd.DataFrame, dd.DataFrame)):
            return df, "Invalid DataFrame input"
        
        if df.empty:
            return df, "DataFrame is empty"
            
        # Validate parameters
        max_features = max(5, min(500, int(max_features)))
        
        # Check for duplicate columns
        if len(set(df.columns)) != len(df.columns):
            return df, "Duplicate column names detected; please resolve before processing"
        
        df_out = df if preview else df.copy()
        
        with st.spinner(f"🔍 Analyzing dataset and generating up to {max_features} intelligent features..."):
            # Advanced dataset analysis
            analysis = analyze_dataset_advanced(df_out, target_col)
            if not analysis:
                return df, "Error analyzing dataset"
            
            all_new_features = []
            feature_budget = max_features
            
            # Feature generation strategy based on data types and patterns
            if not preview:
                st.write("🧠 **Intelligent Feature Generation Strategy:**")
            
            # 1. Advanced numeric features (40% of budget)
            numeric_budget = int(feature_budget * 0.4)
            if analysis['numeric_cols'] and numeric_budget > 0:
                if not preview:
                    st.write(f"📊 Generating {numeric_budget} advanced numeric features...")
                df_out, numeric_features = create_advanced_numeric_features(df_out, analysis, numeric_budget)
                all_new_features.extend(numeric_features)
                feature_budget -= len(numeric_features)
            
            # 2. Advanced categorical features (25% of budget)
            categorical_budget = int(max_features * 0.25)
            if analysis['categorical_cols'] and categorical_budget > 0 and feature_budget > 0:
                if not preview:
                    st.write(f"🏷️ Generating {min(categorical_budget, feature_budget)} categorical features...")
                df_out, cat_features = create_advanced_categorical_features(df_out, analysis, min(categorical_budget, feature_budget))
                all_new_features.extend(cat_features)
                feature_budget -= len(cat_features)
            
            # 3. Advanced datetime features (20% of budget)
            datetime_budget = int(max_features * 0.2)
            if analysis['datetime_cols'] and datetime_budget > 0 and feature_budget > 0:
                if not preview:
                    st.write(f"📅 Generating {min(datetime_budget, feature_budget)} datetime features...")
                df_out, dt_features = create_advanced_datetime_features(df_out, analysis, min(datetime_budget, feature_budget))
                all_new_features.extend(dt_features)
                feature_budget -= len(dt_features)
            
            # 4. Advanced text features (10% of budget)
            text_budget = int(max_features * 0.1)
            if analysis['text_cols'] and text_budget > 0 and feature_budget > 0:
                if not preview:
                    st.write(f"📝 Generating {min(text_budget, feature_budget)} text features...")
                df_out, text_features = create_advanced_text_features(df_out, analysis, min(text_budget, feature_budget))
                all_new_features.extend(text_features)
                feature_budget -= len(text_features)
            
            # 5. Clustering features (3% of budget)
            if feature_budget > 0 and len(analysis['numeric_cols']) >= 2:
                if not preview:
                    st.write("🎯 Generating clustering-based features...")
                df_out, cluster_features = create_clustering_features(df_out, analysis, min(5, feature_budget))
                all_new_features.extend(cluster_features)
                feature_budget -= len(cluster_features)
            
            # 6. PCA features (2% of budget)
            if feature_budget > 0 and len(analysis['numeric_cols']) >= 3:
                if not preview:
                    st.write("🔍 Generating PCA-based features...")
                df_out, pca_features = create_pca_features(df_out, analysis, min(5, feature_budget))
                all_new_features.extend(pca_features)
                feature_budget -= len(pca_features)
            
            # Feature quality assessment and selection
            if all_new_features and not preview:
                st.write("🎯 Performing intelligent feature selection...")
                try:
                    # Remove low-variance features
                    low_var_features = []
                    for col in all_new_features:
                        if col in df_out.columns:
                            col_var = df_out[col].var()
                            if isinstance(df_out, dd.DataFrame):
                                col_var = col_var.compute()
                            if pd.isna(col_var) or col_var < FEATURE_CONFIG['min_variance_threshold']:
                                low_var_features.append(col)
                    
                    if low_var_features:
                        df_out = df_out.drop(columns=low_var_features)
                        all_new_features = [col for col in all_new_features if col not in low_var_features]
                        if not preview:
                            st.write(f"🗑️ Removed {len(low_var_features)} low-variance features")
                    
                    # Remove highly correlated features
                    if len(all_new_features) > 1:
                        numeric_new_features = [col for col in all_new_features if is_numeric_dtype(df_out[col])]
                        if len(numeric_new_features) > 1:
                            # Sample for correlation computation
                            if isinstance(df_out, dd.DataFrame):
                                corr_sample = sample_for_preview(df_out[numeric_new_features], n=5000)
                            else:
                                corr_sample = df_out[numeric_new_features] if len(df_out) <= 5000 else sample_for_preview(df_out[numeric_new_features], n=5000)
                            
                            if not corr_sample.empty:
                                corr_matrix = corr_sample.corr().abs()
                                upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                                upper_corr = corr_matrix.where(upper_triangle)
                                
                                high_corr_features = []
                                for column in upper_corr.columns:
                                    if any(upper_corr[column] > FEATURE_CONFIG['correlation_threshold']):
                                        high_corr_features.append(column)
                                
                                if high_corr_features:
                                    df_out = df_out.drop(columns=high_corr_features)
                                    all_new_features = [col for col in all_new_features if col not in high_corr_features]
                                    if not preview:
                                        st.write(f"🔗 Removed {len(high_corr_features)} highly correlated features")
                    
                    # Feature importance-based selection if target is provided
                    if target_col and target_col in df_out.columns and len(all_new_features) > max_features // 2:
                        if not preview:
                            st.write("🎯 Selecting top features based on importance...")
                        try:
                            # Calculate feature importance for new features
                            importance_scores = calculate_feature_importance(
                                sample_for_preview(df_out, n=5000) if len(df_out) > 5000 else df_out, 
                                target_col, 
                                {'numeric_cols': [col for col in all_new_features if is_numeric_dtype(df_out[col])]}
                            )
                            
                            if importance_scores:
                                # Sort features by importance
                                sorted_features = sorted(
                                    importance_scores.items(), 
                                    key=lambda x: x[1].get('mutual_info', 0) + x[1].get('rf_importance', 0), 
                                    reverse=True
                                )
                                
                                # Keep top features
                                top_features = [feat[0] for feat in sorted_features[:max_features//2]]
                                features_to_remove = [col for col in all_new_features if col in importance_scores and col not in top_features]
                                
                                if features_to_remove:
                                    df_out = df_out.drop(columns=features_to_remove)
                                    all_new_features = [col for col in all_new_features if col not in features_to_remove]
                                    if not preview:
                                        st.write(f"📈 Selected {len(top_features)} most important features")
                        except Exception as e:
                            logger.warning(f"Error in importance-based selection: {str(e)}")
                    
                except Exception as e:
                    logger.warning(f"Error in feature selection: {str(e)}")
        
        # Generate comprehensive report
        msg = f"🎉 Generated {len(all_new_features)} high-quality features"
        
        if all_new_features:
            # Categorize features by type
            feature_types = {
                'numeric_transformations': [f for f in all_new_features if any(prefix in f for prefix in ['log_', 'sqrt_', 'zscore_', 'reciprocal_'])],
                'interactions': [f for f in all_new_features if '_x_' in f or '_div_' in f],
                'categorical_encodings': [f for f in all_new_features if any(prefix in f for prefix in ['freq_', 'rank_', 'is_'])],
                'datetime_features': [f for f in all_new_features if any(suffix in f for suffix in ['_year', '_month', '_day', '_hour', '_weekend', '_sin', '_cos'])],
                'text_features': [f for f in all_new_features if any(prefix in f for prefix in ['len_', 'word_count_', 'digit_count_'])],
                'clustering_features': [f for f in all_new_features if 'cluster' in f],
                'pca_features': [f for f in all_new_features if 'pca_component' in f]
            }
            
            feature_summary = []
            for ftype, features in feature_types.items():
                if features:
                    feature_summary.append(f"{ftype.replace('_', ' ').title()}: {len(features)}")
            
            if feature_summary:
                msg += f"\n📋 Feature breakdown: {', '.join(feature_summary)}"
        
        logger.info(msg)
        
        if not preview:
            with session_lock:
                if 'pipeline' not in st.session_state:
                    st.session_state.pipeline = []
                st.session_state.pipeline.append({
                    "kind": "automated_feature_engineering", 
                    "params": {
                        "max_features": max_features, 
                        "features": all_new_features,
                        "target_col": target_col,
                        "feature_types": {k: len(v) for k, v in feature_types.items() if v}
                    }
                })
        
        return df_out, msg
        
    except ValueError as e:
        logger.error(f"ValueError in automated_feature_engineering: {str(e)}")
        return df, f"Error in automated feature engineering: {str(e)}"
    except MemoryError:
        logger.error("MemoryError in automated_feature_engineering")
        return df, "Error: Dataset too large for automated feature engineering"
    except Exception as e:
        logger.error(f"Unexpected error in automated_feature_engineering: {str(e)}")
        return df, f"Unexpected error: {str(e)}"

def plot_correlation_heatmap(df: Union[pd.DataFrame, dd.DataFrame], columns: Optional[List[str]] = None, threshold: float = 0.5):
    """Plot a correlation heatmap for selected numeric columns."""
    try:
        if df is None or df.empty:
            st.warning("No data available for correlation heatmap")
            return
            
        num_cols, _ = dtype_split(df)
        if not num_cols:
            st.warning("No numeric columns available for correlation heatmap")
            return
        
        # Filter columns if specified
        if columns:
            num_cols = [col for col in columns if col in num_cols]
        
        # Limit to reasonable number of columns for visualization
        num_cols = num_cols[:20]
        
        if not num_cols:
            st.warning("No valid numeric columns selected for correlation heatmap")
            return
        
        with st.spinner("Computing correlation heatmap..."):
            try:
                # Sample data if too large
                if isinstance(df, dd.DataFrame):
                    df_sample = sample_for_preview(df[num_cols], n=10000)
                else:
                    df_sample = df[num_cols] if len(df) <= 10000 else sample_for_preview(df[num_cols], n=10000)
                
                if df_sample.empty:
                    st.warning("No data available after sampling")
                    return
                
                # Compute correlation matrix
                corr_matrix = df_sample.corr()
                corr_matrix = corr_matrix.fillna(0)
                
                # Prepare data for Altair
                corr_data = corr_matrix.reset_index().melt(id_vars=['index'])
                corr_data.columns = ['var1', 'var2', 'correlation']
                
                # Add absolute correlation for filtering
                corr_data['abs_correlation'] = corr_data['correlation'].abs()
                
                # Create base chart
                base = alt.Chart(corr_data)
                
                # Create heatmap with conditional opacity
                heatmap = base.mark_rect().encode(
                    x=alt.X('var1:O', title='', axis=alt.Axis(labelAngle=-45)),
                    y=alt.Y('var2:O', title=''),
                    color=alt.Color(
                        'correlation:Q', 
                        scale=alt.Scale(scheme='redblue', domain=[-1, 1]),
                        title='Correlation'
                    ),
                    opacity=alt.condition(
                        alt.datum.abs_correlation >= threshold,
                        alt.value(1.0),
                        alt.value(0.3)
                    ),
                    tooltip=[
                        alt.Tooltip('var1:O', title='Variable 1'),
                        alt.Tooltip('var2:O', title='Variable 2'),
                        alt.Tooltip('correlation:Q', format='.3f', title='Correlation')
                    ]
                ).properties(
                    title=f'Correlation Heatmap (Highlighted: |Correlation| >= {threshold})',
                    width=min(600, 60 * len(num_cols)),
                    height=min(600, 60 * len(num_cols))
                )
                
                st.altair_chart(heatmap, use_container_width=True)
                
                # Show summary statistics
                high_corr_pairs = corr_data[corr_data['abs_correlation'] >= threshold]
                high_corr_pairs = high_corr_pairs[high_corr_pairs['var1'] != high_corr_pairs['var2']]  # Remove self-correlations
                
                if not high_corr_pairs.empty:
                    st.write(f"**High correlation pairs (|r| >= {threshold}):**")
                    summary_df = high_corr_pairs[['var1', 'var2', 'correlation']].sort_values('correlation', key=abs, ascending=False)
                    st.dataframe(summary_df.head(10), hide_index=True)
                else:
                    st.info(f"No correlation pairs found above threshold {threshold}")
                
                # Provide download option
                csv_buffer = io.StringIO()
                corr_matrix.to_csv(csv_buffer)
                csv_data = csv_buffer.getvalue()
                
                st.download_button(
                    label="📥 Download Correlation Matrix",
                    data=csv_data,
                    file_name=f"correlation_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                logger.error(f"Error creating correlation heatmap: {str(e)}")
                st.error(f"Error creating correlation heatmap: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error in plot_correlation_heatmap: {str(e)}")
        st.error(f"Error plotting correlation heatmap: {str(e)}")

def safe_eval_expression(df: Union[pd.DataFrame, dd.DataFrame], expression: str, new_col: str) -> Tuple[Optional[pd.Series], str]:
    """Safely evaluate a custom expression using pandas eval."""
    try:
        if not expression or not expression.strip():
            return None, "Expression cannot be empty"
        
        # Sanitize expression - remove potentially dangerous operations
        dangerous_keywords = ['import', 'exec', 'eval', 'open', 'file', '__', 'getattr', 'setattr', 'delattr']
        expression_lower = expression.lower()
        for keyword in dangerous_keywords:
            if keyword in expression_lower:
                return None, f"Expression contains forbidden keyword: {keyword}"
        
        # Check if expression references valid columns
        valid_cols = [col for col in df.columns if col in expression]
        if not valid_cols:
            return None, "Expression must reference valid column names"
        
        # Check for allowed operations
        allowed_ops = {'+', '-', '*', '/', '(', ')', 'log', 'exp', 'sin', 'cos', 'sqrt', 'abs'}
        # Simple check for mathematical operations
        has_valid_op = any(op in expression for op in ['+', '-', '*', '/'])
        if not has_valid_op:
            return None, "Expression must include at least one mathematical operation (+, -, *, /)"
        
        # Evaluate expression safely
        if isinstance(df, dd.DataFrame):
            # For Dask, compute a sample first to test
            df_sample = sample_for_preview(df, n=1000)
            result_sample = df_sample.eval(expression, engine='python')  # Use python engine for safety
            if len(result_sample) > 0:  # If sample works, apply to full dataset
                result = df.map_partitions(lambda x: x.eval(expression, engine='python'), meta=('result', 'float64'))
                return result, "Expression evaluated successfully"
            else:
                return None, "Expression evaluation failed on sample"
        else:
            result = df.eval(expression, engine='python')  # Use python engine for safety
            return result, "Expression evaluated successfully"
            
    except SyntaxError as e:
        return None, f"Syntax error in expression: {str(e)}"
    except KeyError as e:
        return None, f"Column not found: {str(e)}"
    except Exception as e:
        return None, f"Error evaluating expression: {str(e)}"

def export_dataframe(df: Union[pd.DataFrame, dd.DataFrame], columns: List[str]) -> Tuple[Optional[bytes], str]:
    """Export selected columns of the DataFrame as a CSV file."""
    try:
        if df is None or df.empty:
            return None, "DataFrame is empty"
            
        # Validate columns
        valid_columns = [c for c in columns if c in df.columns]
        if not valid_columns:
            return None, "No valid columns selected for export"
        
        with st.spinner("Exporting DataFrame..."):
            try:
                if isinstance(df, dd.DataFrame):
                    df_export = df[valid_columns].compute()
                else:
                    df_export = df[valid_columns]
                
                # Create CSV buffer
                csv_buffer = io.StringIO()
                df_export.to_csv(csv_buffer, index=True)
                csv_data = csv_buffer.getvalue().encode('utf-8')
                
                return csv_data, f"Exported {len(valid_columns)} columns ({len(df_export)} rows) as CSV"
                
            except MemoryError:
                return None, "Dataset too large for export. Try selecting fewer columns."
                
    except Exception as e:
        logger.error(f"Error in export_dataframe: {str(e)}")
        return None, f"Error exporting DataFrame: {str(e)}"

def section_feature_engineering():
    """Feature Engineering section with sub-tabs for different operations."""
    try:
        # Check if data is available
        if 'df' not in st.session_state or st.session_state.df is None:
            st.warning("⚠️ Please upload a dataset in the Upload section first.")
            return
        
        # Initialize session state variables
        with session_lock:
            if 'pipeline' not in st.session_state:
                st.session_state.pipeline = []
            if 'history' not in st.session_state:
                st.session_state.history = []
        
        st.header("🎨 Feature Engineering Studio")
        st.markdown("Create, transform, and select features to uncover hidden patterns in your data.")
        
        # Create tabs
        tabs = st.tabs([
            "🖌️ Feature Creation",
            "🔮 Feature Transformation", 
            "✂️ Feature Selection",
            "🤖 Automated Feature Engineering",
            "🧭 Feature Evaluation"
        ])
        
        df = st.session_state.df
        before_stats = compute_basic_stats(df)
        
        # Tab 1: Feature Creation
        with tabs[0]:
            st.subheader("🖌️ Feature Creation")
            st.markdown("Generate new features from existing data.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Polynomial Features**")
                numeric_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
                
                if not numeric_cols:
                    st.info("No numeric columns available for polynomial features.")
                else:
                    poly_cols = st.multiselect(
                        "Select numeric columns for polynomial features", 
                        numeric_cols, 
                        key="poly_cols",
                        help="Select columns to create polynomial and interaction features"
                    )
                    poly_degree = st.slider("Polynomial degree", 1, 5, 2, key="poly_degree")
                    
                    if st.button("🔍 Preview Polynomial Features", key="preview_poly"):
                        if poly_cols:
                            with st.spinner("Generating preview..."):
                                preview_df, preview_msg = create_polynomial_features(df, poly_cols, poly_degree, preview=True)
                                if "Error" not in preview_msg:
                                    st.success(preview_msg)
                                    st.write("**Preview of new features:**")
                                    preview_sample = sample_for_preview(preview_df)
                                    new_cols = [col for col in preview_sample.columns if col not in df.columns]
                                    if new_cols:
                                        st.dataframe(preview_sample[new_cols].head())
                                    
                                    if st.button("✅ Apply Polynomial Features", key="apply_poly"):
                                        df_new, msg = create_polynomial_features(df, poly_cols, poly_degree)
                                        if "Error" not in msg:
                                            with session_lock:
                                                st.session_state.df = df_new
                                            push_history(f"Applied polynomial features (degree={poly_degree})")
                                            st.success("✅ " + msg)
                                            st.experimental_rerun()
                                        else:
                                            st.error("❌ " + msg)
                                else:
                                    st.error("❌ " + preview_msg)
                        else:
                            st.warning("Please select at least one column.")
            
            with col2:
                st.markdown("**Datetime Features**")
                all_cols = list(df.columns)
                
                if not all_cols:
                    st.info("No columns available for datetime features.")
                else:
                    dt_cols = st.multiselect(
                        "Select datetime columns", 
                        all_cols, 
                        key="dt_cols",
                        help="Select columns containing datetime data"
                    )
                    dt_features = st.multiselect(
                        "Select features to extract", 
                        ["year", "month", "day", "hour", "minute", "second", "dayofweek", "quarter"], 
                        default=["year", "month", "day"],
                        key="dt_features"
                    )
                    
                    if st.button("🔍 Preview Datetime Features", key="preview_dt"):
                        if dt_cols and dt_features:
                            with st.spinner("Generating preview..."):
                                preview_df, preview_msg = extract_datetime_features(df, dt_cols, dt_features, preview=True)
                                if "Error" not in preview_msg:
                                    st.success(preview_msg)
                                    st.write("**Preview of new features:**")
                                    preview_sample = sample_for_preview(preview_df)
                                    new_cols = [col for col in preview_sample.columns if col not in df.columns]
                                    if new_cols:
                                        st.dataframe(preview_sample[new_cols].head())
                                    
                                    if st.button("✅ Apply Datetime Features", key="apply_dt"):
                                        df_new, msg = extract_datetime_features(df, dt_cols, dt_features)
                                        if "Error" not in msg:
                                            with session_lock:
                                                st.session_state.df = df_new
                                            push_history(f"Extracted datetime features: {', '.join(dt_features)}")
                                            st.success("✅ " + msg)
                                            st.experimental_rerun()
                                        else:
                                            st.error("❌ " + msg)
                                else:
                                    st.error("❌ " + preview_msg)
                        else:
                            st.warning("Please select columns and features.")
        
        # Tab 2: Feature Transformation
        with tabs[1]:
            st.subheader("🔮 Feature Transformation")
            st.markdown("Transform features to enhance model performance.")
            
            st.markdown("**Binning**")
            numeric_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
            
            if not numeric_cols:
                st.info("No numeric columns available for binning.")
            else:
                bin_cols = st.multiselect(
                    "Select numeric columns to bin", 
                    numeric_cols, 
                    key="bin_cols",
                    help="Convert continuous variables into discrete bins"
                )
                bins = st.slider("Number of bins", 2, 50, 10, key="bins")
                
                if st.button("🔍 Preview Binning", key="preview_bin"):
                    if bin_cols:
                        with st.spinner("Generating preview..."):
                            preview_df, preview_msg = bin_features(df, bin_cols, bins, preview=True)
                            if "Error" not in preview_msg:
                                st.success(preview_msg)
                                st.write("**Preview of binned features:**")
                                preview_sample = sample_for_preview(preview_df)
                                new_cols = [col for col in preview_sample.columns if col not in df.columns]
                                if new_cols:
                                    st.dataframe(preview_sample[new_cols].head())
                                
                                if st.button("✅ Apply Binning", key="apply_bin"):
                                    df_new, msg = bin_features(df, bin_cols, bins)
                                    if "Error" not in msg:
                                        with session_lock:
                                            st.session_state.df = df_new
                                        push_history(f"Binned columns into {bins} bins")
                                        st.success("✅ " + msg)
                                        st.experimental_rerun()
                                    else:
                                        st.error("❌ " + msg)
                            else:
                                st.error("❌ " + preview_msg)
                    else:
                        st.warning("Please select at least one column.")
        
        # Tab 3: Feature Selection
        with tabs[2]:
            st.subheader("✂️ Feature Selection")
            st.markdown("Select the most relevant features to reduce dimensionality.")
            
            st.markdown("**Correlation-based Selection**")
            corr_threshold = st.slider(
                "Correlation threshold", 
                0.1, 0.99, 0.8, 
                step=0.05, 
                key="corr_threshold",
                help="Remove features with correlation above this threshold"
            )
            
            if st.button("🔍 Preview Correlation Selection", key="preview_corr"):
                with st.spinner("Analyzing correlations..."):
                    preview_df, preview_msg = select_features_correlation(df, corr_threshold, preview=True)
                    if "Error" not in preview_msg:
                        st.success(preview_msg)
                        before_cols = set(df.columns)
                        after_cols = set(preview_df.columns)
                        removed_cols = before_cols - after_cols
                        
                        if removed_cols:
                            st.write(f"**Columns to be removed ({len(removed_cols)}):**")
                            st.write(", ".join(sorted(removed_cols)))
                        else:
                            st.info("No columns will be removed with this threshold.")
                        
                        if st.button("✅ Apply Correlation Selection", key="apply_corr"):
                            df_new, msg = select_features_correlation(df, corr_threshold)
                            if "Error" not in msg:
                                with session_lock:
                                    st.session_state.df = df_new
                                push_history(f"Selected features based on correlation (threshold={corr_threshold})")
                                st.success("✅ " + msg)
                                st.experimental_rerun()
                            else:
                                st.error("❌ " + msg)
                    else:
                        st.error("❌ " + preview_msg)
        
        # Tab 4: Automated Feature Engineering
        with tabs[3]:
            st.subheader("🤖 Automated Feature Engineering")
            st.markdown("Generate candidate features automatically based on dataset analysis.")
            
            col1, col2 = st.columns(2)
            with col1:
                max_features = st.slider(
                    "Maximum number of features to generate", 
                    1, 200, 50, 
                    key="max_features",
                    help="Limit the number of new features to prevent overfitting"
                )
            with col2:
                target_col = st.selectbox(
                    "Select target column (optional)", 
                    [None] + list(df.columns), 
                    key="target_col",
                    help="Target column for supervised feature selection"
                )
            
            if st.button("🔍 Preview Automated Features", key="preview_auto"):
                with st.spinner("Analyzing dataset and generating features..."):
                    preview_df, preview_msg = automated_feature_engineering(
                        df, max_features, preview=True, target_col=target_col
                    )
                    if "Error" not in preview_msg:
                        st.success(preview_msg)
                        
                        new_cols = [col for col in preview_df.columns if col not in df.columns]
                        if new_cols:
                            st.write(f"**New features generated ({len(new_cols)}):**")
                            preview_sample = sample_for_preview(preview_df)
                            st.dataframe(preview_sample[new_cols[:10]].head())  # Show first 10 features
                            
                            if len(new_cols) > 10:
                                st.info(f"... and {len(new_cols) - 10} more features")
                        
                        if st.button("✅ Apply Automated Features", key="apply_auto"):
                            df_new, msg = automated_feature_engineering(df, max_features, target_col=target_col)
                            if "Error" not in msg:
                                with session_lock:
                                    st.session_state.df = df_new
                                    if target_col:
                                        st.session_state.target_col = target_col
                                push_history(f"Generated automated features (max: {max_features})")
                                st.success("✅ " + msg)
                                st.experimental_rerun()
                            else:
                                st.error("❌ " + msg)
                    else:
                        st.error("❌ " + preview_msg)
        
        # Tab 5: Feature Evaluation
        with tabs[4]:
            st.subheader("🧭 Feature Evaluation")
            st.markdown("Evaluate feature importance and relationships to understand your dataset.")
            
            # Dataset Overview
            with st.expander("📊 Feature Dashboard", expanded=True):
                st.markdown("**Dataset Overview**")
                current_df = st.session_state.df
                after_stats = compute_basic_stats(current_df)
                comparison = compare_stats(before_stats, after_stats)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Features", comparison['n_columns_before'])
                with col2:
                    st.metric("Current Features", comparison['n_columns_after'])
                with col3:
                    delta = comparison['n_columns_after'] - comparison['n_columns_before']
                    st.metric("Net Change", delta, delta=delta)
                
                if comparison['added_columns']:
                    st.success(f"**Added Features ({len(comparison['added_columns'])}):** {', '.join(comparison['added_columns'][:5])}")
                    if len(comparison['added_columns']) > 5:
                        st.info(f"... and {len(comparison['added_columns']) - 5} more")
                
                if comparison['removed_columns']:
                    st.warning(f"**Removed Features ({len(comparison['removed_columns'])}):** {', '.join(comparison['removed_columns'][:5])}")
                    if len(comparison['removed_columns']) > 5:
                        st.info(f"... and {len(comparison['removed_columns']) - 5} more")
                
                # Feature count chart
                if comparison['n_columns_before'] != comparison['n_columns_after']:
                    chart_data = pd.DataFrame({
                        'Stage': ['Before', 'After'],
                        'Feature Count': [comparison['n_columns_before'], comparison['n_columns_after']]
                    })
                    chart = alt.Chart(chart_data).mark_bar(size=60).encode(
                        x=alt.X('Stage:O', title=''),
                        y=alt.Y('Feature Count:Q', title='Number of Features'),
                        color=alt.Color('Stage:O', scale=alt.Scale(range=['#1f77b4', '#ff7f0e']), legend=None),
                        tooltip=['Stage:O', 'Feature Count:Q']
                    ).properties(
                        title='Feature Count: Before vs After Engineering',
                        width=300,
                        height=200
                    )
                    st.altair_chart(chart, use_container_width=True)
            
            # Pipeline History
            with st.expander("📋 Pipeline History", expanded=False):
                if st.session_state.history:
                    history_df = pd.DataFrame(st.session_state.history)
                    history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
                    history_df = history_df.sort_values('timestamp', ascending=False)
                    st.dataframe(
                        history_df[['timestamp', 'message']], 
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No feature engineering steps applied yet.")
            
            # Correlation Heatmap
            with st.expander("🔗 Correlation Analysis", expanded=True):
                numeric_cols = [c for c in current_df.columns if is_numeric_dtype(current_df[c])]
                
                if not numeric_cols:
                    st.info("No numeric columns available for correlation analysis.")
                else:
                    corr_cols = st.multiselect(
                        "Select columns for correlation heatmap", 
                        numeric_cols, 
                        default=numeric_cols[:min(10, len(numeric_cols))],
                        key="corr_cols",
                        help="Select up to 20 columns for correlation visualization"
                    )
                    corr_threshold_viz = st.slider(
                        "Highlight correlations above", 
                        0.1, 0.9, 0.5, 
                        step=0.05, 
                        key="corr_heatmap_threshold"
                    )
                    
                    if corr_cols:
                        plot_correlation_heatmap(current_df, columns=corr_cols, threshold=corr_threshold_viz)
                    else:
                        st.warning("Please select at least one column.")
            
            # Custom Feature Expression
            with st.expander("✍️ Custom Feature Creator", expanded=False):
                st.markdown("**Create custom features using mathematical expressions**")
                st.info("Use column names and mathematical operators (+, -, *, /, log, exp, sin, cos, sqrt, abs)")
                
                expression = st.text_input(
                    "Enter custom feature expression", 
                    placeholder="e.g., col1 / (col2 + 1)",
                    key="feature_expression",
                    help="Example: log(column1) + sqrt(column2)"
                )
                
                if st.button("🧪 Test Expression", key="test_expr"):
                    if expression:
                        with st.spinner("Testing expression..."):
                            new_col = f"custom_feature_{uuid.uuid4().hex[:8]}"
                            result, msg = safe_eval_expression(current_df, expression, new_col)
                            if result is not None:
                                st.success("✅ " + msg)
                                st.write("**Preview of result:**")
                                if isinstance(result, pd.Series):
                                    preview_data = pd.DataFrame({
                                        'Original Index': result.index[:10],
                                        'Result': result.head(10)
                                    })
                                    st.dataframe(preview_data, hide_index=True)
                                
                                if st.button("➕ Add to Dataset", key="add_custom"):
                                    try:
                                        with session_lock:
                                            st.session_state.df[new_col] = result
                                        push_history(f"Added custom feature: {expression}")
                                        st.success(f"✅ Added custom feature as {new_col}")
                                        st.experimental_rerun()
                                    except Exception as e:
                                        st.error(f"❌ Error adding feature: {str(e)}")
                            else:
                                st.error("❌ " + msg)
                    else:
                        st.warning("Please enter an expression.")
            
            # Export Features
            with st.expander("💾 Export Dataset", expanded=False):
                st.markdown("**Download the transformed dataset**")
                
                export_cols = st.multiselect(
                    "Select columns to export", 
                    current_df.columns, 
                    default=list(current_df.columns),
                    key="export_cols",
                    help="Choose which columns to include in the export"
                )
                
                if export_cols:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Selected Columns", len(export_cols))
                    with col2:
                        total_rows = len(current_df)
                        if isinstance(current_df, dd.DataFrame):
                            total_rows = total_rows.compute() if hasattr(total_rows, 'compute') else total_rows
                        st.metric("Total Rows", total_rows)
                    
                    if st.button("📥 Download CSV", key="download_csv"):
                        with st.spinner("Preparing download..."):
                            csv_data, msg = export_dataframe(current_df, export_cols)
                            if csv_data:
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                st.download_button(
                                    label="📁 Download CSV File",
                                    data=csv_data,
                                    file_name=f"engineered_features_{timestamp}.csv",
                                    mime="text/csv"
                                )
                                st.success("✅ " + msg)
                                push_history(f"Exported {len(export_cols)} columns as CSV")
                            else:
                                st.error("❌ " + msg)
                else:
                    st.warning("Please select at least one column to export.")
    
    except Exception as e:
        logger.error(f"Error in section_feature_engineering: {str(e)}")
        st.error(f"An unexpected error occurred: {str(e)}")
