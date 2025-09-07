import streamlit as st
import pandas as pd
import dask.dataframe as dd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression, mutual_info_classif, f_regression, f_classif
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from dask_ml.preprocessing import PolynomialFeatures as DaskPolynomialFeatures
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, is_string_dtype
import altair as alt
import logging
from typing import List, Tuple, Dict, Any, Optional, Union
from datetime import datetime
import threading
import uuid
import io
import warnings
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import re

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

# Initialize feature metadata in session state
def initialize_session_state():
    """Initialize required session state variables."""
    if 'feature_metadata' not in st.session_state:
        st.session_state.feature_metadata = {}
    if 'undo_stack' not in st.session_state:
        st.session_state.undo_stack = []
    if 'redo_stack' not in st.session_state:
        st.session_state.redo_stack = []
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = []
    if 'history' not in st.session_state:
        st.session_state.history = []

def generate_unique_col_name(df: Union[pd.DataFrame, dd.DataFrame], base_name: str) -> str:
    """Generate a unique column name to prevent overwriting existing columns."""
    try:
        if base_name not in df.columns:
            return base_name
        
        counter = 1
        new_name = f"{base_name}_{counter}"
        while new_name in df.columns:
            counter += 1
            new_name = f"{base_name}_{counter}"
        return new_name
    except Exception as e:
        logger.warning(f"Error generating unique column name: {str(e)}")
        return f"{base_name}_{uuid.uuid4().hex[:4]}"

def save_state_for_undo():
    """Save current state to undo stack."""
    try:
        with session_lock:
            if 'df' in st.session_state:
                state = {
                    'df': st.session_state.df.copy() if hasattr(st.session_state.df, 'copy') else st.session_state.df,
                    'pipeline': st.session_state.pipeline.copy(),
                    'feature_metadata': st.session_state.feature_metadata.copy()
                }
                st.session_state.undo_stack.append(state)
                # Keep only last 10 states
                if len(st.session_state.undo_stack) > 10:
                    st.session_state.undo_stack = st.session_state.undo_stack[-10:]
                # Clear redo stack when new action is performed
                st.session_state.redo_stack = []
    except Exception as e:
        logger.warning(f"Error saving state for undo: {str(e)}")

def undo_last_action():
    """Undo the last feature engineering action."""
    try:
        with session_lock:
            if st.session_state.undo_stack:
                # Save current state to redo stack
                current_state = {
                    'df': st.session_state.df.copy() if hasattr(st.session_state.df, 'copy') else st.session_state.df,
                    'pipeline': st.session_state.pipeline.copy(),
                    'feature_metadata': st.session_state.feature_metadata.copy()
                }
                st.session_state.redo_stack.append(current_state)
                
                # Restore previous state
                last_state = st.session_state.undo_stack.pop()
                st.session_state.df = last_state['df']
                st.session_state.pipeline = last_state['pipeline']
                st.session_state.feature_metadata = last_state['feature_metadata']
                
                push_history("Undid last action")
                return True
    except Exception as e:
        logger.error(f"Error in undo: {str(e)}")
    return False

def redo_last_action():
    """Redo the last undone feature engineering action."""
    try:
        with session_lock:
            if st.session_state.redo_stack:
                # Save current state to undo stack
                current_state = {
                    'df': st.session_state.df.copy() if hasattr(st.session_state.df, 'copy') else st.session_state.df,
                    'pipeline': st.session_state.pipeline.copy(),
                    'feature_metadata': st.session_state.feature_metadata.copy()
                }
                st.session_state.undo_stack.append(current_state)
                
                # Restore redo state
                redo_state = st.session_state.redo_stack.pop()
                st.session_state.df = redo_state['df']
                st.session_state.pipeline = redo_state['pipeline']
                st.session_state.feature_metadata = redo_state['feature_metadata']
                
                push_history("Redid last action")
                return True
    except Exception as e:
        logger.error(f"Error in redo: {str(e)}")
    return False

def export_pipeline_as_code(pipeline: List[Dict]) -> str:
    """Generate Python code that reproduces the feature engineering pipeline."""
    try:
        code_lines = [
            "# Generated Feature Engineering Pipeline",
            "import pandas as pd",
            "import numpy as np",
            "from sklearn.preprocessing import PolynomialFeatures, StandardScaler",
            "from sklearn.cluster import KMeans",
            "from sklearn.decomposition import PCA",
            "import warnings",
            "warnings.filterwarnings('ignore')",
            "",
            "def apply_feature_engineering_pipeline(df):",
            "    \"\"\"Apply the complete feature engineering pipeline to a DataFrame.\"\"\"",
            "    df_processed = df.copy()",
            "    print(f'Starting with {len(df_processed.columns)} columns')",
            ""
        ]
        
        for i, step in enumerate(pipeline):
            kind = step.get('kind', '')
            params = step.get('params', {})
            
            code_lines.append(f"    # Step {i+1}: {kind}")
            
            if kind == 'create_polynomial_features':
                columns = params.get('columns', [])
                degree = params.get('degree', 2)
                code_lines.extend([
                    f"    # Polynomial Features (degree={degree})",
                    f"    poly_cols = {columns}",
                    f"    if all(col in df_processed.columns for col in poly_cols):",
                    f"        poly = PolynomialFeatures(degree={degree}, include_bias=False)",
                    f"        poly_data = df_processed[poly_cols].fillna(df_processed[poly_cols].median())",
                    f"        poly_features = poly.fit_transform(poly_data)",
                    f"        feature_names = poly.get_feature_names_out(poly_cols)",
                    f"        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df_processed.index)",
                    f"        new_cols = [col for col in feature_names if col not in poly_cols]",
                    f"        for col in new_cols:",
                    f"            df_processed[col] = poly_df[col]",
                    f"        print(f'Added {{len(new_cols)}} polynomial features')",
                    ""
                ])
            
            elif kind == 'extract_datetime_features':
                columns = params.get('columns', [])
                features = params.get('features', [])
                code_lines.extend([
                    f"    # Datetime Features",
                    f"    dt_cols = {columns}",
                    f"    dt_features = {features}",
                    f"    for col in dt_cols:",
                    f"        if col in df_processed.columns:",
                    f"            df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')",
                    f"            for feature in dt_features:",
                    f"                new_col = f'{{col}}_{{feature}}'",
                    f"                try:",
                    f"                    df_processed[new_col] = getattr(df_processed[col].dt, feature)",
                    f"                except AttributeError:",
                    f"                    pass",
                    f"    print(f'Extracted datetime features')",
                    ""
                ])
            
            elif kind == 'bin_features':
                columns = params.get('columns', [])
                bins = params.get('bins', 10)
                code_lines.extend([
                    f"    # Binning Features",
                    f"    bin_cols = {columns}",
                    f"    for col in bin_cols:",
                    f"        if col in df_processed.columns and df_processed[col].dtype in ['int64', 'float64']:",
                    f"            try:",
                    f"                df_processed[f'{{col}}_binned'] = pd.qcut(df_processed[col], q={bins}, labels=False, duplicates='drop')",
                    f"            except ValueError:",
                    f"                df_processed[f'{{col}}_binned'] = pd.cut(df_processed[col], bins={bins}, labels=False, include_lowest=True)",
                    f"    print(f'Applied binning to features')",
                    ""
                ])
            
            elif kind == 'select_features_correlation':
                threshold = params.get('threshold', 0.8)
                code_lines.extend([
                    f"    # Correlation-based Feature Selection",
                    f"    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()",
                    f"    if len(numeric_cols) > 1:",
                    f"        corr_matrix = df_processed[numeric_cols].corr().abs()",
                    f"        upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)",
                    f"        upper_corr = corr_matrix.where(upper_triangle)",
                    f"        to_drop = [col for col in upper_corr.columns if any(upper_corr[col] > {threshold})]",
                    f"        df_processed = df_processed.drop(columns=to_drop)",
                    f"        print(f'Dropped {{len(to_drop)}} highly correlated features')",
                    ""
                ])
            
            elif kind == 'automated_feature_engineering':
                max_features = params.get('max_features', 50)
                code_lines.extend([
                    f"    # Automated Feature Engineering",
                    f"    # Note: This is a simplified version of the automated process",
                    f"    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()",
                    f"    ",
                    f"    # Log transformations for skewed columns",
                    f"    for col in numeric_cols[:5]:  # Limit to prevent explosion",
                    f"        try:",
                    f"            if df_processed[col].skew() > 1:",
                    f"                df_processed[f'log_{{col}}'] = np.log1p(np.maximum(df_processed[col], 0))",
                    f"        except:",
                    f"            continue",
                    f"    ",
                    f"    print(f'Applied automated feature engineering (max {max_features} features)')",
                    ""
                ])
        
        code_lines.extend([
            "    print(f'Finished with {len(df_processed.columns)} columns')",
            "    return df_processed",
            "",
            "# Usage example:",
            "# df_engineered = apply_feature_engineering_pipeline(your_dataframe)",
            "# print(df_engineered.head())"
        ])
        
        return "\n".join(code_lines)
        
    except Exception as e:
        logger.error(f"Error generating pipeline code: {str(e)}")
        return f"# Error generating pipeline code: {str(e)}"

def evaluate_feature_impact(df: pd.DataFrame, target_col: str, new_features: List[str]) -> Optional[pd.DataFrame]:
    """Evaluate the impact of new features on model performance."""
    try:
        if not target_col or target_col not in df.columns or not new_features:
            return None
        
        # Prepare feature sets
        all_features = [col for col in df.columns if col != target_col]
        original_features = [col for col in all_features if col not in new_features]
        
        if len(original_features) == 0:
            return None
        
        # Sample data if too large
        df_sample = sample_for_preview(df, n=5000) if len(df) > 5000 else df
        
        # Check if target is classification or regression
        target_nunique = df_sample[target_col].nunique()
        is_classification = target_nunique < 20
        
        results = []
        
        # Evaluate original features
        X_orig = df_sample[original_features].fillna(df_sample[original_features].median())
        y = df_sample[target_col].dropna()
        common_idx = X_orig.index.intersection(y.index)
        
        if len(common_idx) >= 10:
            X_orig = X_orig.loc[common_idx]
            y = y.loc[common_idx]
            
            if is_classification:
                model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
            
            try:
                orig_score = cross_val_score(model, X_orig, y, cv=3).mean()
                results.append({
                    'Feature Set': 'Original Features',
                    'Number of Features': len(original_features),
                    'CV Score': orig_score,
                    'Score Type': 'Accuracy' if is_classification else 'R² Score'
                })
            except Exception as e:
                logger.warning(f"Error evaluating original features: {str(e)}")
        
        # Evaluate with new features
        new_feature_cols = [col for col in new_features if col in df_sample.columns]
        if new_feature_cols:
            X_new = df_sample[original_features + new_feature_cols].fillna(
                df_sample[original_features + new_feature_cols].median()
            )
            X_new = X_new.loc[common_idx]
            
            try:
                new_score = cross_val_score(model, X_new, y, cv=3).mean()
                results.append({
                    'Feature Set': 'With New Features',
                    'Number of Features': len(original_features) + len(new_feature_cols),
                    'CV Score': new_score,
                    'Score Type': 'Accuracy' if is_classification else 'R² Score'
                })
                
                # Add improvement calculation
                if len(results) == 2:
                    improvement = new_score - orig_score
                    results.append({
                        'Feature Set': 'Improvement',
                        'Number of Features': len(new_feature_cols),
                        'CV Score': improvement,
                        'Score Type': 'Δ Score'
                    })
                    
            except Exception as e:
                logger.warning(f"Error evaluating new features: {str(e)}")
        
        return pd.DataFrame(results) if results else None
        
    except Exception as e:
        logger.error(f"Error in feature impact analysis: {str(e)}")
        return None

def generate_smart_recommendations(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate smart feature engineering recommendations based on dataset analysis."""
    try:
        recommendations = []
        
        # Recommendations for skewed numeric columns
        if analysis.get('numeric_cols'):
            for col in analysis['numeric_cols']:
                if col in analysis.get('stats', {}):
                    skewness = analysis['stats'][col].get('skewness', 0)
                    if skewness > 1:
                        recommendations.append({
                            'type': 'transformation',
                            'priority': 'high',
                            'icon': '📊',
                            'title': f'Apply log transformation to {col}',
                            'description': f'High skewness detected ({skewness:.2f}). Log transformation can normalize the distribution.',
                            'action': 'log_transform',
                            'columns': [col]
                        })
                    elif 0.5 < skewness <= 1:
                        recommendations.append({
                            'type': 'transformation',
                            'priority': 'medium',
                            'icon': '📈',
                            'title': f'Consider sqrt transformation for {col}',
                            'description': f'Moderate skewness detected ({skewness:.2f}). Square root transformation may help.',
                            'action': 'sqrt_transform',
                            'columns': [col]
                        })
        
        # Recommendations for datetime columns
        if analysis.get('datetime_cols'):
            for col in analysis['datetime_cols']:
                recommendations.append({
                    'type': 'feature_creation',
                    'priority': 'high',
                    'icon': '📅',
                    'title': f'Extract datetime features from {col}',
                    'description': 'Create year, month, day, and cyclical features to capture temporal patterns.',
                    'action': 'datetime_features',
                    'columns': [col]
                })
        
        # Recommendations for high cardinality categorical columns
        if analysis.get('high_cardinality_cols'):
            for col in analysis['high_cardinality_cols']:
                recommendations.append({
                    'type': 'encoding',
                    'priority': 'medium',
                    'icon': '🏷️',
                    'title': f'Apply frequency encoding to {col}',
                    'description': f'High cardinality detected. Frequency encoding can reduce dimensionality.',
                    'action': 'frequency_encoding',
                    'columns': [col]
                })
        
        # Recommendations for polynomial features
        numeric_cols = analysis.get('numeric_cols', [])
        if len(numeric_cols) >= 2:
            high_var_cols = [col for col in numeric_cols 
                           if col in analysis.get('stats', {}) and 
                           analysis['stats'][col].get('variance', 0) > FEATURE_CONFIG['min_variance_threshold']]
            if len(high_var_cols) >= 2:
                recommendations.append({
                    'type': 'feature_creation',
                    'priority': 'medium',
                    'icon': '🔢',
                    'title': 'Create polynomial features',
                    'description': f'Generate interaction terms for {len(high_var_cols)} numeric columns.',
                    'action': 'polynomial_features',
                    'columns': high_var_cols[:5]  # Limit to prevent explosion
                })
        
        # Recommendations for highly correlated features
        if 'high_correlations' in analysis and analysis['high_correlations']:
            high_corr_count = len(analysis['high_correlations'])
            if high_corr_count > 0:
                recommendations.append({
                    'type': 'feature_selection',
                    'priority': 'high',
                    'icon': '✂️',
                    'title': 'Remove highly correlated features',
                    'description': f'{high_corr_count} pairs of highly correlated features detected. Consider removal to reduce multicollinearity.',
                    'action': 'correlation_selection',
                    'columns': []
                })
        
        # Recommendations for text columns
        if analysis.get('text_cols'):
            for col in analysis['text_cols']:
                patterns = analysis.get('patterns', {}).get(col, {})
                avg_length = patterns.get('avg_length', 0)
                if avg_length > 10:
                    recommendations.append({
                        'type': 'feature_creation',
                        'priority': 'medium',
                        'icon': '📝',
                        'title': f'Extract text features from {col}',
                        'description': f'Create length, word count, and character-based features.',
                        'action': 'text_features',
                        'columns': [col]
                    })
        
        # Sort by priority
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 0), reverse=True)
        
        return recommendations[:8]  # Return top 8 recommendations
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        return []

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
    """Sample DataFrame for preview, with caching."""
    try:
        cache_key = f"cached_sample_{id(df)}_{n}"
        if cache_key in st.session_state:
            return st.session_state[cache_key]
        
        if isinstance(df, dd.DataFrame):
            total_rows = len(df)
            if hasattr(total_rows, 'compute'):
                total_rows = total_rows.compute()
            if total_rows == 0:
                return pd.DataFrame()
            frac = min(1.0, n / max(total_rows, 1))
            sample = df.sample(frac=frac).compute()
            st.session_state[cache_key] = sample
            return sample
        sample = df.sample(n=min(n, len(df))) if len(df) > 0 else df
        st.session_state[cache_key] = sample
        return sample
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
        ).properties(title=f"Histogram of {title}")
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
                'message': str(message)[:500],
                'timestamp': datetime.now().isoformat()
            })
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
            return {
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
            
        df_sample = sample_for_preview(df, n=sample_size)
        if df_sample.empty:
            return {
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
                
                analysis['data_quality'][col] = {
                    'completeness': 1 - missing_rate,
                    'uniqueness': unique_count / max(total_rows, 1),
                    'consistency': 1.0
                }
                
                if is_numeric_dtype(df_sample[col]):
                    analysis['numeric_cols'].append(col)
                    analysis['stats'][col]['type'] = 'numeric'
                    col_data = df_sample[col].dropna()
                    if len(col_data) > 1:
                        analysis['stats'][col]['variance'] = col_data.var()
                        analysis['stats'][col]['skewness'] = stats.skew(col_data)
                        analysis['stats'][col]['kurtosis'] = stats.kurtosis(col_data)
                        if unique_count == 2:
                            analysis['binary_cols'].append(col)
                        
                elif is_datetime64_any_dtype(df_sample[col]):
                    analysis['datetime_cols'].append(col)
                    analysis['stats'][col]['type'] = 'datetime'
                    dt_data = df_sample[col].dropna()
                    if len(dt_data) > 1:
                        date_range = dt_data.max() - dt_data.min()
                        analysis['patterns'][col] = {
                            'date_range_days': date_range.days if hasattr(date_range, 'days') else None,
                            'has_time_component': dt_data.dt.hour.nunique() > 1,
                            'frequency_pattern': 'irregular'
                        }
                        
                elif is_string_dtype(df_sample[col]) or df_sample[col].dtype == 'object':
                    unique_ratio = unique_count / max(total_rows, 1)
                    avg_length = df_sample[col].astype(str).str.len().mean()
                    
                    if unique_count > FEATURE_CONFIG['max_categorical_cardinality']:
                        analysis['high_cardinality_cols'].append(col)
                    
                    if unique_ratio > 0.8 or avg_length > 20:
                        analysis['text_cols'].append(col)
                        analysis['stats'][col]['type'] = 'text'
                        analysis['patterns'][col] = {
                            'avg_length': avg_length,
                            'has_numbers': df_sample[col].astype(str).str.contains(r'\d').any(),
                            'has_special_chars': df_sample[col].astype(str).str.contains(r'[^a-zA-Z0-9\s]').any(),
                            'word_count_avg': df_sample[col].astype(str).str.split().str.len().mean()
                        }
                    else:
                        analysis['categorical_cols'].append(col)
                        analysis['stats'][col]['type'] = 'categorical'
                        if unique_count == 2:
                            analysis['binary_cols'].append(col)
                else:
                    analysis['categorical_cols'].append(col)
                    analysis['stats'][col]['type'] = 'other'
                    
            except Exception as e:
                logger.warning(f"Error analyzing column {col}: {str(e)}")
                continue
        
        if target_col and target_col in df_sample.columns:
            try:
                analysis['feature_importance'] = calculate_feature_importance(df_sample, target_col, analysis)
            except Exception as e:
                logger.warning(f"Error calculating feature importance: {str(e)}")
        
        if len(analysis['numeric_cols']) > 1:
            try:
                numeric_df = df_sample[analysis['numeric_cols']].select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    corr_matrix = numeric_df.corr()
                    analysis['correlations'] = corr_matrix.abs()
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
        return {
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

def calculate_feature_importance(df: pd.DataFrame, target_col: str, analysis: Dict) -> Dict[str, float]:
    """Calculate feature importance using multiple methods."""
    try:
        if target_col not in df.columns:
            return {}
        
        target = df[target_col].dropna()
        if len(target) == 0:
            return {}
        
        is_classification = (
            target_col in analysis['categorical_cols'] or 
            target_col in analysis['binary_cols'] or
            target.nunique() < 20
        )
        
        importance_scores = {}
        numeric_features = [col for col in analysis['numeric_cols'] if col != target_col]
        
        if numeric_features:
            X = df[numeric_features].fillna(df[numeric_features].median())
            y = target
            
            common_idx = X.index.intersection(y.index)
            if len(common_idx) == 0:
                logger.warning("No common indices between features and target")
                return {}
            X = X.loc[common_idx]
            y = y.loc[common_idx]
            
            if len(X) > 10:
                try:
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
        
        for col in high_var_cols[:min(5, len(high_var_cols))]:
            if feature_count >= max_features:
                break
            
            try:
                col_data = df_out[col]
                
                if analysis['stats'][col].get('skewness', 0) > 1:
                    new_col = generate_unique_col_name(df_out, f"log_{col}")
                    if isinstance(df_out, dd.DataFrame):
                        df_out[new_col] = col_data.map_partitions(
                            lambda s: np.log1p(np.maximum(s, 0)), 
                            meta=(new_col, 'float64')
                        )
                    else:
                        df_out[new_col] = np.log1p(np.maximum(col_data, 0))
                    new_features.append(new_col)
                    if 'feature_metadata' not in st.session_state:
                        st.session_state.feature_metadata = {}
                    st.session_state.feature_metadata[new_col] = {
                        'source_columns': [col], 'transformation': 'log'
                    }
                    feature_count += 1
                
                if 0.5 < analysis['stats'][col].get('skewness', 0) <= 1 and feature_count < max_features:
                    new_col = f"sqrt_{col}_{uuid.uuid4().hex[:4]}" if f"sqrt_{col}" in df_out.columns else f"sqrt_{col}"
                    if isinstance(df_out, dd.DataFrame):
                        df_out[new_col] = col_data.map_partitions(
                            lambda s: np.sqrt(np.maximum(s, 0)), 
                            meta=(new_col, 'float64')
                        )
                    else:
                        df_out[new_col] = np.sqrt(np.maximum(col_data, 0))
                    new_features.append(new_col)
                    st.session_state.feature_metadata[new_col] = {
                        'source_columns': [col], 'transformation': 'sqrt'
                    }
                    feature_count += 1
                
                if feature_count < max_features:
                    new_col = f"reciprocal_{col}_{uuid.uuid4().hex[:4]}" if f"reciprocal_{col}" in df_out.columns else f"reciprocal_{col}"
                    if isinstance(df_out, dd.DataFrame):
                        df_out[new_col] = col_data.map_partitions(
                            lambda s: s.where(s.abs() > 1e-6, 1e-6 * s.sign()), 
                            meta=(new_col, 'float64')
                        )
                    else:
                        df_out[new_col] = col_data.where(col_data.abs() > 1e-6, 1e-6 * col_data.sign())
                    new_features.append(new_col)
                    st.session_state.feature_metadata[new_col] = {
                        'source_columns': [col], 'transformation': 'reciprocal'
                    }
                    feature_count += 1
                    
            except Exception as e:
                logger.warning(f"Error creating transformations for {col}: {str(e)}")
        
        if len(high_var_cols) >= 2 and feature_count < max_features:
            try:
                for col in high_var_cols[:3]:
                    if feature_count >= max_features:
                        break
                    new_col = f"zscore_{col}_{uuid.uuid4().hex[:4]}" if f"zscore_{col}" in df_out.columns else f"zscore_{col}"
                    if isinstance(df_out, dd.DataFrame):
                        col_mean, col_std = dd.compute(df_out[col].mean(), df_out[col].std())
                        df_out[new_col] = (df_out[col] - col_mean) / (col_std + 1e-8)
                    else:
                        col_mean = df_out[col].mean()
                        col_std = df_out[col].std()
                        df_out[new_col] = (df_out[col] - col_mean) / (col_std + 1e-8)
                    new_features.append(new_col)
                    st.session_state.feature_metadata[new_col] = {
                        'source_columns': [col], 'transformation': 'zscore'
                    }
                    feature_count += 1
                    
            except Exception as e:
                logger.warning(f"Error creating statistical features: {str(e)}")
        
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
                            if 0.3 <= corr_val <= 0.8:
                                new_col = f"{col1}_x_{col2}_{uuid.uuid4().hex[:4]}" if f"{col1}_x_{col2}" in df_out.columns else f"{col1}_x_{col2}"
                                df_out[new_col] = df_out[col1] * df_out[col2]
                                new_features.append(new_col)
                                st.session_state.feature_metadata[new_col] = {
                                    'source_columns': [col1, col2], 'transformation': 'multiplication'
                                }
                                feature_count += 1
                                interaction_count += 1
                                
                                if feature_count < max_features:
                                    new_col = f"{col1}_div_{col2}_{uuid.uuid4().hex[:4]}" if f"{col1}_div_{col2}" in df_out.columns else f"{col1}_div_{col2}"
                                    if isinstance(df_out, dd.DataFrame):
                                        df_out[new_col] = (df_out[col1] / df_out[col2].map_partitions(
                                            lambda s: s.where(s.abs() > 1e-6, 1e-6 * s.sign()), meta=(new_col, 'float64')
                                        ))
                                    else:
                                        df_out[new_col] = df_out[col1] / df_out[col2].where(df_out[col2].abs() > 1e-6, 1e-6 * df_out[col2].sign())
                                    new_features.append(new_col)
                                    st.session_state.feature_metadata[new_col] = {
                                        'source_columns': [col1, col2], 'transformation': 'division'
                                    }
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
                new_col = f"freq_{col}_{uuid.uuid4().hex[:4]}" if f"freq_{col}" in df_out.columns else f"freq_{col}"
                if isinstance(df_out, dd.DataFrame):
                    freq_map = df_out[col].value_counts().compute().to_dict()
                    df_out[new_col] = df_out[col].map(freq_map, meta=(new_col, 'int64'))
                else:
                    freq_map = df_out[col].value_counts().to_dict()
                    df_out[new_col] = df_out[col].map(freq_map)
                new_features.append(new_col)
                st.session_state.feature_metadata[new_col] = {
                    'source_columns': [col], 'transformation': 'frequency_encoding'
                }
                feature_count += 1
                
                if feature_count < max_features:
                    new_col = f"rank_{col}_{uuid.uuid4().hex[:4]}" if f"rank_{col}" in df_out.columns else f"rank_{col}"
                    if isinstance(df_out, dd.DataFrame):
                        rank_map = df_out[col].value_counts().compute().rank(ascending=False).to_dict()
                        df_out[new_col] = df_out[col].map(rank_map, meta=(new_col, 'float64'))
                    else:
                        rank_map = df_out[col].value_counts().rank(ascending=False).to_dict()
                        df_out[new_col] = df_out[col].map(rank_map)
                    new_features.append(new_col)
                    st.session_state.feature_metadata[new_col] = {
                        'source_columns': [col], 'transformation': 'rank_encoding'
                    }
                    feature_count += 1
                
                unique_count = analysis['stats'][col]['unique_count']
                if 3 <= unique_count <= 10 and feature_count < max_features:
                    if isinstance(df_out, dd.DataFrame):
                        top_categories = df_out[col].value_counts().compute().head(3).index
                    else:
                        top_categories = df_out[col].value_counts().head(3).index
                    
                    for category in top_categories:
                        if feature_count >= max_features:
                            break
                        new_col = f"is_{col}_{str(category)[:10]}_{uuid.uuid4().hex[:4]}" if f"is_{col}_{str(category)[:10]}" in df_out.columns else f"is_{col}_{str(category)[:10]}"
                        df_out[new_col] = (df_out[col] == category).astype(int)
                        new_features.append(new_col)
                        st.session_state.feature_metadata[new_col] = {
                            'source_columns': [col], 'transformation': f'binary_encoding_{category}'
                        }
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
                if not is_datetime64_any_dtype(df_out[col]):
                    if isinstance(df_out, dd.DataFrame):
                        df_out[col] = dd.to_datetime(df_out[col], errors='coerce')
                    else:
                        df_out[col] = pd.to_datetime(df_out[col], errors='coerce')
                
                basic_features = ["year", "month", "day", "dayofweek", "hour", "quarter"]
                for feature in basic_features:
                    if feature_count >= max_features:
                        break
                    new_col = f"{col}_{feature}_{uuid.uuid4().hex[:4]}" if f"{col}_{feature}" in df_out.columns else f"{col}_{feature}"
                    try:
                        if isinstance(df_out, dd.DataFrame):
                            df_out[new_col] = getattr(df_out[col].dt, feature)
                        else:
                            df_out[new_col] = getattr(df_out[col].dt, feature)
                        new_features.append(new_col)
                        st.session_state.feature_metadata[new_col] = {
                            'source_columns': [col], 'transformation': f'datetime_{feature}'
                        }
                        feature_count += 1
                    except AttributeError:
                        continue
                
                if feature_count < max_features:
                    new_col = f"{col}_is_weekend_{uuid.uuid4().hex[:4]}" if f"{col}_is_weekend" in df_out.columns else f"{col}_is_weekend"
                    if isinstance(df_out, dd.DataFrame):
                        df_out[new_col] = (df_out[col].dt.dayofweek >= 5).astype(int)
                    else:
                        df_out[new_col] = (df_out[col].dt.dayofweek >= 5).astype(int)
                    new_features.append(new_col)
                    st.session_state.feature_metadata[new_col] = {
                        'source_columns': [col], 'transformation': 'is_weekend'
                    }
                    feature_count += 1
                
                if feature_count < max_features:
                    new_col = f"{col}_is_month_end_{uuid.uuid4().hex[:4]}" if f"{col}_is_month_end" in df_out.columns else f"{col}_is_month_end"
                    if isinstance(df_out, dd.DataFrame):
                        df_out[new_col] = df_out[col].dt.is_month_end.astype(int)
                    else:
                        df_out[new_col] = df_out[col].dt.is_month_end.astype(int)
                    new_features.append(new_col)
                    st.session_state.feature_metadata[new_col] = {
                        'source_columns': [col], 'transformation': 'is_month_end'
                    }
                    feature_count += 1
                
                if feature_count < max_features - 1:
                    new_col_sin = f"{col}_month_sin_{uuid.uuid4().hex[:4]}" if f"{col}_month_sin" in df_out.columns else f"{col}_month_sin"
                    new_col_cos = f"{col}_month_cos_{uuid.uuid4().hex[:4]}" if f"{col}_month_cos" in df_out.columns else f"{col}_month_cos"
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
                    st.session_state.feature_metadata[new_col_sin] = {
                        'source_columns': [col], 'transformation': 'month_sin'
                    }
                    st.session_state.feature_metadata[new_col_cos] = {
                        'source_columns': [col], 'transformation': 'month_cos'
                    }
                    feature_count += 2
                
                if feature_count < max_features:
                    new_col = f"{col}_timestamp_{uuid.uuid4().hex[:4]}" if f"{col}_timestamp" in df_out.columns else f"{col}_timestamp"
                    if isinstance(df_out, dd.DataFrame):
                        df_out[new_col] = df_out[col].map_partitions(
                            lambda s: s.astype('int64') // 10**9, 
                            meta=(new_col, 'int64')
                        )
                    else:
                        df_out[new_col] = df_out[col].astype('int64') // 10**9
                    new_features.append(new_col)
                    st.session_state.feature_metadata[new_col] = {
                        'source_columns': [col], 'transformation': 'timestamp'
                    }
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
                new_col = f"len_{col}_{uuid.uuid4().hex[:4]}" if f"len_{col}" in df_out.columns else f"len_{col}"
                if isinstance(df_out, dd.DataFrame):
                    df_out[new_col] = df_out[col].map_partitions(
                        lambda s: s.astype(str).str.len(), 
                        meta=(new_col, 'int64')
                    )
                else:
                    df_out[new_col] = df_out[col].astype(str).str.len()
                new_features.append(new_col)
                st.session_state.feature_metadata[new_col] = {
                    'source_columns': [col], 'transformation': 'length'
                }
                feature_count += 1
                
                if feature_count < max_features:
                    new_col = f"word_count_{col}_{uuid.uuid4().hex[:4]}" if f"word_count_{col}" in df_out.columns else f"word_count_{col}"
                    if isinstance(df_out, dd.DataFrame):
                        df_out[new_col] = df_out[col].map_partitions(
                            lambda s: s.astype(str).str.split().str.len(), 
                            meta=(new_col, 'int64')
                        )
                    else:
                        df_out[new_col] = df_out[col].astype(str).str.split().str.len()
                    new_features.append(new_col)
                    st.session_state.feature_metadata[new_col] = {
                        'source_columns': [col], 'transformation': 'word_count'
                    }
                    feature_count += 1
                
                if feature_count < max_features:
                    new_col = f"digit_count_{col}_{uuid.uuid4().hex[:4]}" if f"digit_count_{col}" in df_out.columns else f"digit_count_{col}"
                    if isinstance(df_out, dd.DataFrame):
                        df_out[new_col] = df_out[col].map_partitions(
                            lambda s: s.astype(str).str.count(r'\d'), 
                            meta=(new_col, 'int64')
                        )
                    else:
                        df_out[new_col] = df_out[col].astype(str).str.count(r'\d')
                    new_features.append(new_col)
                    st.session_state.feature_metadata[new_col] = {
                        'source_columns': [col], 'transformation': 'digit_count'
                    }
                    feature_count += 1
                
                if feature_count < max_features:
                    new_col = f"special_char_count_{col}_{uuid.uuid4().hex[:4]}" if f"special_char_count_{col}" in df_out.columns else f"special_char_count_{col}"
                    if isinstance(df_out, dd.DataFrame):
                        df_out[new_col] = df_out[col].map_partitions(
                            lambda s: s.astype(str).str.count(r'[^a-zA-Z0-9\s]'), 
                            meta=(new_col, 'int64')
                        )
                    else:
                        df_out[new_col] = df_out[col].astype(str).str.count(r'[^a-zA-Z0-9\s]')
                    new_features.append(new_col)
                    st.session_state.feature_metadata[new_col] = {
                        'source_columns': [col], 'transformation': 'special_char_count'
                    }
                    feature_count += 1
                
                if feature_count < max_features:
                    new_col = f"avg_word_len_{col}_{uuid.uuid4().hex[:4]}" if f"avg_word_len_{col}" in df_out.columns else f"avg_word_len_{col}"
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
                    st.session_state.feature_metadata[new_col] = {
                        'source_columns': [col], 'transformation': 'avg_word_length'
                    }
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
        
        df_sample = sample_for_preview(df_out[numeric_cols], n=10000)
        if df_sample.empty:
            return df, []
        
        df_sample_filled = df_sample.fillna(df_sample.median())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_sample_filled)
        
        n_clusters = min(5, max(2, len(df_sample) // 1000))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        new_col = f"cluster_kmeans_{uuid.uuid4().hex[:4]}"
        if isinstance(df_out, dd.DataFrame):
            df_full_filled = df_out[numeric_cols].fillna(df_out[numeric_cols].median()).persist()
            X_full_scaled = df_full_filled.map_partitions(
                lambda x: pd.DataFrame(scaler.transform(x), index=x.index, columns=x.columns),
                meta=df_full_filled
            ).persist()
            df_out[new_col] = X_full_scaled.map_partitions(
                lambda x: pd.Series(kmeans.predict(x), index=x.index),
                meta=(new_col, 'int64')
            )
        else:
            df_full_filled = df_out[numeric_cols].fillna(df_out[numeric_cols].median())
            X_full_scaled = scaler.transform(df_full_filled)
            df_out[new_col] = kmeans.predict(X_full_scaled)
        
        new_features.append(new_col)
        st.session_state.feature_metadata[new_col] = {
            'source_columns': numeric_cols, 'transformation': 'kmeans_cluster'
        }
        
        if len(new_features) < max_features:
            new_col = f"cluster_distance_{uuid.uuid4().hex[:4]}"
            if isinstance(df_out, dd.DataFrame):
                df_out[new_col] = X_full_scaled.map_partitions(
                    lambda x: pd.Series(np.min(np.linalg.norm(x.values - kmeans.cluster_centers_, axis=1), axis=1), index=x.index),
                    meta=(new_col, 'float64')
                )
            else:
                df_out[new_col] = np.min(np.linalg.norm(X_full_scaled[:, np.newaxis] - kmeans.cluster_centers_, axis=2), axis=1)
            new_features.append(new_col)
            st.session_state.feature_metadata[new_col] = {
                'source_columns': numeric_cols, 'transformation': 'cluster_distance'
            }
        
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
        
        if len(numeric_cols) < 3:
            return df, []
        
        df_sample = sample_for_preview(df_out[numeric_cols], n=10000)
        if df_sample.empty:
            return df, []
        
        df_sample_filled = df_sample.fillna(df_sample.median())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_sample_filled)
        
        n_components = min(max_features, len(numeric_cols), 5)
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        for i in range(n_components):
            new_col = f"pca_component_{i+1}_{uuid.uuid4().hex[:4]}"
            if isinstance(df_out, dd.DataFrame):
                df_full_filled = df_out[numeric_cols].fillna(df_out[numeric_cols].median()).persist()
                X_full_scaled = df_full_filled.map_partitions(
                    lambda x: pd.DataFrame(scaler.transform(x), index=x.index, columns=x.columns),
                    meta=df_full_filled
                ).persist()
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
            st.session_state.feature_metadata[new_col] = {
                'source_columns': numeric_cols, 'transformation': f'pca_component_{i+1}'
            }
        
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
            
        valid_columns = []
        for col in columns:
            if col not in df.columns:
                continue
            if not is_numeric_dtype(df[col]):
                continue
            try:
                df[col].astype(float)
                col_var = df[col].var()
                if isinstance(df, dd.DataFrame):
                    col_var = col_var.compute()
                if pd.isna(col_var) or col_var == 0:
                    continue
                valid_columns.append(col)
            except (ValueError, TypeError):
                continue
        
        if not valid_columns:
            return df, "No valid numeric columns with sufficient variance selected for polynomial features"
        
        degree = max(1, min(5, int(degree)))
        if len(valid_columns) > 10 and degree > 2:
            return df, "Too many columns for high-degree polynomial features (memory constraint)"
        
        df_out = df.copy()  # Always create a copy to prevent unintended modifications
        
        with st.spinner(f"Generating polynomial features (degree={degree})..."):
            try:
                if isinstance(df_out, dd.DataFrame):
                    poly = DaskPolynomialFeatures(degree=degree, include_bias=False)
                    selected_data = df_out[valid_columns].fillna(df_out[valid_columns].median())
                    poly_features = poly.fit_transform(selected_data)
                    feature_names = poly.get_feature_names_out(valid_columns)
                    poly_df = dd.from_array(poly_features, columns=feature_names)
                    new_feature_names = [name for name in feature_names if name not in valid_columns]
                    if new_feature_names:
                        poly_df_new = poly_df[new_feature_names]
                        df_out = dd.concat([df_out, poly_df_new], axis=1)
                        for name in new_feature_names:
                            st.session_state.feature_metadata[name] = {
                                'source_columns': valid_columns, 'transformation': 'polynomial'
                            }
                else:
                    poly = PolynomialFeatures(degree=degree, include_bias=False)
                    selected_data = df_out[valid_columns].fillna(df_out[valid_columns].median())
                    poly_features = poly.fit_transform(selected_data)
                    feature_names = poly.get_feature_names_out(valid_columns)
                    poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df_out.index)
                    new_feature_names = [name for name in feature_names if name not in valid_columns]
                    if new_feature_names:
                        for name in new_feature_names:
                            if name in df_out.columns:
                                name = f"{name}_{uuid.uuid4().hex[:4]}"
                            df_out[name] = poly_df[name]
                            st.session_state.feature_metadata[name] = {
                                'source_columns': valid_columns, 'transformation': 'polynomial'
                            }
                
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
                return df, "Error: Dataset too large for polynomial feature creation. Try fewer columns or lower degree."
                
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
            
        valid_columns = [c for c in columns if c in df.columns]
        if not valid_columns:
            return df, "No valid columns selected for datetime features"
        
        valid_features_list = ["year", "month", "day", "hour", "minute", "second", "dayofweek", "quarter"]
        valid_features = [f for f in features if f in valid_features_list]
        if not valid_features:
            return df, "No valid datetime features selected"
        
        df_out = df.copy()  # Always create a copy to prevent unintended modifications
        invalid_counts = {}
        
        with st.spinner(f"Extracting datetime features ({', '.join(valid_features)})..."):
            for col in valid_columns:
                try:
                    if not is_datetime64_any_dtype(df_out[col]):
                        if isinstance(df_out, dd.DataFrame):
                            df_out[col] = dd.to_datetime(df_out[col], errors='coerce')
                        else:
                            df_out[col] = pd.to_datetime(df_out[col], errors='coerce')
                    
                    invalid_count = df_out[col].isna().sum()
                    if isinstance(df_out, dd.DataFrame):
                        invalid_count = invalid_count.compute()
                    if invalid_count > 0:
                        invalid_counts[col] = invalid_count
                    
                    for feature in valid_features:
                        new_col = f"{col}_{feature}_{uuid.uuid4().hex[:4]}" if f"{col}_{feature}" in df_out.columns else f"{col}_{feature}"
                        try:
                            if isinstance(df_out, dd.DataFrame):
                                df_out[new_col] = getattr(df_out[col].dt, feature)
                            else:
                                df_out[new_col] = getattr(df_out[col].dt, feature)
                            st.session_state.feature_metadata[new_col] = {
                                'source_columns': [col], 'transformation': f'datetime_{feature}'
                            }
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
        return df, "Error: Dataset too large for datetime extraction. Try fewer columns."
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
            
        valid_columns = []
        for col in columns:
            if col not in df.columns:
                continue
            if not is_numeric_dtype(df[col]):
                continue
            unique_count = df[col].nunique()
            if isinstance(df, dd.DataFrame):
                unique_count = unique_count.compute()
            if unique_count < 2:
                continue
            valid_columns.append(col)
        
        if not valid_columns:
            return df, "No valid numeric columns with sufficient unique values selected for binning"
        
        bins = max(2, min(50, int(bins)))
        df_out = df.copy()  # Always create a copy to prevent unintended modifications
        
        with st.spinner(f"Binning columns into {bins} bins..."):
            for col in valid_columns:
                new_col = f"{col}_binned_{uuid.uuid4().hex[:4]}" if f"{col}_binned" in df_out.columns else f"{col}_binned"
                try:
                    if isinstance(df_out, dd.DataFrame):
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
                        try:
                            df_out[new_col] = pd.qcut(df_out[col], q=bins, labels=False, duplicates='drop')
                        except ValueError:
                            df_out[new_col] = pd.cut(df_out[col], bins=bins, labels=False, include_lowest=True, duplicates='drop')
                            
                    st.session_state.feature_metadata[new_col] = {
                        'source_columns': [col], 'transformation': 'binning'
                    }
                    
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
        return df, "Error: Dataset too large for binning. Try fewer columns."
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
            
        threshold = max(0.1, min(0.99, float(threshold)))
        
        num_cols, _ = dtype_split(df)
        valid_num_cols = []
        for col in num_cols[:50]:  # Limit to 50 columns
            try:
                col_var = df[col].var()
                if isinstance(df, dd.DataFrame):
                    col_var = col_var.compute()
                if not pd.isna(col_var) and col_var > 1e-10:
                    valid_num_cols.append(col)
            except Exception:
                continue
        
        if len(valid_num_cols) < 2:
            return df, "Insufficient numeric columns for correlation analysis"
        
        with st.spinner("Computing correlation matrix..."):
            try:
                if isinstance(df, dd.DataFrame):
                    df_sample = sample_for_preview(df[valid_num_cols], n=5000)
                    corr_matrix = df_sample.corr()
                else:
                    df_sample = df[valid_num_cols] if len(df) <= 5000 else sample_for_preview(df[valid_num_cols], n=5000)
                    corr_matrix = df_sample.corr()
                
                corr_matrix = corr_matrix.fillna(0)
                corr_matrix_abs = corr_matrix.abs()
                upper_triangle = np.triu(np.ones(corr_matrix_abs.shape), k=1).astype(bool)
                upper_corr = corr_matrix_abs.where(upper_triangle)
                
                to_drop = []
                for column in upper_corr.columns:
                    if any(upper_corr[column] > threshold):
                        to_drop.append(column)
                
                df_out = df.copy()  # Always create a copy to prevent unintended modifications
                
                if to_drop:
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
        return df, "Error: Dataset too large for correlation analysis. Try fewer columns."
    except Exception as e:
        logger.error(f"Unexpected error in select_features_correlation: {str(e)}")
        return df, f"Unexpected error: {str(e)}"

def evaluate_features(df: pd.DataFrame, features: List[str], target_col: str, is_classification: bool) -> float:
    """Evaluate feature set using cross-validation."""
    try:
        if not features or target_col not in df.columns:
            return 0.0
        X = df[features].fillna(df[features].median())
        y = df[target_col].dropna()
        common_idx = X.index.intersection(y.index)
        if len(common_idx) < 10:
            return 0.0
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        if is_classification:
            model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
        else:
            model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
        
        scores = cross_val_score(model, X, y, cv=3)
        return np.mean(scores)
    except Exception as e:
        logger.warning(f"Error evaluating features: {str(e)}")
        return 0.0

def automated_feature_engineering(df: Union[pd.DataFrame, dd.DataFrame], max_features: int = 50, preview: bool = False, target_col: Optional[str] = None) -> Tuple[Union[pd.DataFrame, dd.DataFrame], str]:
    """Enhanced automated feature engineering with advanced techniques and intelligent feature selection."""
    try:
        if not isinstance(df, (pd.DataFrame, dd.DataFrame)):
            return df, "Invalid DataFrame input"
        
        if df.empty:
            return df, "DataFrame is empty"
            
        if target_col and target_col not in df.columns:
            return df, f"Error: Target column '{target_col}' not found in DataFrame"
            
        max_features = max(5, min(500, int(max_features)))
        if len(set(df.columns)) != len(df.columns):
            return df, "Duplicate column names detected; please resolve before processing"
        
        df_out = df.copy()  # Always create a copy to prevent unintended modifications
        progress = st.progress(0)
        steps = 7  # Analysis + 6 feature types
        step_count = 0
        
        with st.spinner("Analyzing dataset..."):
            analysis = analyze_dataset_advanced(df_out, target_col)
            step_count += 1
            progress.progress(step_count / steps)
            if not analysis:
                return df, "Error analyzing dataset"
        
        all_new_features = []
        feature_budget = max_features
        
        if not preview:
            st.write("🧠 **Intelligent Feature Generation Strategy:**")
        
        numeric_budget = int(feature_budget * 0.4)
        if analysis['numeric_cols'] and numeric_budget > 0:
            if not preview:
                st.write(f"📊 Generating {numeric_budget} advanced numeric features...")
            df_out, numeric_features = create_advanced_numeric_features(df_out, analysis, numeric_budget)
            all_new_features.extend(numeric_features)
            feature_budget -= len(numeric_features)
            step_count += 1
            progress.progress(step_count / steps)
        
        categorical_budget = int(max_features * 0.25)
        if analysis['categorical_cols'] and categorical_budget > 0 and feature_budget > 0:
            if not preview:
                st.write(f"🏷️ Generating {min(categorical_budget, feature_budget)} categorical features...")
            df_out, cat_features = create_advanced_categorical_features(df_out, analysis, min(categorical_budget, feature_budget))
            all_new_features.extend(cat_features)
            feature_budget -= len(cat_features)
            step_count += 1
            progress.progress(step_count / steps)
        
        datetime_budget = int(max_features * 0.2)
        if analysis['datetime_cols'] and datetime_budget > 0 and feature_budget > 0:
            if not preview:
                st.write(f"📅 Generating {min(datetime_budget, feature_budget)} datetime features...")
            df_out, dt_features = create_advanced_datetime_features(df_out, analysis, min(datetime_budget, feature_budget))
            all_new_features.extend(dt_features)
            feature_budget -= len(dt_features)
            step_count += 1
            progress.progress(step_count / steps)
        
        text_budget = int(max_features * 0.1)
        if analysis['text_cols'] and text_budget > 0 and feature_budget > 0:
            if not preview:
                st.write(f"📝 Generating {min(text_budget, feature_budget)} text features...")
            df_out, text_features = create_advanced_text_features(df_out, analysis, min(text_budget, feature_budget))
            all_new_features.extend(text_features)
            feature_budget -= len(text_features)
            step_count += 1
            progress.progress(step_count / steps)
        
        if feature_budget > 0 and len(analysis['numeric_cols']) >= 2:
            if not preview:
                st.write("🎯 Generating clustering-based features...")
            df_out, cluster_features = create_clustering_features(df_out, analysis, min(5, feature_budget))
            all_new_features.extend(cluster_features)
            feature_budget -= len(cluster_features)
            step_count += 1
            progress.progress(step_count / steps)
        
        if feature_budget > 0 and len(analysis['numeric_cols']) >= 3:
            if not preview:
                st.write("🔍 Generating PCA-based features...")
            df_out, pca_features = create_pca_features(df_out, analysis, min(5, feature_budget))
            all_new_features.extend(pca_features)
            feature_budget -= len(pca_features)
            step_count += 1
            progress.progress(step_count / steps)
        
        if all_new_features and not preview:
            st.write("🎯 Performing intelligent feature selection...")
            try:
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
                    st.write(f"🗑️ Removed {len(low_var_features)} low-variance features")
                
                if len(all_new_features) > 1:
                    numeric_new_features = [col for col in all_new_features if is_numeric_dtype(df_out[col])]
                    if len(numeric_new_features) > 1:
                        corr_sample = sample_for_preview(df_out[numeric_new_features], n=5000)
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
                                st.write(f"🔗 Removed {len(high_corr_features)} highly correlated features")
                
                if target_col and target_col in df_out.columns and len(all_new_features) > max_features // 2:
                    st.write("🎯 Selecting top features based on importance...")
                    importance_scores = calculate_feature_importance(
                        sample_for_preview(df_out, n=5000) if len(df_out) > 5000 else df_out, 
                        target_col, 
                        {'numeric_cols': [col for col in all_new_features if is_numeric_dtype(df_out[col])]}
                    )
                    
                    if importance_scores:
                        sorted_features = sorted(
                            importance_scores.items(), 
                            key=lambda x: x[1].get('mutual_info', 0) + x[1].get('rf_importance', 0), 
                            reverse=True
                        )
                        top_features = [feat[0] for feat in sorted_features[:max_features//2]]
                        features_to_remove = [col for col in all_new_features if col in importance_scores and col not in top_features]
                        
                        if features_to_remove:
                            df_out = df_out.drop(columns=features_to_remove)
                            all_new_features = [col for col in all_new_features if col not in features_to_remove]
                            st.write(f"📈 Selected {len(top_features)} most important features")
                
                if target_col and all_new_features:
                    is_classification = (
                        target_col in analysis['categorical_cols'] or 
                        target_col in analysis['binary_cols'] or
                        df_out[target_col].nunique() < 20
                    )
                    score = evaluate_features(
                        sample_for_preview(df_out, n=5000) if len(df_out) > 5000 else df_out,
                        all_new_features,
                        target_col,
                        is_classification
                    )
                    st.write(f"📊 Cross-validation score with new features: {score:.3f}")
                    
            except Exception as e:
                logger.warning(f"Error in feature selection: {str(e)}")
        
        msg = f"🎉 Generated {len(all_new_features)} high-quality features"
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
        
        progress.progress(1.0)
        return df_out, msg
        
    except ValueError as e:
        logger.error(f"ValueError in automated_feature_engineering: {str(e)}")
        return df, f"Error in automated feature engineering: {str(e)}"
    except MemoryError:
        logger.error("MemoryError in automated_feature_engineering")
        return df, "Error: Dataset too large for automated feature engineering. Try fewer features."
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
        
        if columns:
            num_cols = [col for col in columns if col in num_cols]
        num_cols = num_cols[:20]
        
        if not num_cols:
            st.warning("No valid numeric columns selected for correlation heatmap")
            return
        
        with st.spinner("Computing correlation heatmap..."):
            try:
                df_sample = sample_for_preview(df[num_cols], n=10000)
                if df_sample.empty:
                    st.warning("No data available after sampling")
                    return
                
                corr_matrix = df_sample.corr()
                corr_matrix = corr_matrix.fillna(0)
                
                corr_data = corr_matrix.reset_index().melt(id_vars=['index'])
                corr_data.columns = ['var1', 'var2', 'correlation']
                corr_data['abs_correlation'] = corr_data['correlation'].abs()
                
                base = alt.Chart(corr_data)
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
                
                high_corr_pairs = corr_data[corr_data['abs_correlation'] >= threshold]
                high_corr_pairs = high_corr_pairs[high_corr_pairs['var1'] != high_corr_pairs['var2']]
                
                if not high_corr_pairs.empty:
                    st.write(f"**High correlation pairs (|r| >= {threshold}):**")
                    summary_df = high_corr_pairs[['var1', 'var2', 'correlation']].sort_values('correlation', key=abs, ascending=False)
                    st.dataframe(summary_df.head(10), hide_index=True)
                else:
                    st.info(f"No correlation pairs found above threshold {threshold}")
                
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
                st.info("💡 Try selecting fewer columns or reducing the sample size.")
                
    except Exception as e:
        logger.error(f"Error in plot_correlation_heatmap: {str(e)}")
        st.error(f"Error plotting correlation heatmap: {str(e)}")
        st.info("💡 Try selecting fewer columns or reducing the sample size.")

def safe_eval_expression(df: Union[pd.DataFrame, dd.DataFrame], expression: str, new_col: str) -> Tuple[Optional[pd.Series], str]:
    """Safely evaluate a custom expression using pandas eval with enhanced domain error checking."""
    try:
        if not expression or not expression.strip():
            return None, "Expression cannot be empty"
        
        # Sanitize expression - remove potentially dangerous operations
        dangerous_keywords = ['import', 'exec', 'eval', 'open', 'file', '__', 'getattr', 'setattr', 'delattr', 'globals', 'locals']
        expression_lower = expression.lower()
        for keyword in dangerous_keywords:
            if keyword in expression_lower:
                return None, f"Expression contains forbidden keyword: {keyword}"
        
        # Check if expression references valid columns
        import re
        # Extract potential column names (alphanumeric + underscore)
        potential_cols = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', expression)
        valid_cols = [col for col in potential_cols if col in df.columns]
        
        if not valid_cols:
            return None, "Expression must reference valid column names from your dataset"
        
        # Check for allowed mathematical functions and operations
        allowed_funcs = {'log', 'log10', 'log1p', 'exp', 'sin', 'cos', 'tan', 'sqrt', 'abs', 'round', 'floor', 'ceil'}
        allowed_ops = {'+', '-', '*', '/', '(', ')', '**', '^'}
        
        # Simple check for mathematical operations
        has_valid_op = any(op in expression for op in ['+', '-', '*', '/', '**'])
        if not has_valid_op and not any(func in expression_lower for func in allowed_funcs):
            return None, "Expression must include mathematical operations (+, -, *, /, **) or functions (log, sqrt, etc.)"
        
        # Pre-validate for potential domain errors
        domain_warnings = []
        if 'log(' in expression_lower:
            domain_warnings.append("⚠️ log() function detected - ensure no zero/negative values")
        if 'sqrt(' in expression_lower:
            domain_warnings.append("⚠️ sqrt() function detected - ensure no negative values")
        if '/' in expression:
            domain_warnings.append("⚠️ Division detected - ensure no division by zero")
        
        # Test on sample first
        try:
            df_sample = sample_for_preview(df, n=1000) if len(df) > 1000 else df
            if isinstance(df, dd.DataFrame) and hasattr(df_sample, 'compute'):
                df_sample = df_sample.compute()
                
            # Replace mathematical functions for pandas eval compatibility
            eval_expression = expression
            eval_expression = eval_expression.replace('log(', 'log(')
            eval_expression = eval_expression.replace('sqrt(', 'sqrt(')
            
            # Test evaluation on sample
            result_sample = df_sample.eval(eval_expression, engine='python')
            
            # Check for invalid results
            if pd.isna(result_sample).all():
                return None, "Expression results in all NaN values - check for domain errors"
            
            invalid_count = pd.isna(result_sample).sum()
            if invalid_count > 0:
                domain_warnings.append(f"⚠️ {invalid_count} invalid/NaN values generated")
            
            # If sample works, apply to full dataset
            if isinstance(df, dd.DataFrame):
                result = df.map_partitions(
                    lambda x: x.eval(eval_expression, engine='python'), 
                    meta=(new_col, 'float64')
                )
            else:
                result = df.eval(eval_expression, engine='python')
            
            success_msg = "✅ Expression evaluated successfully"
            if domain_warnings:
                success_msg += "\n" + "\n".join(domain_warnings)
            
            return result, success_msg
            
        except ZeroDivisionError:
            return None, "❌ Division by zero detected in expression"
        except ValueError as ve:
            if 'log' in str(ve).lower():
                return None, "❌ Invalid log operation - cannot take log of zero/negative values"
            elif 'sqrt' in str(ve).lower():
                return None, "❌ Invalid sqrt operation - cannot take square root of negative values"
            else:
                return None, f"❌ Mathematical domain error: {str(ve)}"
        except OverflowError:
            return None, "❌ Numerical overflow - result too large to compute"
            
    except SyntaxError as e:
        return None, f"❌ Syntax error in expression: {str(e)}"
    except KeyError as e:
        return None, f"❌ Column not found: {str(e)}"
    except Exception as e:
        logger.error(f"Error in safe_eval_expression: {str(e)}")
        return None, f"❌ Error evaluating expression: {str(e)}"

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
    """Enhanced Feature Engineering section with improved workflow and smart recommendations."""
    try:
        # Initialize session state
        initialize_session_state()
        
        # Robust session state validation
        if 'df' not in st.session_state or st.session_state.df is None or st.session_state.df.empty:
            st.error("❌ No valid dataset found. Please upload a dataset in the Upload section.")
            st.info("💡 Upload a CSV, Excel, or JSON file to get started with feature engineering.")
            return
        
        st.header("🎨 Feature Engineering Studio")
        st.markdown(
            "**Discover hidden patterns and create powerful features with AI-powered recommendations.**\n"
            "Transform your raw data into machine learning-ready features using cutting-edge techniques."
        )
        
        df = st.session_state.df
        
        # Add undo/redo controls
        col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
        with col1:
            if st.button("↶ Undo", disabled=len(st.session_state.undo_stack) == 0):
                if undo_last_action():
                    st.success("✅ Action undone")
                    st.experimental_rerun()
        with col2:
            if st.button("↷ Redo", disabled=len(st.session_state.redo_stack) == 0):
                if redo_last_action():
                    st.success("✅ Action redone")
                    st.experimental_rerun()
        with col3:
            if st.button("🎓 Tutorial", key="tutorial_btn"):
                st.session_state.show_tutorial = not st.session_state.get('show_tutorial', False)
                st.experimental_rerun()
        
        # Interactive Tutorial
        if st.session_state.get('show_tutorial', False):
            with st.expander("🎓 Interactive Feature Engineering Guide", expanded=True):
                tutorial_tab = st.radio(
                    "Choose a topic to learn about:",
                    ["🔍 Basics", "🔢 Advanced", "🤖 Automation", "📊 Evaluation"],
                    horizontal=True,
                    key="tutorial_tab"
                )
                
                if tutorial_tab == "🔍 Basics":
                    st.markdown("""
                    **What is Feature Engineering?**
                    
                    Feature engineering is the process of creating new features or transforming existing ones to improve machine learning model performance.
                    
                    **Common Techniques:**
                    - **Polynomial Features**: Create interactions (A × B) and powers (A²) of numeric columns
                    - **Datetime Features**: Extract year, month, day, etc. from date columns 
                    - **Binning**: Convert continuous values into discrete bins (e.g., age groups)
                    - **Transformations**: Apply log, sqrt to handle skewed data
                    
                    **Example**: For columns [Age, Income], polynomial features create [Age², Income², Age×Income]
                    """)
                    
                elif tutorial_tab == "🔢 Advanced":
                    st.markdown("""
                    **Advanced Techniques:**
                    
                    **Correlation Analysis**: Remove redundant features that are highly correlated
                    - Reduces multicollinearity
                    - Improves model interpretability
                    
                    **Text Features**: Extract meaningful patterns from text
                    - Character count, word count
                    - Special character frequency
                    
                    **Clustering Features**: Discover hidden patterns
                    - K-means clustering to group similar data points
                    - Distance to cluster centers as features
                    
                    **Cyclical Encoding**: Handle cyclical data (months, hours)
                    - Sin/Cos transformations preserve circular relationships
                    """)
                    
                elif tutorial_tab == "🤖 Automation":
                    st.markdown("""
                    **Automated Feature Engineering:**
                    
                    Our AI system analyzes your data and suggests optimal transformations:
                    
                    **Smart Detection:**
                    - Identifies skewed distributions → Suggests log/sqrt transforms
                    - Finds datetime patterns → Recommends temporal features 
                    - Detects high cardinality → Proposes encoding strategies
                    
                    **Intelligent Feature Selection:**
                    - Removes low-variance features
                    - Eliminates highly correlated features
                    - Ranks features by importance
                    
                    **Best Practice**: Start with automated engineering, then fine-tune manually
                    """)
                    
                elif tutorial_tab == "📊 Evaluation":
                    st.markdown("""
                    **Evaluating Feature Quality:**
                    
                    **Feature Impact Analysis**:
                    - Compares model performance before/after feature engineering
                    - Uses cross-validation for robust estimates
                    - Shows improvement in accuracy/R² score
                    
                    **Feature Importance**:
                    - Mutual information: Measures non-linear relationships
                    - Random Forest importance: Tree-based feature ranking
                    - F-statistics: Linear relationship strength
                    
                    **Quality Metrics**:
                    - Completeness: % of non-missing values
                    - Uniqueness: Diversity of values
                    - Stability: Consistency across data splits
                    """)
                    
                st.info("💡 **Pro Tip**: Use the recommendations in the 'Explore & Recommend' tab to get started with AI-suggested improvements!")
        
        # Improved tab layout
        tabs = st.tabs([
            "🔍 Explore & Recommend",
            "🛠️ Create & Transform", 
            "✂️ Select & Optimize",
            "🤖 Automated Engineering",
            "📊 Evaluate & Export"
        ])
        
        df = st.session_state.df
        before_stats = compute_basic_stats(df)
        
        # Tab 1: Explore & Recommend
        with tabs[0]:
            st.subheader("🔍 Dataset Analysis & Smart Recommendations")
            st.markdown("Get intelligent insights about your data and personalized feature engineering recommendations.")
            
            # Dataset Analysis
            with st.expander("📊 Dataset Overview", expanded=True):
                with st.spinner("Analyzing your dataset..."):
                    analysis = analyze_dataset_advanced(df)
                    
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📈 Numeric Features", len(analysis.get('numeric_cols', [])))
                with col2:
                    st.metric("🏷️ Categorical Features", len(analysis.get('categorical_cols', [])))
                with col3:
                    st.metric("📅 DateTime Features", len(analysis.get('datetime_cols', [])))
                with col4:
                    st.metric("📝 Text Features", len(analysis.get('text_cols', [])))
                
                # Data Quality Indicators
                if analysis.get('data_quality'):
                    st.markdown("**Data Quality Summary:**")
                    quality_data = []
                    for col, quality in list(analysis['data_quality'].items())[:10]:  # Show top 10
                        quality_data.append({
                            'Column': col,
                            'Completeness': f"{quality.get('completeness', 0):.1%}",
                            'Uniqueness': f"{quality.get('uniqueness', 0):.1%}",
                            'Type': analysis['stats'].get(col, {}).get('type', 'unknown')
                        })
                    if quality_data:
                        st.dataframe(pd.DataFrame(quality_data), use_container_width=True, hide_index=True)
            
            # Smart Recommendations
            with st.expander("🧠 Smart Recommendations", expanded=True):
                recommendations = generate_smart_recommendations(analysis)
                
                if recommendations:
                    st.markdown("**AI-Powered Feature Engineering Suggestions:**")
                    
                    for i, rec in enumerate(recommendations):
                        priority_color = {
                            'high': '🔴',
                            'medium': '🟡', 
                            'low': '🟢'
                        }.get(rec['priority'], '⚪')
                        
                        with st.container():
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.markdown(f"{priority_color} **{rec['icon']} {rec['title']}**")
                                st.caption(rec['description'])
                            with col2:
                                if st.button(f"Apply", key=f"apply_rec_{i}", type="primary"):
                                    # Store recommendation for application
                                    st.session_state[f'selected_recommendation'] = rec
                                    st.info(f"💡 Recommendation selected! Go to the Create & Transform tab to apply.")
                            st.divider()
                else:
                    st.info("🎉 Your dataset looks great! No immediate recommendations.")
                    st.markdown("**Why no recommendations?**\n- All features appear well-distributed\n- No high correlations detected\n- Data types are appropriate")
            
            # Feature Correlation Preview
            with st.expander("🔗 Feature Relationships", expanded=False):
                numeric_cols = analysis.get('numeric_cols', [])
                if len(numeric_cols) >= 2:
                    selected_cols = st.multiselect(
                        "Select columns to visualize correlations",
                        numeric_cols,
                        default=numeric_cols[:min(8, len(numeric_cols))],
                        key="explore_corr_cols"
                    )
                    
                    if selected_cols and len(selected_cols) >= 2:
                        threshold = st.slider("Correlation threshold", 0.1, 0.9, 0.5, key="explore_corr_threshold")
                        plot_correlation_heatmap(df, columns=selected_cols, threshold=threshold)
                    else:
                        st.warning("Select at least 2 numeric columns to view correlations.")
                else:
                    st.info("Need at least 2 numeric columns for correlation analysis.")
        
        # Tab 2: Create & Transform
        with tabs[1]:
            st.subheader("🛠️ Create & Transform Features")
            st.markdown("Generate new features and transform existing ones with advanced techniques.")
            
            # Check for selected recommendation
            if 'selected_recommendation' in st.session_state:
                rec = st.session_state['selected_recommendation']
                st.success(f"🎯 Applying recommendation: {rec['title']}")
                st.info(f"📝 {rec['description']}")
                del st.session_state['selected_recommendation']  # Clear after showing
            
            # Batch Feature Creation
            with st.expander("🛠️ Batch Feature Preview", expanded=True):
                st.markdown("**Preview multiple feature engineering operations before applying:**")
                
                # Multi-select for batch operations
                batch_operations = st.multiselect(
                    "Select operations to preview together",
                    [
                        "Polynomial Features (degree 2)",
                        "Datetime Features (year, month, day)",
                        "Binning (10 bins)",
                        "Log Transformations (skewed columns)"
                    ],
                    key="batch_operations"
                )
                
                if batch_operations and st.button("🔍 Preview Batch Operations", key="batch_preview"):
                    preview_df = df.copy()
                    batch_messages = []
                    
                    with st.spinner("Generating batch preview..."):
                        # Apply selected operations in sequence
                        for operation in batch_operations:
                            try:
                                if "Polynomial" in operation:
                                    numeric_cols = [c for c in preview_df.columns if is_numeric_dtype(preview_df[c])][:3]  # Limit for demo
                                    if numeric_cols:
                                        preview_df, msg = create_polynomial_features(preview_df, numeric_cols, 2, preview=True)
                                        batch_messages.append(f"✅ Polynomial: {msg}")
                                
                                elif "Datetime" in operation:
                                    dt_cols = [c for c in preview_df.columns if is_datetime64_any_dtype(preview_df[c]) or 
                                             'date' in c.lower() or 'time' in c.lower()][:2]
                                    if dt_cols:
                                        preview_df, msg = extract_datetime_features(preview_df, dt_cols, ["year", "month", "day"], preview=True)
                                        batch_messages.append(f"✅ Datetime: {msg}")
                                
                                elif "Binning" in operation:
                                    numeric_cols = [c for c in preview_df.columns if is_numeric_dtype(preview_df[c])][:2]
                                    if numeric_cols:
                                        preview_df, msg = bin_features(preview_df, numeric_cols, 10, preview=True)
                                        batch_messages.append(f"✅ Binning: {msg}")
                                
                                elif "Log" in operation:
                                    # Apply log to skewed columns
                                    numeric_cols = [c for c in preview_df.columns if is_numeric_dtype(preview_df[c])]
                                    for col in numeric_cols[:3]:  # Limit for demo
                                        try:
                                            skewness = preview_df[col].skew()
                                            if skewness > 1:
                                                new_col = generate_unique_col_name(preview_df, f"log_{col}")
                                                preview_df[new_col] = np.log1p(np.maximum(preview_df[col], 0))
                                                batch_messages.append(f"✅ Log transformation: {col} -> {new_col}")
                                        except:
                                            continue
                            except Exception as e:
                                batch_messages.append(f"❌ {operation}: {str(e)}")
                    
                    # Show batch results
                    if batch_messages:
                        st.markdown("**Batch Operation Results:**")
                        for msg in batch_messages:
                            st.markdown(msg)
                    
                    # Show new columns preview
                    new_cols = [col for col in preview_df.columns if col not in df.columns]
                    if new_cols:
                        st.markdown(f"**New Features Created ({len(new_cols)}):**")
                        preview_sample = sample_for_preview(preview_df)
                        st.dataframe(preview_sample[new_cols].head(10), use_container_width=True)
                        
                        if st.button("✅ Apply All Batch Operations", key="apply_batch", type="primary"):
                            save_state_for_undo()
                            st.session_state.df = preview_df
                            push_history(f"Applied batch operations: {', '.join(batch_operations)}")
                            st.success("✅ Batch operations applied successfully!")
                            st.experimental_rerun()
                    else:
                        st.warning("No new features were created. Try different operations or check your data.")
            
            # Individual Feature Operations (Collapsible)
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
        
        # Tab 5: Evaluate & Export 
        with tabs[4]:
            st.subheader("📊 Evaluate Performance & Export Results")
            st.markdown("Analyze feature impact, track your progress, and export your engineered dataset.")
            
            # Feature Impact Analysis
            with st.expander("🏆 Feature Impact Analysis", expanded=True):
                target_col = st.selectbox(
                    "Select target column for impact analysis",
                    [None] + list(df.columns),
                    key="impact_target",
                    help="Choose the column you want to predict to evaluate feature usefulness"
                )
                
                if target_col:
                    # Get new features (created during this session)
                    new_features = [col for col in df.columns if col in st.session_state.feature_metadata]
                    
                    if new_features:
                        with st.spinner("Evaluating feature impact..."):
                            impact_results = evaluate_feature_impact(df, target_col, new_features)
                            
                        if impact_results is not None and not impact_results.empty:
                            st.markdown("**📊 Model Performance Comparison:**")
                            st.dataframe(impact_results, use_container_width=True, hide_index=True)
                            
                            # Create visualization
                            if len(impact_results) >= 2:
                                chart = alt.Chart(impact_results[impact_results['Feature Set'] != 'Improvement']).mark_bar(size=60).encode(
                                    x=alt.X('Feature Set:O', title=''),
                                    y=alt.Y('CV Score:Q', title='Cross-Validation Score'),
                                    color=alt.Color('Feature Set:O', scale=alt.Scale(range=['#1f77b4', '#ff7f0e']), legend=None),
                                    tooltip=['Feature Set:O', 'CV Score:Q', 'Number of Features:O']
                                ).properties(
                                    title='Model Performance: Before vs After Feature Engineering',
                                    width=400,
                                    height=300
                                )
                                st.altair_chart(chart, use_container_width=True)
                                
                                # Show improvement summary
                                improvement_row = impact_results[impact_results['Feature Set'] == 'Improvement']
                                if not improvement_row.empty:
                                    improvement = improvement_row.iloc[0]['CV Score']
                                    if improvement > 0:
                                        st.success(f"🚀 **Great work!** Your feature engineering improved model performance by {improvement:.4f}")
                                    elif improvement < -0.01:
                                        st.warning(f"⚠️ Model performance decreased by {abs(improvement):.4f}. Consider removing some features.")
                                    else:
                                        st.info("📊 Performance remained similar. Features may provide different insights.")
                        else:
                            st.info("📝 Unable to evaluate impact. Try with a different target column or ensure sufficient data quality.")
                    else:
                        st.info("🚧 No new features detected in this session. Create some features first!")
                else:
                    st.info("🔎 Select a target column to analyze feature impact.")
            
            # Feature Metadata and Grouping
            with st.expander("📋 Feature Metadata & Lineage", expanded=True):
                if st.session_state.feature_metadata:
                    st.markdown("**🧠 Feature Lineage Tracking:**")
                    
                    # Group features by transformation type
                    feature_groups = {}
                    for feature, metadata in st.session_state.feature_metadata.items():
                        transformation = metadata.get('transformation', 'unknown')
                        if transformation not in feature_groups:
                            feature_groups[transformation] = []
                        feature_groups[transformation].append({
                            'Feature': feature,
                            'Source Columns': ', '.join(metadata.get('source_columns', [])),
                            'Transformation': transformation
                        })
                    
                    # Display grouped features
                    for transformation, features in feature_groups.items():
                        with st.container():
                            st.markdown(f"**{transformation.replace('_', ' ').title()} Features ({len(features)}):**")
                            feature_df = pd.DataFrame(features)
                            st.dataframe(feature_df, use_container_width=True, hide_index=True)
                            
                    # Feature tree visualization
                    st.markdown("**🌳 Feature Dependency Tree:**")
                    tree_data = []
                    for feature, metadata in st.session_state.feature_metadata.items():
                        source_cols = metadata.get('source_columns', [])
                        for source in source_cols:
                            tree_data.append({
                                'source': source,
                                'target': feature,
                                'transformation': metadata.get('transformation', 'unknown')
                            })
                    
                    if tree_data:
                        tree_df = pd.DataFrame(tree_data)
                        st.dataframe(tree_df, use_container_width=True, hide_index=True)
                else:
                    st.info("🌱 No feature metadata available. Features created in this session will be tracked here.")
            
            # Dataset Overview
            with st.expander("📊 Dataset Evolution Dashboard", expanded=True):
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
            
            # Enhanced Export Section
            with st.expander("📦 Export Dataset & Pipeline", expanded=False):
                st.markdown("**Download your engineered dataset and reproduce the pipeline**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 Dataset Export**")
                    export_cols = st.multiselect(
                        "Select columns to export", 
                        current_df.columns, 
                        default=list(current_df.columns),
                        key="export_cols",
                        help="Choose which columns to include in the export"
                    )
                
                with col2:
                    st.markdown("**📝 Pipeline Export**")
                    export_format = st.selectbox(
                        "Export pipeline as:",
                        ["JSON Configuration", "Python Code", "Both"],
                        key="export_format",
                        help="Choose format for pipeline export"
                    )
                
                if export_cols:
                    col1a, col1b = st.columns(2)
                    with col1a:
                        st.metric("Selected Columns", len(export_cols))
                    with col1b:
                        total_rows = len(current_df)
                        if isinstance(current_df, dd.DataFrame):
                            total_rows = total_rows.compute() if hasattr(total_rows, 'compute') else total_rows
                        st.metric("Total Rows", total_rows)
                    
                    # Dataset CSV Export
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
                    
                    # Pipeline Export
                    if st.button("🧰 Download Pipeline", key="download_pipeline"):
                        with st.spinner("Generating pipeline export..."):
                            # Export JSON configuration
                            json_bytes = None
                            code_str = None
                            if export_format in ["JSON Configuration", "Both"]:
                                try:
                                    import json
                                    pipeline_json = json.dumps(st.session_state.pipeline, indent=2).encode('utf-8')
                                    json_bytes = pipeline_json
                                except Exception as e:
                                    st.error(f"❌ Error generating JSON: {str(e)}")
                            if export_format in ["Python Code", "Both"]:
                                code_str = export_pipeline_as_code(st.session_state.pipeline)
                            
                            # Provide downloads
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            if json_bytes:
                                st.download_button(
                                    label="📄 Download Pipeline (JSON)",
                                    data=json_bytes,
                                    file_name=f"feature_pipeline_{timestamp}.json",
                                    mime="application/json"
                                )
                            if code_str:
                                st.download_button(
                                    label="🐍 Download Pipeline (Python)",
                                    data=code_str.encode('utf-8'),
                                    file_name=f"feature_pipeline_{timestamp}.py",
                                    mime="text/x-python"
                                )
                            if not json_bytes and not code_str:
                                st.warning("No pipeline export available")
                else:
                    st.warning("Please select at least one column to export.")
    
    except Exception as e:
        logger.error(f"Error in section_feature_engineering: {str(e)}")
        st.error(f"An unexpected error occurred: {str(e)}")
