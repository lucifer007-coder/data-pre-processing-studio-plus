import logging
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st
import dask.dataframe as dd
import re

logger = logging.getLogger(__name__)

# Constants for security limits
MAX_TEXT_LENGTH = 100000  # 100KB limit
MAX_DIMENSION = 2000
MIN_DIMENSION = 100
MAX_BINS_LIMIT = 1000

def validate_column_name(column_name):
    """Validate column name to prevent injection attacks."""
    if not isinstance(column_name, str) or not column_name.strip():
        raise ValueError(f"Column name must be a non-empty string")
    # Allow alphanumeric, underscore, hyphen, and space characters
    if not re.match(r'^[a-zA-Z0-9_\-\s]+$', column_name):
        raise ValueError(f"Invalid column name: {column_name}")
    return column_name

def validate_dimensions(width, height):
    """Validate and sanitize width and height dimensions."""
    if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
        raise ValueError("Width and height must be numeric")
    if width <= 0 or height <= 0:
        raise ValueError("Height and width must be positive")
    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        raise ValueError(f"Height and width must be reasonable (≤{MAX_DIMENSION})")
    return width, height

def alt_histogram(df: pd.DataFrame | dd.DataFrame, column: str, title: str, max_bins=50, height=300, width=400):
    """Generate a histogram for a numeric column."""
    try:
        # Input validation
        validate_column_name(column)
        validate_dimensions(width, height)
        
        if not isinstance(title, str):
            raise ValueError("Title must be a string")
        if not isinstance(max_bins, int) or max_bins <= 0 or max_bins > MAX_BINS_LIMIT:
            raise ValueError(f"max_bins must be between 1 and {MAX_BINS_LIMIT}")
        
        if isinstance(df, dd.DataFrame):
            sample_df = df[[column]].head(1000).compute()
        else:
            sample_df = df[[column]].head(1000)
        
        # Check if column exists
        if column not in sample_df.columns:
            logger.warning(f"Column {column} not found in DataFrame")
            st.warning(f"Column {column} not found in DataFrame")
            return None
        
        if sample_df[column].nunique() <= 1:
            logger.warning(f"Column {column} has only one unique value, skipping histogram.")
            st.warning(f"Cannot plot histogram for {column}: only one unique value.")
            return None
        
        if sample_df[column].isna().all():
            logger.warning(f"Column {column} contains only missing values.")
            st.warning(f"Cannot plot histogram for {column}: all values are missing.")
            return None

        # Clip outliers at 1st and 99th percentiles - create copy to avoid mutation
        p01, p99 = sample_df[column].quantile([0.01, 0.99]).values
        filtered_df = sample_df[sample_df[column].between(p01, p99)].copy()
        
        # Validate column name for safe string interpolation
        safe_column = validate_column_name(column)
        
        chart = alt.Chart(filtered_df).mark_bar().encode(
            x=alt.X(f"{safe_column}:Q", bin=alt.Bin(maxbins=max_bins)),
            y='count()',
            tooltip=['count()']
        ).properties(
            title=title,
            height=height,
            width=width
        )
        return chart
    
    except Exception as e:
        logger.error(f"Error in alt_histogram for {column}: {e}")
        st.error(f"Error generating histogram for {column}: {e}")
        return None

def alt_line_plot(df: pd.DataFrame | dd.DataFrame, time_col: str, value_col: str, title: str, height=300, width=400):
    """Generate a line plot for time-series data."""
    try:
        # Input validation
        validate_column_name(time_col)
        validate_column_name(value_col)
        validate_dimensions(width, height)
        
        if not isinstance(title, str):
            raise ValueError("Title must be a string")
        
        if isinstance(df, dd.DataFrame):
            sample_df = df[[time_col, value_col]].head(1000).compute()
        else:
            sample_df = df[[time_col, value_col]].head(1000)
        
        # Check if columns exist
        if time_col not in sample_df.columns or value_col not in sample_df.columns:
            logger.warning(f"One or both columns ({time_col}, {value_col}) not found in DataFrame.")
            st.warning(f"One or both columns ({time_col}, {value_col}) not found in DataFrame.")
            return None
        
        if not pd.api.types.is_datetime64_any_dtype(sample_df[time_col]):
            logger.warning(f"Column {time_col} is not datetime type.")
            st.warning(f"Column {time_col} must be datetime for line plot.")
            return None
        
        if not pd.api.types.is_numeric_dtype(sample_df[value_col]):
            logger.warning(f"Column {value_col} is not numeric.")
            st.warning(f"Column {value_col} must be numeric for line plot.")
            return None

        # Validate column names for safe string interpolation
        safe_time_col = validate_column_name(time_col)
        safe_value_col = validate_column_name(value_col)

        chart = alt.Chart(sample_df).mark_line().encode(
            x=alt.X(f"{safe_time_col}:T", title=time_col),
            y=alt.Y(f"{safe_value_col}:Q", title=value_col),
            tooltip=[time_col, value_col]
        ).properties(
            title=title,
            height=height,
            width=width
        )
        return chart
    
    except Exception as e:
        logger.error(f"Error in alt_line_plot for {time_col}, {value_col}: {e}")
        st.error(f"Error generating line plot: {e}")
        return None

def word_cloud(df: pd.DataFrame | dd.DataFrame, column: str, title: str, width=400, height=300):
    """Generate a word cloud for a text column."""
    try:
        # Input validation
        validate_column_name(column)
        validate_dimensions(width, height)
        
        if not isinstance(title, str):
            raise ValueError("Title must be a string")
        
        if isinstance(df, dd.DataFrame):
            sample_df = df[[column]].head(1000).compute()
        else:
            sample_df = df[[column]].head(1000)
        
        # Check if column exists
        if column not in sample_df.columns:
            logger.warning(f"Column {column} not found in DataFrame")
            st.warning(f"Column {column} not found in DataFrame")
            return False
        
        # Create text with memory protection
        text = ' '.join(sample_df[column].dropna().astype(str))
        if len(text) > MAX_TEXT_LENGTH:
            text = text[:MAX_TEXT_LENGTH]
            logger.warning(f"Text truncated to {MAX_TEXT_LENGTH} characters for memory protection")
        
        if not text.strip():
            logger.warning(f"No valid text data in {column} for word cloud.")
            st.warning(f"No valid text data in {column} for word cloud.")
            return False
        
        # Limit dimensions for WordCloud
        safe_width = min(max(width, MIN_DIMENSION), MAX_DIMENSION)
        safe_height = min(max(height, MIN_DIMENSION), MAX_DIMENSION)
        
        wordcloud = WordCloud(
            width=safe_width, 
            height=safe_height, 
            background_color='white',
            max_words=200  # Limit number of words for performance
        ).generate(text)
        
        # Safe figure size calculation
        fig_width = max(width/100, 1)
        fig_height = max(height/100, 1)
        
        fig = plt.figure(figsize=(fig_width, fig_height))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        st.pyplot(plt)
        plt.close(fig)  # Close the specific figure
        return True
    
    except Exception as e:
        logger.error(f"Error in word_cloud for {column}: {e}")
        st.error(f"Error generating word cloud for {column}: {e}")
        return False
