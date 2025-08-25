import logging
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st
import dask.dataframe as dd

logger = logging.getLogger(__name__)

def alt_histogram(df: pd.DataFrame | dd.DataFrame, column: str, title: str, max_bins=50, height=300, width=400):
    """Generate a histogram for a numeric column."""
    try:
        if isinstance(df, dd.DataFrame):
            sample_df = df[[column]].head(1000).compute()
        else:
            sample_df = df[[column]].head(1000)
        
        if sample_df[column].nunique() <= 1:
            logger.warning(f"Column {column} has only one unique value, skipping histogram.")
            st.warning(f"Cannot plot histogram for {column}: only one unique value.")
            return None
        
        if sample_df[column].isna().all():
            logger.warning(f"Column {column} contains only missing values.")
            st.warning(f"Cannot plot histogram for {column}: all values are missing.")
            return None

        # Clip outliers at 1st and 99th percentiles
        p01, p99 = sample_df[column].quantile([0.01, 0.99]).values
        sample_df = sample_df[sample_df[column].between(p01, p99)]
        
        chart = alt.Chart(sample_df).mark_bar().encode(
            x=alt.X(f"{column}:Q", bin=alt.Bin(maxbins=max_bins)),
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
        if isinstance(df, dd.DataFrame):
            sample_df = df[[time_col, value_col]].head(1000).compute()
        else:
            sample_df = df[[time_col, value_col]].head(1000)
        
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

        chart = alt.Chart(sample_df).mark_line().encode(
            x=alt.X(f"{time_col}:T", title=time_col),
            y=alt.Y(f"{value_col}:Q", title=value_col),
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
        if isinstance(df, dd.DataFrame):
            sample_df = df[[column]].head(1000).compute()
        else:
            sample_df = df[[column]].head(1000)
        
        text = ' '.join(sample_df[column].dropna().astype(str))
        if not text.strip():
            logger.warning(f"No valid text data in {column} for word cloud.")
            st.warning(f"No valid text data in {column} for word cloud.")
            return None
        
        wordcloud = WordCloud(width=width, height=height, background_color='white').generate(text)
        plt.figure(figsize=(width/100, height/100))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        st.pyplot(plt)
        plt.close()
        return True
    
    except Exception as e:
        logger.error(f"Error in word_cloud for {column}: {e}")
        st.error(f"Error generating word cloud for {column}: {e}")
        return None
