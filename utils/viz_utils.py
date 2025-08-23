import logging
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from typing import Optional
import wordcloud
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def alt_histogram(
    df: pd.DataFrame,
    column: str,
    title: Optional[str] = None,
    max_bins: int = 40,
    height: int = 250,
    width: int = 400
) -> Optional[alt.Chart]:
    """Create a robust Altair histogram with comprehensive error handling."""
    try:
        if not isinstance(df, pd.DataFrame):
            logger.error("df must be a pandas DataFrame")
            st.error("Invalid data type for histogram")
            return None
        if column not in df.columns:
            logger.error(f"Column '{column}' not found in DataFrame")
            st.error(f"Column '{column}' not found in data")
            return None
        if not pd.api.types.is_numeric_dtype(df[column]):
            logger.error(f"Column '{column}' is not numeric")
            st.error(f"Column '{column}' is not numeric")
            return None

        clean_df = df[[column]].dropna()
        if clean_df.empty:
            st.warning(f"Column '{column}' contains only missing values")
            return None

        if clean_df[column].nunique() == 1:
            value = clean_df[column].iloc[0]
            single_df = pd.DataFrame({column: [value], 'count': [len(clean_df)]})
            chart = (
                alt.Chart(single_df)
                .mark_bar()
                .encode(
                    x=alt.X(f"{column}:Q", title=column),
                    y=alt.Y("count:Q", title="Count"),
                    tooltip=[alt.Tooltip(f"{column}:Q", title=column), alt.Tooltip("count:Q", title="Count")]
                )
                .properties(height=height, width=width, title=title or f"Distribution of {column} (Single Value)")
            )
            return chart

        q1, q99 = clean_df[column].quantile([0.01, 0.99])
        filtered_df = clean_df[(clean_df[column] >= q1) & (clean_df[column] <= q99)].copy()
        if filtered_df.empty:
            filtered_df = clean_df.copy()

        n_unique = filtered_df[column].nunique()
        actual_bins = min(max_bins, max(10, min(n_unique, int(np.sqrt(len(filtered_df))))))
        title = title or f"Distribution of {column}"

        chart = (
            alt.Chart(filtered_df)
            .mark_bar(opacity=0.7, stroke='white', strokeWidth=0.5)
            .encode(
                x=alt.X(f"{column}:Q", bin=alt.Bin(maxbins=actual_bins), title=column),
                y=alt.Y("count()", title="Count"),
                tooltip=[
                    alt.Tooltip(f"{column}:Q", title=column, format='.3f'),
                    alt.Tooltip("count()", title="Count")
                ],
                color=alt.value('steelblue')
            )
            .properties(height=height, width=width, title=alt.TitleParams(text=title, fontSize=14, anchor='start'))
            .interactive()
        )
        return chart
    except Exception as e:
        logger.error(f"Error creating histogram for column '{column}': {e}")
        st.error(f"Failed to create histogram: {e}")
        return None

def alt_line_plot(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    title: Optional[str] = None,
    height: int = 250,
    width: int = 400
) -> Optional[alt.Chart]:
    """
    Create an Altair line plot for time-series data.
    Args:
        df: Input DataFrame.
        time_col: Datetime column for the x-axis.
        value_col: Numeric column for the y-axis.
        title: Plot title.
        height: Plot height in pixels.
        width: Plot width in pixels.
    Returns:
        Altair Chart object or None if an error occurs.
    """
    try:
        if not isinstance(df, pd.DataFrame):
            logger.error("df must be a pandas DataFrame")
            st.error("Invalid data type for line plot")
            return None
        if time_col not in df.columns or value_col not in df.columns:
            logger.error(f"Columns {time_col} or {value_col} not found.")
            st.error(f"Columns {time_col} or {value_col} not found.")
            return None
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            logger.error(f"Column {time_col} is not datetime.")
            st.error(f"Column {time_col} is not datetime.")
            return None
        if not pd.api.types.is_numeric_dtype(df[value_col]):
            logger.error(f"Column {value_col} is not numeric.")
            st.error(f"Column {value_col} is not numeric.")
            return None
        if df[[time_col, value_col]].dropna().empty:
            st.warning(f"No valid data for {time_col} and {value_col} after removing missing values.")
            return None
        chart = alt.Chart(df).mark_line().encode(
            x=alt.X(f"{time_col}:T", title=time_col),
            y=alt.Y(f"{value_col}:Q", title=value_col),
            tooltip=[time_col, value_col]
        ).properties(
            height=height,
            width=width,
            title=title or f"{value_col} over Time"
        ).interactive()
        return chart
    except Exception as e:
        logger.error(f"Error creating line plot: {e}")
        st.error(f"Error creating line plot: {e}")
        return None

def word_cloud(
    df: pd.DataFrame,
    column: str,
    title: Optional[str] = None,
    width: int = 400,
    height: int = 200
) -> None:
    """
    Generate a word cloud for a text column.
    Args:
        df: Input DataFrame.
        column: Text column to visualize.
        title: Word cloud title.
        width: Image width in pixels.
        height: Image height in pixels.
    """
    try:
        if not isinstance(df, pd.DataFrame):
            logger.error("df must be a pandas DataFrame")
            st.error("Invalid data type for word cloud")
            return
        if column not in df.columns:
            logger.error(f"Column {column} not found.")
            st.error(f"Column {column} not found.")
            return
        text = ' '.join(df[column].astype(str).str.strip())
        if not text.strip():
            st.warning(f"Column {column} contains no valid text data.")
            return
        wordcloud = WordCloud(width=width, height=height, max_words=100).generate(text)
        plt.figure(figsize=(width / 100, height / 100))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title or f"Word Cloud for {column}")
        st.pyplot(plt)
        plt.close()  # Close the figure to free memory
    except Exception as e:
        logger.error(f"Error creating word cloud: {e}")
        st.error(f"Error creating word cloud: {e}")
