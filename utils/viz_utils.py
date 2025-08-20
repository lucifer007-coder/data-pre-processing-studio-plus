import logging
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from typing import Optional

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