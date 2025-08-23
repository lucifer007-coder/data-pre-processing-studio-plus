import streamlit as st
from utils.data_utils import dtype_split, _arrowize, sample_for_preview
from preprocessing.steps import smooth_time_series, resample_time_series
from utils.viz_utils import alt_line_plot
import logging

logger = logging.getLogger(__name__)

def section_time_series():
    st.header("⏳ Time-Series Preprocessing")
    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first.")
        return

    try:
        num_cols, _ = dtype_split(df)
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
        # Include string columns that look like dates
        date_like_cols = [
            col for col in df.columns
            if col not in datetime_cols and df[col].astype(str).str.match(r'\d{4}-\d{2}-\d{2}', na=False).any()
        ]
        datetime_cols.extend(date_like_cols)
        if not num_cols or not datetime_cols:
            st.info("No numeric or datetime/date-like columns available for time-series preprocessing.")
            return

        # Warning for large datasets
        if len(df) > 100_000:
            st.warning(
                "Large dataset detected (>100,000 rows). Previews will be sampled, and full processing may be slow. Consider sampling the dataset."
            )

        with st.expander("Smooth Time-Series", expanded=True):
            st.markdown("**Smooth numeric data to reduce noise.**")
            smooth_col = st.selectbox(
                "Numeric column to smooth",
                num_cols,
                help="Select a numeric column for smoothing."
            )
            smooth_method = st.radio(
                "Smoothing method",
                ["moving_average", "savitzky_golay"],
                horizontal=True,
                help="Moving Average: Simple averaging over a window. Savitzky-Golay: Polynomial fitting for smoother curves."
            )
            window = st.slider(
                "Window size",
                3, 21, 5, 2,
                help="Size of the smoothing window. Must be odd for Savitzky-Golay.",
                disabled=smooth_method == "moving_average"
            )
            if smooth_method == "savitzky_golay":
                window = window if window % 2 == 1 else window + 1
                st.caption(f"Adjusted window to {window} (odd) for Savitzky-Golay.")
            interpolate = st.selectbox(
                "Interpolate missing values",
                ["linear", "ffill", "bfill", "none"],
                help="Method to handle missing values before smoothing."
            )
            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("🔍 Preview Smoothing", help="Preview smoothing on a sampled dataset"):
                    if not smooth_col:
                        st.warning("Please select a numeric column.")
                        return
                    with st.spinner("Generating smoothing preview..."):
                        prev = sample_for_preview(df)
                        preview_df, msg = smooth_time_series(
                            prev, smooth_col, window, smooth_method,
                            interpolate if interpolate != "none" else None, preview=True
                        )
                        st.session_state.last_preview = (preview_df, msg)
                        st.info(msg)
                        st.dataframe(_arrowize(preview_df.head(10)))
                        # Find a datetime column for visualization
                        vis_time_col = datetime_cols[0] if datetime_cols else None
                        if vis_time_col:
                            chart = alt_line_plot(preview_df, vis_time_col, smooth_col, "Smoothed Time-Series Preview")
                            if chart:
                                st.altair_chart(chart, use_container_width=True)
            with c2:
                if st.button("📦 Add Smoothing to Pipeline", help="Add smoothing step to pipeline"):
                    if not smooth_col:
                        st.warning("Please select a numeric column.")
                        return
                    step = {
                        "kind": "smooth_time_series",
                        "params": {
                            "column": smooth_col,
                            "window": window,
                            "method": smooth_method,
                            "interpolate": interpolate if interpolate != "none" else None
                        }
                    }
                    st.session_state.pipeline.append(step)
                    st.success("Added smoothing step to pipeline.")

        with st.expander("Resample Time-Series", expanded=True):
            st.markdown("**Resample data to a uniform time frequency.**")
            time_col = st.selectbox(
                "Datetime column",
                datetime_cols,
                help="Select a datetime or date-like column for resampling."
            )
            freq = st.text_input(
                "Resampling frequency",
                value="1H",
                help="Enter a pandas frequency string (e.g., '1H' for hourly, '1D' for daily, '1W' for weekly)."
            )
            agg_func = st.selectbox(
                "Aggregation function",
                ["mean", "sum", "last", "first"],
                help="How to aggregate data during resampling."
            )
            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("🔍 Preview Resampling", help="Preview resampling on a sampled dataset"):
                    if not time_col:
                        st.warning("Please select a datetime column.")
                        return
                    with st.spinner("Generating resampling preview..."):
                        prev = sample_for_preview(df)
                        preview_df, msg = resample_time_series(prev, time_col, freq, agg_func, preview=True)
                        st.session_state.last_preview = (preview_df, msg)
                        st.info(msg)
                        st.dataframe(_arrowize(preview_df.head(10)))
                        if smooth_col and time_col in preview_df.columns:
                            chart = alt_line_plot(preview_df, time_col, smooth_col, "Resampled Time-Series Preview")
                            if chart:
                                st.altair_chart(chart, use_container_width=True)
            with c2:
                if st.button("📦 Add Resampling to Pipeline", help="Add resampling step to pipeline"):
                    if not time_col:
                        st.warning("Please select a datetime column.")
                        return
                    step = {
                        "kind": "resample_time_series",
                        "params": {"time_column": time_col, "freq": freq, "agg_func": agg_func}
                    }
                    st.session_state.pipeline.append(step)
                    st.success("Added resampling step to pipeline.")

        c3 = st.columns(1)[0]
        with c3:
            if st.button("🔄 Reset Selection", help="Clear all selections"):
                st.rerun()
    except Exception as e:
        logger.error(f"Error in section_time_series: {e}")
        st.error(f"Error in time-series section: {e}")