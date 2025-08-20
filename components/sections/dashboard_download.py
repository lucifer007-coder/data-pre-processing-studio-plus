import streamlit as st
import pandas as pd
import altair as alt
import io
import numpy as np
from scipy import stats
from utils.stats_utils import compute_basic_stats, compare_stats
from utils.viz_utils import alt_histogram
from utils.recommendations import PreprocessingRecommendations

def section_dashboard_download():
    st.header("📊 Dashboard & Download")
    df = st.session_state.df
    raw = st.session_state.raw_df
    if df is None or raw is None:
        st.warning("Upload a dataset first.")
        return

    try:
        before_stats = compute_basic_stats(raw)
        after_stats = compute_basic_stats(df)
        comp = compare_stats(before_stats, after_stats)

        # Recommendations Scorecard
        st.subheader("Data Quality Scorecard")
        recommender = PreprocessingRecommendations()
        recommendations = recommender.analyze_dataset(df)
        if not recommendations:
            st.info("No significant issues detected.")
        else:
            for i, rec in enumerate(recommendations, 1):
                with st.expander(f"{i}. {rec['type'].replace('_', ' ').title()} (Severity: {rec.get('severity', 'medium')})"):
                    st.write(f"**Suggestion**: {rec['suggestion']}")
                    if 'columns' in rec:
                        st.write(f"**Affected Columns**: {', '.join(rec['columns'])}")
                    if 'column' in rec:
                        st.write(f"**Column**: {rec['column']}")
                    if 'count' in rec:
                        st.write(f"**Outlier Count**: {rec['count']}")

        # AI Bias Detection Visuals
        st.subheader("AI Bias Detection")
        cat_cols = after_stats["categorical_cols"]
        if cat_cols:
            for col in cat_cols:
                value_counts = df[col].value_counts(normalize=True)
                if value_counts.max() > 0.8:
                    chart = alt.Chart(pd.DataFrame({
                        'Category': value_counts.index,
                        'Proportion': value_counts.values
                    })).mark_bar().encode(
                        x=alt.X('Category:N', title=col),
                        y=alt.Y('Proportion:Q', title='Proportion'),
                        color=alt.condition(
                            alt.datum.Proportion > 0.8,
                            alt.value('red'),
                            alt.value('steelblue')
                        ),
                        tooltip=['Category', 'Proportion']
                    ).properties(
                        title=f"Bias Risk in {col}",
                        width=400,
                        height=300
                    )
                    st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No categorical columns for bias analysis.")

        # Correlation Analysis
        st.subheader("Correlation Analysis")
        num_cols = after_stats["numeric_cols"]
        if num_cols:
            corr_matrix = df[num_cols].corr()
            corr_data = corr_matrix.stack().reset_index().rename(columns={0: 'Correlation', 'level_0': 'Variable1', 'level_1': 'Variable2'})
            chart = alt.Chart(corr_data).mark_rect().encode(
                x=alt.X('Variable1:N', title=''),
                y=alt.Y('Variable2:N', title=''),
                color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='redblue', domain=[-1, 1])),
                tooltip=['Variable1', 'Variable2', 'Correlation']
            ).properties(
                title="Correlation Heatmap",
                width=400,
                height=400
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No numeric columns for correlation analysis.")

        # Before/After Comparisons
        st.subheader("Before/After Transformations")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Before**")
            st.dataframe(raw.head())
        with col2:
            st.write("**After**")
            st.dataframe(df.head())
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Rows", f"{after_stats['shape'][0]}", f"{comp['rows_change']}")
        with c2:
            st.metric("Columns", f"{after_stats['shape'][1]}", f"{comp['columns_change']}")
        with c3:
            st.metric("Missing Values", f"{comp['missing_total_after']}", f"{comp['missing_change']}")
        with c4:
            st.metric("Memory (MB)", f"{after_stats.get('memory_usage_mb', 0):.2f}")

        # Statistical Test Results
        st.subheader("Statistical Test Results")
        if num_cols:
            for col in num_cols[:3]:  # Limit to 3 for brevity
                clean_series = df[col].dropna()
                if len(clean_series) < 3:  # Shapiro-Wilk requires at least 3 samples
                    st.info(f"Skipping Shapiro-Wilk test for {col}: insufficient non-NaN values.")
                    continue
                stat, p_value = stats.shapiro(clean_series)
                st.write(f"**{col} Normality Test (Shapiro-Wilk)**: Statistic={stat:.3f}, p-value={p_value:.3f}")
                if p_value < 0.05:
                    st.warning(f"{col} is not normally distributed (p < 0.05).")
                else:
                    st.info(f"{col} appears normally distributed (p >= 0.05).")
        else:
            st.info("No numeric columns for statistical tests.")

        # Anomaly Detection Heatmap
        st.subheader("Anomaly Detection Heatmap")
        if num_cols:
            z_scores = pd.DataFrame(index=df.index)
            for col in num_cols:
                clean_series = df[col].dropna()
                if len(clean_series) < 2:  # Z-score requires at least 2 values
                    z_scores[col] = 0
                    continue
                z_values = np.abs(stats.zscore(clean_series))
                z_series = pd.Series(z_values, index=clean_series.index).reindex(df.index, fill_value=0)
                z_scores[col] = z_series
            heatmap_data = z_scores.stack().reset_index().rename(columns={0: 'Z_Score', 'level_1': 'Column'})
            chart = alt.Chart(heatmap_data).mark_rect().encode(
                x=alt.X('Column:N', title='Columns'),
                y=alt.Y('level_0:O', title='Row Index'),
                color=alt.Color('Z_Score:Q', scale=alt.Scale(scheme='reds')),
                tooltip=['Column', 'Z_Score', 'level_0']
            ).properties(
                title="Anomaly Detection Heatmap (Z-Scores)",
                width=400,
                height=300
            )
            st.altair_chart(chart, use_container_width=True)
            st.caption("Note: Edge computing preview simulated locally; no IoT data provided.")
        else:
            st.info("No numeric columns for anomaly detection.")

        # Existing Dashboard Tabs
        t1, t2, t3 = st.tabs(["Summary", "Distributions", "Change Log"])
        with t1:
            if comp.get('added_columns'):
                st.success(f"Added columns: {', '.join(comp['added_columns'])}")
            if comp.get('removed_columns'):
                st.warning(f"Removed columns: {', '.join(comp['removed_columns'])}")
            st.subheader("Missing by Column (After)")
            miss_after = pd.Series(after_stats["missing_by_col"])
            if miss_after.sum() > 0:
                st.dataframe(miss_after[miss_after > 0].rename("missing_count"))
            else:
                st.info("No missing values remaining!")
            with st.expander("Dtypes (After)"):
                st.json(after_stats["dtypes"])
            with st.expander("Numeric Describe (After)"):
                if after_stats["numeric_cols"]:
                    st.dataframe(pd.DataFrame(after_stats["describe_numeric"]))
                else:
                    st.info("No numeric columns present.")

        with t2:
            if not num_cols:
                st.info("No numeric columns to visualize.")
            else:
                col = st.selectbox("Select numeric column", num_cols)
                a, b = st.columns(2)
                with a:
                    st.subheader("Before")
                    chart1 = alt_histogram(raw, col, f"Before: {col}")
                    if chart1:
                        st.altair_chart(chart1, use_container_width=True)
                with b:
                    st.subheader("After")
                    chart2 = alt_histogram(df, col, f"After: {col}")
                    if chart2:
                        st.altair_chart(chart2, use_container_width=True)

        with t3:
            if not st.session_state.changelog:
                st.info("No changes yet.")
            else:
                for i, msg in enumerate(st.session_state.changelog, start=1):
                    st.write(f"{i}. {msg}")

        st.markdown("---")
        st.subheader("Download Processed Data")
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        st.download_button(
            "💾 Download CSV",
            data=buf.getvalue(),
            file_name="preprocessed_data.csv",
            mime="text/csv",
            help="Download the final processed dataset as a CSV file.",
        )
        st.caption("All processing happens in your browser session.")
    except Exception as e:
        st.error(f"Error in dashboard section: {e}")
