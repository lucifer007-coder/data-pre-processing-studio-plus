import streamlit as st
import pandas as pd
import altair as alt
import io
import json
import numpy as np
from scipy import stats
from utils.stats_utils import compute_basic_stats
from utils.viz_utils import alt_histogram
from utils.recommendations import PreprocessingRecommendations
from preprocessing.pipeline import run_pipeline

# ----------------------------------------------------------
# Helper: lightweight compare_stats (no circular import)
# ----------------------------------------------------------
def compare_stats(before, after) -> dict:
    if not isinstance(before, dict):
        before = {}
    if not isinstance(after, dict):
        after = {}
    shape_before = before.get("shape", (0, 0))
    shape_after = after.get("shape", (0, 0))
    missing_before = before.get("missing_total", 0)
    missing_after = after.get("missing_total", 0)
    cols_before = before.get("columns", [])
    cols_after = after.get("columns", [])

    rows_change = shape_after[0] - shape_before[0]
    cols_change = shape_after[1] - shape_before[1]
    missing_change = missing_after - missing_before

    rows_pct_change = (rows_change / max(shape_before[0], 1)) * 100
    missing_pct_change = (missing_change / max(missing_before, 1)) * 100

    added_columns = list(set(cols_after) - set(cols_before))
    removed_columns = list(set(cols_before) - set(cols_after))

    return {
        "shape_before": shape_before,
        "shape_after": shape_after,
        "rows_change": rows_change,
        "rows_pct_change": round(rows_pct_change, 2),
        "columns_change": cols_change,
        "missing_total_before": int(missing_before),
        "missing_total_after": int(missing_after),
        "missing_change": int(missing_change),
        "missing_pct_change": round(missing_pct_change, 2),
        "added_columns": added_columns,
        "removed_columns": removed_columns,
    }

# ----------------------------------------------------------
# Main dashboard section
# ----------------------------------------------------------
def section_dashboard_download():
    st.header("📊 Dashboard & Download")
    df = st.session_state.df
    raw = st.session_state.raw_df
    if df is None or raw is None:
        st.warning("Upload a dataset first.")
        return

    try:
        # cached stats
        raw_stats   = compute_basic_stats(raw)
        after_stats = compute_basic_stats(df)
        comp = compare_stats(raw_stats, after_stats)

        # ── 1. Data Quality Scorecard -----------------------------
        st.subheader("Data Quality Scorecard")
        recommender = PreprocessingRecommendations()
        recommendations = recommender.analyze_dataset(df)
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                with st.expander(
                    f"{i}. {rec['type'].replace('_', ' ').title()} "
                    f"(Severity: {rec.get('severity', 'medium')})"
                ):
                    st.write(f"**Suggestion**: {rec['suggestion']}")
                    if "columns" in rec:
                        st.write(f"**Affected Columns**: {', '.join(rec['columns'])}")
                    if "column" in rec:
                        st.write(f"**Column**: {rec['column']}")
                    if "count" in rec:
                        st.write(f"**Count**: {rec['count']}")
                    chart = recommender.visualize_recommendation(df, rec)
                    if chart:
                        st.altair_chart(chart, use_container_width=True)

        # ── 2. AI Bias Detection ----------------------------------
        st.subheader("AI Bias Detection")
        cat_cols = after_stats["categorical_cols"]
        if cat_cols:
            for col in cat_cols:
                value_counts = df[col].value_counts(normalize=True)
                if value_counts.max() > 0.8:
                    chart = alt.Chart(
                        pd.DataFrame(
                            {"Category": value_counts.index, "Proportion": value_counts.values}
                        )
                    ).mark_bar().encode(
                        x=alt.X("Category:N", title=col),
                        y=alt.Y("Proportion:Q", title="Proportion"),
                        color=alt.condition(
                            alt.datum.Proportion > 0.8,
                            alt.value("red"),
                            alt.value("steelblue"),
                        ),
                        tooltip=["Category", "Proportion"],
                    ).properties(title=f"Bias Risk in {col}", width=400, height=300)
                    st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No categorical columns for bias analysis.")

        # ── 3. Correlation Analysis -------------------------------
        st.subheader("Correlation Analysis")
        num_cols = after_stats["numeric_cols"]
        if num_cols:
            corr_matrix = df[num_cols].corr()
            corr_data = (
                corr_matrix.stack()
                .reset_index()
                .rename(columns={0: "Correlation", "level_0": "Variable1", "level_1": "Variable2"})
            )
            chart = alt.Chart(corr_data).mark_rect().encode(
                x=alt.X("Variable1:N", title=""),
                y=alt.Y("Variable2:N", title=""),
                color=alt.Color("Correlation:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1])),
                tooltip=["Variable1", "Variable2", "Correlation"],
            ).properties(title="Correlation Heatmap", width=400, height=400)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No numeric columns for correlation analysis.")

        # ── 4. Before / After Comparison --------------------------
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

        # ── 5. Statistical Test Results ---------------------------
        st.subheader("Statistical Test Results")
        if num_cols:
            for col in num_cols[:3]:
                clean_series = df[col].dropna()
                if len(clean_series) < 3:
                    st.info(f"Skipping Shapiro-Wilk test for {col}: insufficient non-NaN values.")
                    continue
                stat, p_value = stats.shapiro(clean_series)
                st.write(f"**{col} Normality Test (Shapiro-Wilk)**: Statistic={stat:.3f}, p-value={p_value:.3f}")
                if p_value < 0.05:
                    st.warning(f"{col} is not normally distributed (p < 0.05).")
                else:
                    st.info(f"{col} appears normally distributed (p ≥ 0.05).")
        else:
            st.info("No numeric columns for statistical tests.")

        # ── 6. Anomaly Detection Heatmap --------------------------
        st.subheader("Anomaly Detection Heatmap")
        if num_cols:
            z_scores = pd.DataFrame(index=df.index)
            for col in num_cols:
                clean_series = df[col].dropna()
                if len(clean_series) < 2:
                    z_scores[col] = 0
                    continue
                z_values = np.abs(stats.zscore(clean_series))
                z_series = pd.Series(z_values, index=clean_series.index).reindex(df.index, fill_value=0)
                z_scores[col] = z_series
            heatmap_data = (
                z_scores.stack()
                .reset_index()
                .rename(columns={0: "Z_Score", "level_1": "Column"})
            )
            chart = alt.Chart(heatmap_data).mark_rect().encode(
                x=alt.X("Column:N", title="Columns"),
                y=alt.Y("level_0:O", title="Row Index"),
                color=alt.Color("Z_Score:Q", scale=alt.Scale(scheme="reds")),
                tooltip=["Column", "Z_Score", "level_0"],
            ).properties(title="Anomaly Detection Heatmap (Z-Scores)", width=400, height=300)
            st.altair_chart(chart, use_container_width=True)
            st.caption("Note: Edge computing preview simulated locally; no IoT data provided.")
        else:
            st.info("No numeric columns for anomaly detection.")

        # ── 7. Dashboard Tabs -------------------------------------
        t1, t2, t3 = st.tabs(["Summary", "Distributions", "Change Log"])
        with t1:
            # Added / Removed columns
            if comp.get("added_columns"):
                st.success(f"Added columns: {', '.join(comp['added_columns'])}")
            if comp.get("removed_columns"):
                st.warning(f"Removed columns: {', '.join(comp['removed_columns'])}")

            # Missing values
            st.subheader("Missing by Column (After)")
            miss_after = pd.Series(after_stats["missing_by_col"])
            if miss_after.sum() > 0:
                st.dataframe(miss_after[miss_after > 0].rename("missing_count"))
            else:
                st.info("No missing values remaining!")

            # Column Impact Tracker
            st.subheader("📈 Column Impact Tracker")
            st.markdown(
                """
                **mean_shift**: Absolute change in the column’s average value after preprocessing.  
                **null_rate_change**: How much the missing-value ratio improved (positive = fewer NaNs).
                """,
                unsafe_allow_html=True,
            )
            before_num = raw_stats.get("describe_numeric", {})
            after_num  = after_stats.get("describe_numeric", {})
            impacts = []
            for col in after_num:
                if col in before_num:
                    b = before_num[col]
                    a = after_num[col]
                    mean_shift = abs(a.get("mean", 0) - b.get("mean", 0))
                    raw_null_rate   = raw_stats["missing_by_col"].get(col, 0) / max(raw_stats["shape"][0], 1)
                    clean_null_rate = after_stats["missing_by_col"].get(col, 0) / max(after_stats["shape"][0], 1)
                    null_shift = raw_null_rate - clean_null_rate
                    impacts.append({
                        "column": col,
                        "mean_shift": round(mean_shift, 4),
                        "null_rate_change": round(null_shift, 4),
                    })

            if impacts:
                impacts_df = pd.DataFrame(impacts).sort_values(
                    by="mean_shift", ascending=False
                ).reset_index(drop=True)
                st.dataframe(impacts_df, use_container_width=True)
            else:
                st.info("No numeric columns to compare.")

            # dtypes & numeric describe
            with st.expander("Dtypes (After)"):
                st.json(after_stats["dtypes"])
            with st.expander("Numeric Describe (After)"):
                if after_stats["numeric_cols"]:
                    st.dataframe(pd.DataFrame(after_stats["describe_numeric"]))
                else:
                    st.info("No numeric columns present.")

            # Paginated preview
            st.subheader("Full Data Preview (paginated)")
            page_size = st.slider("Rows per page", 100, 5_000, 1_000, key="dash_page_size")
            total_rows = len(df)
            max_page = max(1, total_rows // page_size + (1 if total_rows % page_size else 0))
            page = st.number_input("Page", 1, max_page, 1, key="dash_page_num")
            start = (page - 1) * page_size
            st.dataframe(df.iloc[start : start + page_size])

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

        # ── 8. Download section (CSV + JSON + PDF + Notebook) -----
        st.markdown("---")
        st.subheader("Download & Export")

        # CSV
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        st.download_button(
            "💾 Download CSV",
            data=buf.getvalue(),
            file_name="preprocessed_data.csv",
            mime="text/csv",
            help="Download the final processed dataset.",
        )

        # JSON Pipeline Recipe
        recipe = {
            "steps": st.session_state.pipeline,
            "shape_before": raw.shape,
            "shape_after": df.shape,
        }
        json_buf = io.StringIO(json.dumps(recipe, indent=2))
        st.download_button(
            "📋 Download Pipeline Recipe (JSON)",
            data=json_buf.getvalue(),
            file_name="pipeline_recipe.json",
            mime="application/json",
            help="Save the pipeline and replay it later.",
        )
