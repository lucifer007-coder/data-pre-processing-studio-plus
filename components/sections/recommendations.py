import streamlit as st
import pandas as pd
from utils.recommendations import PreprocessingRecommendations
from utils.data_utils import _arrowize
from utils.stats_utils import compute_basic_stats
import logging

logger = logging.getLogger(__name__)

def section_recommendations():
    st.header("🔍 Data Quality Recommendations")
    df = st.session_state.get('df', None)
    if df is None:
        st.warning("Upload a dataset first.")
        return

    try:
        recommender = PreprocessingRecommendations()
        recommendations = recommender.analyze_dataset(df)
        
        if not recommendations:
            st.info("No significant issues detected in the dataset.")
            return

        st.subheader("Summary of Recommendations")
        summary_data = []
        for i, rec in enumerate(recommendations, 1):
            summary_data.append({
                "ID": i,
                "Type": rec['type'].replace('_', ' ').title(),
                "Priority": f"{rec.get('priority', 0.5):.2f}",
                "Severity": rec.get('severity', 'medium').title(),
                "Column": rec.get('column', rec.get('columns', ['N/A'])[0]),
                "Details": rec.get('suggestion', '')
            })
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

        st.subheader("Detailed Recommendations")
        for i, rec in enumerate(recommendations, 1):
            with st.expander(f"{i}. {rec['type'].replace('_', ' ').title()} (Priority: {rec.get('priority', 0.5):.2f})"):
                st.write(f"**Suggestion**: {rec['suggestion']}")
                if 'column' in rec:
                    st.write(f"**Column**: {rec['column']}")
                if 'columns' in rec:
                    st.write(f"**Affected Columns**: {', '.join(rec['columns'])}")
                if 'missing_count' in rec:
                    st.write(f"**Missing Count**: {rec['missing_count']} ({rec['missing_ratio']*100:.1f}%)")
                if 'count' in rec:
                    st.write(f"**Count**: {rec['count']}")

                # Visualization
                chart = recommender.visualize_recommendation(df, rec)
                if chart:
                    st.altair_chart(chart, use_container_width=True)

                # Actionable buttons
                col1, col2 = st.columns(2)
                if rec['type'] == 'missing_data':
                    with col1:
                        if st.button(f"📦 Add Imputation", key=f"impute_{i}", help="Add imputation step to pipeline"):
                            strategy = 'mean' if rec['column'] in dtype_split(df)[0] else 'mode'
                            step = {
                                "kind": "impute",
                                "params": {"columns": [rec['column']], "strategy": strategy}
                            }
                            st.session_state.pipeline.append(step)
                            st.success("Added imputation step to pipeline.")
                    with col2:
                        if st.button(f"📦 Add Drop", key=f"drop_{i}", help="Add drop step to pipeline"):
                            step = {
                                "kind": "drop_missing",
                                "params": {"axis": "rows", "columns": [rec['column']]}
                            }
                            st.session_state.pipeline.append(step)
                            st.success("Added drop step to pipeline.")
                elif rec['type'] == 'outliers':
                    with col1:
                        if st.button(f"📦 Add Outlier Handling", key=f"outliers_{i}", help="Add outlier handling step"):
                            step = {
                                "kind": "outliers",
                                "params": {"columns": [rec['column']], "method": "iqr", "factor": 1.5}
                            }
                            st.session_state.pipeline.append(step)
                            st.success("Added outlier handling step to pipeline.")
                elif rec['type'] == 'bias_risk':
                    with col1:
                        if st.button(f"📦 Add Rebalancing", key=f"bias_{i}", help="Add rebalancing step"):
                            step = {
                                "kind": "rebalance",
                                "params": {"target": rec['column'], "method": "oversample", "ratio": 1.0}
                            }
                            st.session_state.pipeline.append(step)
                            st.success("Added rebalancing step to pipeline.")
                elif rec['type'] == 'duplicates':
                    with col1:
                        if st.button(f"📦 Add Duplicate Removal", key=f"duplicates_{i}", help="Add duplicate removal step"):
                            step = {
                                "kind": "duplicates",
                                "params": {"subset": None, "keep": "first"}
                            }
                            st.session_state.pipeline.append(step)
                            st.success("Added duplicate removal step to pipeline.")
                elif rec['type'] == 'data_type_mismatch':
                    with col1:
                        if st.button(f"📦 Add Type Conversion", key=f"type_{i}", help="Add type conversion step"):
                            type_val = "numeric" if "numeric" in rec['suggestion'] else "datetime" if "datetime" in rec['suggestion'] else "boolean"
                            step = {
                                "kind": "type_convert" if type_val != "datetime" else "standardize_dates",
                                "params": {"column": rec['column'], "type": type_val} if type_val != "datetime" else {"columns": [rec['column']]}
                            }
                            st.session_state.pipeline.append(step)
                            st.success("Added type conversion step to pipeline.")
                elif rec['type'] == 'skewness':
                    with col1:
                        transform = 'log' if 'log' in rec['suggestion'] else 'square_root'
                        if st.button(f"📦 Add {transform.title()} Transformation", key=f"skew_{i}", help=f"Add {transform} transformation"):
                            step = {
                                "kind": "skewness_transform",
                                "params": {"column": rec['column'], "transform": transform}
                            }
                            st.session_state.pipeline.append(step)
                            st.success(f"Added {transform} transformation step to pipeline.")
                elif rec['type'] == 'sensitive_data':
                    with col1:
                        if st.button(f"📦 Add PII Masking", key=f"pii_{i}", help="Add PII masking step"):
                            step = {
                                "kind": "mask_pii",
                                "params": {"column": rec['column'], "pii_types": ["email", "phone", "credit_card"]}
                            }
                            st.session_state.pipeline.append(step)
                            st.success("Added PII masking step to pipeline.")
                elif rec['type'] == 'auto_pipeline':
                    with col1:
                        if st.button(f"🔍 Preview Auto Pipeline", key=f"preview_auto_{i}", help="Preview the auto pipeline"):
                            preview_df, messages = recommender.preview_pipeline(df, rec['pipeline'])
                            st.session_state.last_preview = (preview_df, "\n".join(messages))
                            st.write("**Preview Results**:")
                            for msg in messages:
                                st.write(msg)
                            st.dataframe(preview_df.head(10))
                            # Display before/after stats
                            before_stats = compute_basic_stats(df)
                            after_stats = compute_basic_stats(preview_df)
                            st.write("**Preview Statistics**")
                            col_stats1, col_stats2 = st.columns(2)
                            with col_stats1:
                                st.write("Before")
                                st.write(f"Shape: {before_stats['shape']}")
                                st.write(f"Missing Values: {before_stats['missing_total']}")
                            with col_stats2:
                                st.write("After Preview")
                                st.write(f"Shape: {after_stats['shape']}")
                                st.write(f"Missing Values: {after_stats['missing_total']}")
                    with col2:
                        if st.button(f"📦 Add Auto Pipeline", key=f"auto_{i}", help="Add all auto pipeline steps"):
                            for step in rec['pipeline']:
                                st.session_state.pipeline.append(step)
                            st.success("Added auto pipeline steps to pipeline.")

        if st.button("🔄 Clear Recommendations", help="Reset recommendations and clear preview"):
            st.session_state.last_preview = None
            st.rerun()

    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        st.error(f"Error generating recommendations: {e}")
