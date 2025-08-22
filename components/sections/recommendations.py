import streamlit as st
import pandas as pd
from utils.recommendations import PreprocessingRecommendations
from utils.data_utils import _arrowize

def section_recommendations():
    st.header("🔍 Data Quality Recommendations")
    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first.")
        return

    try:
        recommender = PreprocessingRecommendations()
        recommendations = recommender.analyze_dataset(df)
        
        if not recommendations:
            st.info("No significant issues detected in the dataset.")
            return

        st.subheader("Recommended Preprocessing Steps")
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
                if rec['type'] == 'missing_data':
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"📦 Add Imputation", key=f"impute_{i}"):
                            strategy = 'mean' if rec['column'] in dtype_split(df)[0] else 'mode'
                            step = {
                                "kind": "impute",
                                "params": {"columns": [rec['column']], "strategy": strategy}
                            }
                            st.session_state.pipeline.append(step)
                            st.success("Added imputation step to pipeline.")
                    with col2:
                        if st.button(f"📦 Add Drop", key=f"drop_{i}"):
                            step = {
                                "kind": "drop_missing",
                                "params": {"axis": "rows", "columns": [rec['column']]}
                            }
                            st.session_state.pipeline.append(step)
                            st.success("Added drop step to pipeline.")
                elif rec['type'] == 'outliers':
                    if st.button(f"📦 Add Outlier Handling", key=f"outliers_{i}"):
                        step = {
                            "kind": "outliers",
                            "params": {"columns": [rec['column']], "method": "iqr", "factor": 1.5}
                        }
                        st.session_state.pipeline.append(step)
                        st.success("Added outlier handling step to pipeline.")
                elif rec['type'] == 'bias_risk':
                    if st.button(f"📦 Add Rebalancing", key=f"bias_{i}"):
                        step = {
                            "kind": "rebalance",
                            "params": {"target": rec['column'], "method": "oversample", "ratio": 1.0}
                        }
                        st.session_state.pipeline.append(step)
                        st.success("Added rebalancing step to pipeline.")
                elif rec['type'] == 'duplicates':
                    if st.button(f"📦 Add Duplicate Removal", key=f"duplicates_{i}"):
                        step = {
                            "kind": "duplicates",
                            "params": {"subset": None, "keep": "first"}
                        }
                        st.session_state.pipeline.append(step)
                        st.success("Added duplicate removal step to pipeline.")
                elif rec['type'] == 'data_type_mismatch':
                    if st.button(f"📦 Add Type Conversion", key=f"type_{i}"):
                        type_val = "numeric" if "numeric" in rec['suggestion'] else "datetime"
                        step = {
                            "kind": "type_convert" if type_val == "numeric" else "standardize_dates",
                            "params": {"column": rec['column'], "type": type_val} if type_val == "numeric" else {"columns": [rec['column']]}
                        }
                        st.session_state.pipeline.append(step)
                        st.success("Added type conversion step to pipeline.")
                elif rec['type'] == 'skewness':
                    transform = 'log' if 'log' in rec['suggestion'] else 'square_root'
                    if st.button(f"📦 Add {transform.title()} Transformation", key=f"skew_{i}"):
                        step = {
                            "kind": "skewness_transform",
                            "params": {"column": rec['column'], "transform": transform}
                        }
                        st.session_state.pipeline.append(step)
                        st.success(f"Added {transform} transformation step to pipeline.")
                elif rec['type'] == 'sensitive_data':
                    if st.button(f"📦 Add PII Masking", key=f"pii_{i}"):
                        step = {
                            "kind": "mask_pii",
                            "params": {"column": rec['column'], "pii_types": ["email", "phone", "credit_card"]}
                        }
                        st.session_state.pipeline.append(step)
                        st.success("Added PII masking step to pipeline.")
                elif rec['type'] == 'auto_pipeline':
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"🔍 Preview Auto Pipeline", key=f"preview_auto_{i}"):
                            preview_df, messages = recommender.preview_pipeline(df, rec['pipeline'])
                            st.session_state.last_preview = (preview_df, "\n".join(messages))
                            st.write("**Preview Results**:")
                            for msg in messages:
                                st.write(msg)
                            st.dataframe(_arrowize(preview_df.head(10)))
                    with col2:
                        if st.button(f"📦 Add Auto Pipeline", key=f"auto_{i}"):
                            for step in rec['pipeline']:
                                st.session_state.pipeline.append(step)
                            st.success("Added auto pipeline steps to pipeline.")

        if st.button("🔄 Clear Recommendations", help="Reset recommendations"):
            st.session_state.last_preview = None
            st.rerun()

    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        st.error(f"Error generating recommendations: {e}")
