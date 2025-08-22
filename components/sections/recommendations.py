import streamlit as st
import pandas as pd
from utils.recommendations import PreprocessingRecommendations

def section_recommendations():
    st.header("Recommendations")
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

        st.subheader("Data Quality Recommendations")
        for i, rec in enumerate(recommendations, 1):
            with st.expander(f"{i}. {rec['type'].replace('_', ' ').title()} (Severity: {rec.get('severity', 'medium')} | Priority: {rec.get('priority', 0.5):.2f})"):
                st.write(f"**Suggestion**: {rec['suggestion']}")
                if 'columns' in rec:
                    st.write(f"**Affected Columns**: {', '.join(rec['columns'])}")
                if 'column' in rec:
                    st.write(f"**Column**: {rec['column']}")
                if 'count' in rec:
                    st.write(f"**Count**: {rec['count']}")
                if 'pipeline' in rec:
                    st.write(f"**Suggested Pipeline**: {rec['suggestion'].split(': ')[1]}")
                
                # Visualization
                chart = recommender.visualize_recommendation(df, rec)
                if chart:
                    st.altair_chart(chart, use_container_width=True)

                # Actionable buttons
                if rec['type'] == 'missing_data':
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"📦 Add Imputation to Pipeline", key=f"impute_{i}"):
                            step = {
                                "kind": "impute",
                                "params": {"columns": rec['columns'], "strategy": "mean"}
                            }
                            st.session_state.pipeline.append(step)
                            st.success("Added imputation to pipeline.")
                    with col2:
                        if st.button(f"📦 Add Drop to Pipeline", key=f"drop_{i}"):
                            step = {
                                "kind": "drop_missing",
                                "params": {"axis": "rows", "columns": rec['columns']}
                            }
                            st.session_state.pipeline.append(step)
                            st.success("Added drop to pipeline.")
                elif rec['type'] == 'outliers':
                    if st.button(f"📦 Add Outlier Handling to Pipeline", key=f"outliers_{i}"):
                        step = {
                            "kind": "outliers",
                            "params": {"columns": [rec['column']], "method": "cap", "detect_method": "Z-score"}
                        }
                        st.session_state.pipeline.append(step)
                        st.success("Added outlier handling to pipeline.")
                elif rec['type'] == 'bias_risk':
                    if st.button(f"📦 Add Rebalancing to Pipeline", key=f"bias_{i}"):
                        step = {
                            "kind": "rebalance",
                            "params": {"target": rec['column'], "method": "oversample", "ratio": 1.0}
                        }
                        st.session_state.pipeline.append(step)
                        st.success("Added rebalancing to pipeline.")
                elif rec['type'] == 'duplicates':
                    if st.button(f"📦 Add Duplicate Removal to Pipeline", key=f"duplicates_{i}"):
                        step = {
                            "kind": "duplicates",
                            "params": {"subset": None, "keep": "first"}
                        }
                        st.session_state.pipeline.append(step)
                        st.success("Added duplicate removal to pipeline.")
                elif rec['type'] == 'data_type_mismatch':
                    if st.button(f"📦 Add Type Conversion to Pipeline", key=f"type_{i}"):
                        step = {
                            "kind": "type_convert",
                            "params": {"column": rec['column'], "type": "numeric"}
                        }
                        st.session_state.pipeline.append(step)
                        st.success("Added type conversion to pipeline.")
                elif rec['type'] == 'skewness':
                    transform = 'log' if df[rec['column']].min() > 0 else 'square_root'
                    if st.button(f"📦 Add {transform.title()} Transformation to Pipeline", key=f"skew_{i}"):
                        step = {
                            "kind": "skewness_transform",
                            "params": {"column": rec['column'], "transform": transform}
                        }
                        st.session_state.pipeline.append(step)
                        st.success(f"Added {transform} transformation to pipeline.")
                elif rec['type'] == 'sensitive_data':
                    if st.button(f"📦 Add Masking to Pipeline", key=f"pii_{i}"):
                        step = {
                            "kind": "mask_pii",
                            "params": {"column": rec['column']}
                        }
                        st.session_state.pipeline.append(step)
                        st.success("Added PII masking to pipeline.")
                elif rec['type'] == 'auto_pipeline':
                    if st.button(f"📦 Add Auto Pipeline", key=f"auto_{i}"):
                        for step in rec['pipeline']:
                            st.session_state.pipeline.append(step)
                        st.success("Added auto pipeline steps.")

    except Exception as e:

        st.error(f"Error generating recommendations: {e}")
