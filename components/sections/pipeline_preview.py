import logging
import streamlit as st
import time
from utils.data_utils import dtype_split, _arrowize, sample_for_preview
from utils.viz_utils import alt_histogram
from utils.stats_utils import compute_basic_stats
from preprocessing.pipeline import run_pipeline
from session import push_history

logger = logging.getLogger(__name__)

def section_pipeline_preview():
    st.header("🧪 Pipeline & Preview")
    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first.")
        return

    try:
        st.subheader("Queued Pipeline Steps")
        if not st.session_state.pipeline:
            st.info("Pipeline is empty. Add steps from the sections on the left.")
        else:
            for i, step in enumerate(st.session_state.pipeline, start=1):
                st.write(f"{i}. **{step['kind'].replace('_', ' ').title()}** — {step.get('params', {})}")

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🔍 Preview Full Pipeline", help="Preview the pipeline on a sampled dataset"):
                prev = sample_for_preview(df)
                preview_df, msgs = run_pipeline(prev, st.session_state.pipeline, preview=True)
                st.session_state.last_preview = (preview_df, "\n".join(msgs))
                st.success("Pipeline preview complete.")
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
            if st.button("🚮 Clear Pipeline", help="Clear all pipeline steps"):
                st.session_state.pipeline = []
                st.session_state.last_preview = None
                st.info("Cleared pipeline.")
                st.rerun()

        with col3:
            progress_placeholder = st.empty()
            if st.button("✅ Apply Pipeline to Data", help="Apply the pipeline to the full dataset"):
                if not st.session_state.pipeline:
                    st.warning("Pipeline is empty.")
                    return
                push_history("Before pipeline")
                msgs = []
                tmp_df = st.session_state.df.copy()
                steps = st.session_state.pipeline.copy()
                total = len(steps)
                for i, step in enumerate(steps, start=1):
                    progress_placeholder.progress(
                        i / total,
                        text=f"Applying step {i}/{total}: {step['kind'].replace('_', ' ').title()}"
                    )
                    tmp_df, msg = run_pipeline(tmp_df, [step])
                    msgs.append(msg[0])
                    time.sleep(0.05)
                progress_placeholder.empty()
                st.session_state.df = tmp_df
                st.session_state.changelog.extend([f"✅ {m}" for m in msgs])
                st.success("Applied pipeline to full dataset.")
                st.session_state.pipeline = []

        st.markdown("---")
        if st.session_state.last_preview is not None:
            prev_df, msg = st.session_state.last_preview
            st.subheader("Latest Preview Result")
            with st.expander("Preview Summary", expanded=True):
                st.code(msg or "", language="text")
            st.dataframe(_arrowize(prev_df.head(10)))

            num_cols, _ = dtype_split(prev_df)
            if num_cols:
                st.subheader("Visualize Preview")
                column = st.selectbox(
                    "Preview histogram for column",
                    num_cols,
                    help="Select a numeric column to compare distributions."
                )
                left, right = st.columns(2)
                with left:
                    st.write("**Current Data**")
                    chart1 = alt_histogram(sample_for_preview(df), column, "Current Data")
                    if chart1:
                        st.altair_chart(chart1, use_container_width=True)
                with right:
                    st.write("**Preview Data**")
                    chart2 = alt_histogram(prev_df, column, "Preview Data")
                    if chart2:
                        st.altair_chart(chart2, use_container_width=True)

    except Exception as e:
        logger.error(f"Error in section_pipeline_preview: {e}")
        st.error(f"Error in pipeline preview section: {e}")
