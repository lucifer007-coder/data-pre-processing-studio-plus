import logging
import streamlit as st
import time
from data_preprocessing_studio.utils.data_utils import dtype_split, _arrowize, sample_for_preview
from data_preprocessing_studio.utils.viz_utils import alt_histogram
from data_preprocessing_studio.preprocessing.pipeline import run_pipeline
from data_preprocessing_studio.session import push_history

logger = logging.getLogger(__name__)

def section_pipeline_preview():
    st.header("9) Pipeline & Preview")
    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first.")
        return

    try:
        st.subheader("Queued Steps")
        if not st.session_state.pipeline:
            st.info("Pipeline is empty. Add steps from the sections on the left.")
        else:
            for i, step in enumerate(st.session_state.pipeline, start=1):
                st.write(f"{i}. **{step['kind']}** — {step.get('params', {})}")

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🧪 Preview Full Pipeline (on sample)"):
                prev = sample_for_preview(df)
                preview_df, msgs = run_pipeline(prev, st.session_state.pipeline)
                st.session_state.last_preview = (preview_df, "\n".join(msgs))
                st.success("Pipeline preview complete.")
        with col2:
            if st.button("🚮 Clear Pipeline"):
                st.session_state.pipeline = []
                st.info("Cleared pipeline.")
        with col3:
            progress_placeholder = st.empty()
            if st.button("✅ Apply Pipeline to Data"):
                if not st.session_state.pipeline:
                    st.warning("Pipeline is empty.")
                else:
                    push_history("Before pipeline")
                    msgs = []
                    tmp_df = st.session_state.df.copy()
                    steps = st.session_state.pipeline.copy()
                    total = len(steps)
                    for i, step in enumerate(steps, start=1):
                        progress_placeholder.progress(i / total, text=f"Applying step {i}/{total}: {step['kind']}")
                        tmp_df, msg = run_pipeline(tmp_df, [step])  # Apply one step at a time
                        msgs.append(msg[0])  # run_pipeline returns a list of messages
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
            st.dataframe(_arrowize(prev_df))

            num_cols, _ = dtype_split(prev_df)
            if num_cols:
                column = st.selectbox("Preview histogram for column", num_cols)
                left, right = st.columns(2)
                with left:
                    chart1 = alt_histogram(sample_for_preview(st.session_state.df), column, "Current Data")
                    if chart1:
                        st.altair_chart(chart1, use_container_width=True)
                with right:
                    chart2 = alt_histogram(prev_df, column, "Preview Data")
                    if chart2:
                        st.altair_chart(chart2, use_container_width=True)
    except Exception as e:
        logger.error(f"Error in section_pipeline_preview: {e}")
        st.error(f"Error in pipeline preview section: {e}")