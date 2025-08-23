import streamlit as st
from utils.data_utils import dtype_split, _arrowize, sample_for_preview
from preprocessing.steps import clean_text, extract_tfidf
from utils.viz_utils import word_cloud
import logging

logger = logging.getLogger(__name__)

def section_text():
    st.header("📝 Text Preprocessing")
    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first.")
        return

    try:
        _, cat_cols = dtype_split(df)
        if not cat_cols:
            st.info("No categorical/text columns available.")
            return

        # Warning for large datasets
        if len(df) > 100_000:
            st.warning(
                "Large dataset detected (>100,000 rows). Previews will be sampled, and full processing may be slow. Consider sampling the dataset."
            )

        with st.expander("Text Cleaning", expanded=True):
            st.markdown("**Clean text by normalizing and removing noise.**")
            text_col = st.selectbox(
                "Text column",
                cat_cols,
                help="Select a text column for cleaning."
            )
            remove_stopwords = st.checkbox(
                "Remove stopwords",
                help="Remove common English stopwords (e.g., 'the', 'and')."
            )
            lemmatize = st.checkbox(
                "Lemmatize",
                help="Reduce words to their base form (e.g., 'running' to 'run'). Requires NLTK WordNet."
            )
            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("🔍 Preview Text Cleaning", help="Preview text cleaning on a sampled dataset"):
                    if not text_col:
                        st.warning("Please select a text column.")
                        return
                    with st.spinner("Generating text cleaning preview..."):
                        prev = sample_for_preview(df)
                        # Validate text content
                        if prev[text_col].astype(str).str.strip().eq('').all():
                            st.error(f"Column {text_col} contains only empty or whitespace text.")
                            return
                        preview_df, msg = clean_text(prev, text_col, remove_stopwords, lemmatize, preview=True)
                        st.session_state.last_preview = (preview_df, msg)
                        st.info(msg)
                        st.dataframe(_arrowize(preview_df.head(10)))
                        word_cloud(preview_df, text_col, "Word Cloud of Cleaned Text")
            with c2:
                if st.button("📦 Add Cleaning to Pipeline", help="Add text cleaning step to pipeline"):
                    if not text_col:
                        st.warning("Please select a text column.")
                        return
                    step = {
                        "kind": "clean_text",
                        "params": {"column": text_col, "remove_stopwords": remove_stopwords, "lemmatize": lemmatize}
                    }
                    st.session_state.pipeline.append(step)
                    st.success("Added text cleaning step to pipeline.")

        with st.expander("TF-IDF Feature Extraction", expanded=True):
            st.markdown("**Extract numerical features from text using TF-IDF.**")
            max_features = st.number_input(
                "Max TF-IDF features",
                10, 200, 100, 10,
                help="Maximum number of features to extract. Limited by vocabulary size."
            )
            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("🔍 Preview TF-IDF Extraction", help="Preview TF-IDF extraction on a sampled dataset"):
                    if not text_col:
                        st.warning("Please select a text column.")
                        return
                    with st.spinner("Generating TF-IDF preview..."):
                        prev = sample_for_preview(df)
                        # Validate text content
                        if prev[text_col].astype(str).str.strip().eq('').all():
                            st.error(f"Column {text_col} contains only empty or whitespace text.")
                            return
                        preview_df, clean_msg = clean_text(prev, text_col, remove_stopwords, lemmatize, preview=True)
                        preview_df, tfidf_msg = extract_tfidf(preview_df, text_col, max_features, preview=True)
                        if "Extracted" in tfidf_msg:
                            actual_features = int(tfidf_msg.split()[1])
                            if actual_features < max_features:
                                st.warning(
                                    f"Only {actual_features} TF-IDF features extracted due to limited vocabulary size. "
                                    "Try reducing the number of features or using a larger text dataset."
                                )
                        st.session_state.last_preview = (preview_df, f"{clean_msg}\n{tfidf_msg}")
                        st.info(f"{clean_msg}\n{tfidf_msg}")
                        st.dataframe(_arrowize(preview_df.head(10)))
            with c2:
                if st.button("📦 Add TF-IDF to Pipeline", help="Add TF-IDF extraction step to pipeline"):
                    if not text_col:
                        st.warning("Please select a text column.")
                        return
                    step = {
                        "kind": "extract_tfidf",
                        "params": {"column": text_col, "max_features": max_features}
                    }
                    st.session_state.pipeline.append(step)
                    st.success("Added TF-IDF extraction step to pipeline.")

        c3 = st.columns(1)[0]
        with c3:
            if st.button("🔄 Reset Selection", help="Clear all selections"):
                st.rerun()
    except Exception as e:
        logger.error(f"Error in section_text: {e}")
        st.error(f"Error in text section: {e}")