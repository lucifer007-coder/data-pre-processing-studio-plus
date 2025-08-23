import streamlit as st
from session import reset_all, undo_last

def sidebar_navigation() -> str:
    st.sidebar.title("🧭 Navigation")
    section = st.sidebar.radio(
        "Go to section",
        [
            "Upload",
            "Recommendations",
            "Missing Data",
            "Data Inconsistency",
            "Outliers / Noisy Data",
            "Duplicates",
            "Categorical Encoding",
            "Scaling / Normalization",
            "Imbalanced Data",
            "Pipeline & Preview",
            "Dashboard & Download",
        ],
        help="Choose what you want to work on.",
        key="navigation"
    )

    st.sidebar.markdown("---")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.button("🔄 Reset All", help="Clear dataset, pipeline, and history."):
            reset_all()
            st.rerun()
    with c2:
        if st.button("↩️ Undo Last", help="Undo the last applied step."):
            undo_last()
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.caption("Tip: Upload a CSV file or .dps bundle, add preprocessing steps, export as a .dps bundle to resume later, and run them together.")

    return section
