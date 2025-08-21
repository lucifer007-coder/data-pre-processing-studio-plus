import streamlit as st
from session import reset_all, undo_last

def sidebar_navigation() -> str:
    # Logo at the top
    st.sidebar.image(
        "https://raw.githubusercontent.com/streamlit/streamlit/develop/docs/_static/img/streamlit-logo-light-text.svg",
        use_container_width=True,
    )

    # ── Dark-Mode Toggle ──
    if st.sidebar.checkbox("🌙 Dark mode"):
        st.markdown(
            """
            <style>
            /* Main app area */
            .main {
                background-color: #0e1117;
                color: #fafafa;
            }
            /* Sidebar */
            [data-testid="stSidebar"] {
                background-color: #161b22;
                color: #fafafa;
            }
            /* Text & widgets inside sidebar */
            [data-testid="stSidebar"] .css-1d391kg,
            [data-testid="stSidebar"] .css-17eq0hr {
                color: #fafafa;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

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
    st.sidebar.caption("Tip: Use 'Add to Pipeline' on each section, then run them together.")

    return section
