import streamlit as st
from session import reset_all, undo_last

def sidebar_navigation() -> str:
    # Logo at the top
    st.sidebar.image(
        "https://raw.githubusercontent.com/streamlit/streamlit/develop/docs/_static/img/streamlit-logo-light-text.svg",
        use_container_width=True,
    )

    # ── Dark-Mode Toggle ──
    if st.sidebar.checkbox("🌙 Dark mode", value=False):
        st.markdown(
            """
            <style>
            /* 1. Entire app background & default text */
            .stApp, .main, html, body {
                background-color: #111827 !important;
                color: #f3f4f6 !important;
            }

            /* 2. Sidebar */
            [data-testid="stSidebar"] {
                background-color: #1f2937 !important;
                color: #f3f4f6 !important;
            }

            /* 3. All text & label colors */
            .css-1d391kg, .css-17eq0hr, .stMarkdown, .stText {
                color: #f3f4f6 !important;
            }

            /* 4. Inputs, sliders, select boxes */
            .stTextInput > div > div > input,
            .stNumberInput > div > div > input,
            .stSelectbox > div > div > select,
            .stSlider > div > div > div {
                background-color: #374151 !important;
                color: #f3f4f6 !important;
                border: 1px solid #4b5563 !important;
            }

            /* 5. Buttons */
            .stButton > button {
                background-color: #3b82f6 !important;
                color: #ffffff !important;
                border: none !important;
                border-radius: 0.375rem !important;
                box-shadow: 0 1px 2px 0 rgba(0,0,0,0.05) !important;
            }
            .stButton > button:hover {
                background-color: #2563eb !important;
            }

            /* 6. Tables & DataFrame */
            .dataframe {
                color: #f3f4f6 !important;
                background-color: #1f2937 !important;
            }
            .dataframe th {
                background-color: #374151 !important;
                color: #f3f4f6 !important;
            }
            .dataframe td {
                background-color: #111827 !important;
                color: #d1d5db !important;
            }

            /* 7. Expander headers */
            .streamlit-expanderHeader {
                background-color: #1f2937 !important;
                color: #f3f4f6 !important;
            }
            [data-testid="stExpander"] details summary {
                color: #f3f4f6 !important;
            }

            /* 8. Radio / Checkbox marks */
            .stRadio > label, .stCheckbox > label {
                color: #f3f4f6 !important;
            }

            /* 9. Metric cards */
            .stMetric {
                background-color: #1f2937 !important;
                color: #f3f4f6 !important;
                border-radius: 0.375rem !important;
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
