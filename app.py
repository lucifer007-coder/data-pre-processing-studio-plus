import sys
import os
import streamlit as st
from components.sidebar import sidebar_navigation
from components.sections.upload import section_upload
from components.sections.recommendations import section_recommendations
from components.sections.missing_data import section_missing_data
from components.sections.inconsistency import section_inconsistency
from components.sections.outliers import section_outliers
from components.sections.duplicates import section_duplicates
from components.sections.encoding import section_encoding
from components.sections.scaling import section_scaling
from components.sections.imbalanced import section_imbalanced
from components.sections.pipeline_preview import section_pipeline_preview
from components.sections.time_series import section_time_series
from components.sections.text import section_text
from components.sections.feature_engineering import section_feature_engineering
from components.sections.dashboard import section_dashboard
from components.sections.export import section_export
from session import init_session

# Fix module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    try:
        # Initialize session state
        init_session()

        st.set_page_config(
            page_title="Data Preprocessing Studio",
            page_icon="🧹",
            layout="wide",
            initial_sidebar_state="collapsed",
        )

        section = sidebar_navigation()

        if section == "Upload":
            section_upload()
        elif section == "Recommendations":
            section_recommendations()
        elif section == "Missing Data":
            section_missing_data()
        elif section == "Data Inconsistency":
            section_inconsistency()
        elif section == "Outliers / Noisy Data":
            section_outliers()
        elif section == "Duplicates":
            section_duplicates()
        elif section == "Categorical Encoding":
            section_encoding()
        elif section == "Scaling / Normalization":
            section_scaling()
        elif section == "Imbalanced Data":
            section_imbalanced()
        elif section == "Feature Engineering":
            section_feature_engineering()
        elif section == "Pipeline & Preview":
            section_pipeline_preview()
        elif section == "Time-Series Preprocessing":
            section_time_series()
        elif section == "Text Preprocessing":
            section_text()
        elif section == "Dashboard":
            section_dashboard()
        elif section == "Export":
            section_export()

        st.markdown("---")
        st.caption(
            "Pro tip: Add multiple steps to the pipeline and apply them in one go. "
            "Use the Dashboard to understand how your dataset changed."
        )
    except Exception as e:
        st.error(f"Application error: {e}")

if __name__ == "__main__":
    main()
