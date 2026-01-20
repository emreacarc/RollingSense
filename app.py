"""
Streamlit Dashboard for RollingSense Predictive Maintenance System.
"""

import streamlit as st
from src.app_utils import load_model_and_preprocessor, load_model_report
from src.app_ui import render_live_monitor_tab, render_project_report_tab, render_failure_scenarios_tab, render_failure_insights_tab

# Page configuration
st.set_page_config(
    page_title="RollingSense - Predictive Maintenance",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .sidebar .logo-container {
        text-align: center;
        padding: 1rem 0;
    }
    .sidebar .logo-image {
        width: 80%;
        height: auto;
        margin: 0 auto;
        display: block;
    }
    .sidebar .subtitle {
        font-size: 0.85rem;
        color: #ffffff;
        font-weight: normal;
        margin-top: 0.5rem;
        text-align: center;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        color: #155724;
    }
    .danger-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        color: #721c24;
    }
    </style>
""", unsafe_allow_html=True)


def main():
    """Main Streamlit application."""
    
    # Sidebar with logo, subtitle, and navigation
    with st.sidebar:
        # Logo and subtitle
        st.markdown('<div class="logo-container">', unsafe_allow_html=True)
        st.image("logo/logo.jpg", use_container_width=True)
        st.markdown('<p class="subtitle">Predictive Maintenance System for Rolling Mills</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Modules menu
        st.markdown("### Modules")
        page = st.radio(
            "Select Page",
            options=["Live Monitor", "Sample Failure Scenarios", "Failure Insights & Analytics", "About Project"],
            label_visibility="collapsed"
        )
    
    # Load models (once, outside of page selection)
    model, preprocessor = load_model_and_preprocessor()
    model_report = load_model_report()
    
    # Render selected page
    if page == "Live Monitor":
        render_live_monitor_tab(model, preprocessor)
    elif page == "Sample Failure Scenarios":
        render_failure_scenarios_tab(model, preprocessor)
    elif page == "Failure Insights & Analytics":
        render_failure_insights_tab()
    elif page == "About Project":
        render_project_report_tab(model_report)


if __name__ == "__main__":
    main()
