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
    .contact-box {
        background-color: #1e3a5f;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-top: 2rem;
        color: #ffffff;
    }
    .contact-box h4 {
        color: #ffffff;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    .contact-box p {
        color: #ffffff;
        margin: 0.3rem 0;
    }
    .contact-box a {
        color: #b0b0b0;
        text-decoration: underline;
    }
    .contact-box a:hover {
        text-decoration: underline;
    }
    .contact-box .email-text {
        color: #b0b0b0;
    }
    /* Darker green for primary buttons */
    .stButton > button[kind="primary"] {
        background-color: #006400 !important;
        border-color: #006400 !important;
        color: white !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #004d00 !important;
        border-color: #004d00 !important;
        color: white !important;
    }
    .stButton > button[kind="primary"]:focus:not(:active) {
        background-color: #006400 !important;
        border-color: #006400 !important;
        color: white !important;
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
        
        # Contact Developer section
        st.markdown("---")
        st.markdown("""
        <div class="contact-box">
            <h4>Contact Developer</h4>
            <p style="font-weight: bold;">Emre AÇAR</p>
            <p><a href="https://www.linkedin.com/in/emreacarc/" target="_blank">My LinkedIn Profile</a></p>
            <p class="email-text">ar.emreacar@gmail.com</p>
        </div>
        """, unsafe_allow_html=True)
    
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
