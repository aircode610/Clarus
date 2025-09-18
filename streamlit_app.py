"""
Clarus - Streamlit UI

A multi-tab interface for the Clarus document structuring system.
Features Idea Capture, Structure, Review, and Prose modes.
"""

import warnings
# Suppress pkg_resources deprecation warnings from ctranslate2/faster-whisper
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, module="ctranslate2")

import streamlit as st
from app import create_clarus_app
from ui import structure_tab, review_tab, prose_tab

# Import idea_capture_tab only if available (depends on voice dependencies)
try:
    from ui import idea_capture_tab
    _IDEA_UI_AVAILABLE = True
except ImportError:
    _IDEA_UI_AVAILABLE = False
    idea_capture_tab = None

# Page configuration
st.set_page_config(
    page_title="Clarus - Text Writing Assistant",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "clarus_app" not in st.session_state:
    st.session_state.clarus_app = create_clarus_app()
    st.session_state.messages = []
    st.session_state.assertions = []
    st.session_state.deleted_assertions = []
    st.session_state.current_mode = "Idea Capture"
    # Conflict resolution state
    st.session_state.conflicts_resolved = False
    st.session_state.chose_resolution_method = False
    st.session_state.ordered_graph_generated = False
    st.session_state.global_graph = None
    # Review state
    st.session_state.review_analysis_done = False
    st.session_state.paragraphs = []
    st.session_state.ordered_paragraphs = []
    st.session_state.all_issues = []
    st.session_state.accepted_issues = []
    st.session_state.declined_issues = []
    # Prose state
    st.session_state.generated_text = ""
    st.session_state.prose_style = "Academic"
    st.session_state.prose_temperature = 0.3
    st.session_state.prose_add_headings = False

def main():
    """Main Streamlit application."""
    # Sidebar
    with st.sidebar:
        st.title("📝 Clarus")
        st.markdown("**Intelligent Document Structuring System**")
        
        st.markdown("---")
        
        # Mode selector
        st.subheader("Current Mode")
        available_modes = ["Structure", "Review", "Prose"]
        if _IDEA_UI_AVAILABLE:
            available_modes = ["Idea Capture"] + available_modes
        
        current_mode = st.selectbox(
            "Select Mode",
            available_modes,
            index=available_modes.index(st.session_state.get("current_mode", available_modes[0])),
            help="Click to select mode"
        )
        
        # Update session state when mode changes
        default_mode = available_modes[0]
        if current_mode != st.session_state.get("current_mode", default_mode):
            st.session_state.current_mode = current_mode
            st.rerun()
        
        st.markdown("---")
        
        # Session info
        st.subheader("Session Info")
        st.write(f"**Assertions:** {len(st.session_state.assertions)}")
        st.write(f"**Messages:** {len(st.session_state.messages)}")
        
        # Reset session
        if st.button("🔄 Reset Session"):
            st.session_state.clarus_app.reset_session()
            st.session_state.messages = []
            st.session_state.assertions = []
            st.session_state.deleted_assertions = []
            # Reset conflict resolution state
            st.session_state.conflicts_resolved = False
            st.session_state.chose_resolution_method = False
            st.session_state.ordered_graph_generated = False
            st.session_state.global_graph = None
            # Reset review state
            st.session_state.review_analysis_done = False
            st.session_state.paragraphs = []
            st.session_state.ordered_paragraphs = []
            st.session_state.all_issues = []
            st.session_state.accepted_issues = []
            st.session_state.declined_issues = []
            # Reset prose state
            st.session_state.generated_text = ""
            st.session_state.prose_style = "Academic"
            st.session_state.prose_temperature = 0.3
            st.session_state.prose_add_headings = False
            st.rerun()
        
        st.markdown("---")
        
        # About
        st.subheader("About")
        st.markdown("""
        Clarus helps you transform raw thoughts into structured documents through four modes:
        
        1. **💡 Idea Capture** - Extract assertions from your ideas
        2. **🏗️ Structure** - Edit and arrange assertions  
        3. **🔍 Review** - Flag potential issues
        4. **📖 Prose** - Generate fluent text
        """)
    
    # Main content area
    current_mode = st.session_state.get("current_mode", available_modes[0])
    if current_mode == "Idea Capture" and _IDEA_UI_AVAILABLE:
        idea_capture_tab()
    elif current_mode == "Structure":
        structure_tab()
    elif current_mode == "Review":
        review_tab()
    elif current_mode == "Prose":
        prose_tab()
    else:
        st.error("Selected mode is not available. Please check your dependencies.")

if __name__ == "__main__":
    main()