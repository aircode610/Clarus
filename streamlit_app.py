"""
Clarus - Streamlit UI

A multi-tab interface for the Clarus document structuring system.
Features Idea Capture, Structure, Review, and Prose modes.
"""

import streamlit as st
import json
from typing import List, Dict, Any
from models import Assertion
from app import ClarusApp, create_clarus_app

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
    st.session_state.current_mode = "idea_capture"

def display_assertions(assertions: List[Assertion]):
    """Display the current assertions in a nice format."""
    if not assertions:
        st.info("No assertions extracted yet. Start a conversation to extract assertions from your ideas!")
        return
    
    st.subheader(f"📋 Current Assertions ({len(assertions)})")
    
    for i, assertion in enumerate(assertions, 1):
        # Check if we're editing this assertion
        editing_key = f"editing_assertion_{i}"
        if editing_key in st.session_state and st.session_state[editing_key]:
            # Edit mode
            with st.expander(f"✏️ Editing Assertion {i}", expanded=True):
                new_content = st.text_area(
                    "Edit assertion content:",
                    value=assertion.content,
                    key=f"edit_content_{i}",
                    height=100
                )
                
                col_save, col_cancel = st.columns(2)
                with col_save:
                    if st.button("💾 Save", key=f"save_{i}"):
                        # Update the assertion
                        updated_assertions = assertions.copy()
                        updated_assertions[i-1].content = new_content
                        st.session_state.assertions = updated_assertions
                        st.session_state.clarus_app.current_assertions = updated_assertions
                        del st.session_state[editing_key]
                        st.rerun()
                
                with col_cancel:
                    if st.button("❌ Cancel", key=f"cancel_{i}"):
                        del st.session_state[editing_key]
                        st.rerun()
        else:
            # Display mode
            with st.expander(f"Assertion {i}: {assertion.content[:50]}{'...' if len(assertion.content) > 50 else ''}", expanded=False):
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.write(f"**Content:** {assertion.content}")
                    st.write(f"**Source:** {assertion.source}")
                
                with col2:
                    # Action buttons for each assertion
                    col_edit, col_delete = st.columns(2)
                    with col_edit:
                        if st.button("✏️", key=f"edit_{i}", help="Edit assertion"):
                            st.session_state[editing_key] = True
                            st.rerun()
                    
                    with col_delete:
                        if st.button("🗑️", key=f"delete_{i}", help="Delete assertion"):
                            # Remove assertion
                            updated_assertions = [a for j, a in enumerate(assertions) if j != i-1]
                            st.session_state.assertions = updated_assertions
                            st.session_state.clarus_app.current_assertions = updated_assertions
                            
                            # Add to deleted assertions list for LLM context
                            if "deleted_assertions" not in st.session_state:
                                st.session_state.deleted_assertions = []
                            st.session_state.deleted_assertions.append(assertion.content)
                            
                            st.rerun()

def idea_capture_tab():
    """Idea Capture mode - Chat interface with assertions display."""
    st.header("💡 Idea Capture Mode")
    st.markdown("Share your thoughts and ideas. I'll help extract discrete, atomic assertions from your input.")
    
    # Create two columns: chat on left, assertions on right
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat with Clarus")
        
        # Display chat messages
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Share your ideas, thoughts, or feedback..."):
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Process with Clarus
            with st.chat_message("assistant"):
                with st.spinner("Processing your ideas..."):
                    try:
                        # Use the smart mixed input processor with deleted assertions context
                        result = st.session_state.clarus_app.process_mixed_input(
                            prompt, 
                            st.session_state.get("deleted_assertions", [])
                        )
                        
                        # Update assertions if they were modified
                        if "assertions" in result:
                            st.session_state.assertions = result["assertions"]
                            st.session_state.clarus_app.current_assertions = result["assertions"]
                        
                        # Extract AI response
                        ai_response = "I've processed your input."
                        if "messages" in result and result["messages"]:
                            # Get the last AI message
                            for msg in reversed(result["messages"]):
                                if hasattr(msg, 'content') and msg.__class__.__name__ == 'AIMessage':
                                    ai_response = msg.content
                                    break
                        elif "ai_response" in result:
                            ai_response = result["ai_response"]
                        
                        st.markdown(ai_response)
                        st.session_state.messages.append({"role": "assistant", "content": ai_response})
                        
                    except Exception as e:
                        error_msg = f"Sorry, I encountered an error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    with col2:
        st.subheader("📝 Extracted Assertions")
        
        # Display current assertions
        display_assertions(st.session_state.assertions)
        
        # Action buttons
        st.markdown("---")
        col_clear, col_export = st.columns(2)
        
        with col_clear:
            if st.button("🗑️ Clear All", help="Clear all assertions"):
                st.session_state.assertions = []
                st.session_state.clarus_app.current_assertions = []
                st.session_state.messages = []
                st.session_state.deleted_assertions = []
                st.rerun()
        
        with col_export:
            if st.button("📤 Export", help="Export assertions as JSON"):
                if st.session_state.assertions:
                    assertions_data = [a.model_dump() for a in st.session_state.assertions]
                    st.download_button(
                        label="Download Assertions",
                        data=json.dumps(assertions_data, indent=2),
                        file_name="assertions.json",
                        mime="application/json"
                    )
                else:
                    st.warning("No assertions to export")

def structure_tab():
    """Structure mode - Edit and rearrange assertions."""
    st.header("🏗️ Structure Mode")
    st.markdown("Edit and rearrange your assertions. AI will help identify relationships between them.")
    
    st.info("🚧 Structure mode is coming soon! This will allow you to edit and rearrange assertions with AI-powered relationship analysis.")
    
    if st.session_state.assertions:
        st.subheader("Current Assertions")
        for i, assertion in enumerate(st.session_state.assertions, 1):
            st.write(f"{i}. {assertion.content}")

def review_tab():
    """Review mode - Flag potential issues."""
    st.header("🔍 Review Mode")
    st.markdown("Review your assertions for potential issues like missing justification, vague language, or unclear logical flow.")
    
    st.info("🚧 Review mode is coming soon! This will flag potential issues in your assertions.")

def prose_tab():
    """Prose mode - Transform assertions into fluent text."""
    st.header("📖 Prose Mode")
    st.markdown("Transform your refined assertions into fluent, reader-friendly text.")
    
    st.info("🚧 Prose mode is coming soon! This will transform your assertions into well-structured prose.")

def main():
    """Main Streamlit application."""
    # Sidebar
    with st.sidebar:
        st.title("📝 Clarus")
        st.markdown("**Intelligent Document Structuring System**")
        
        st.markdown("---")
        
        # Mode selector
        st.subheader("Current Mode")
        current_mode = st.selectbox(
            "Select Mode",
            ["Idea Capture", "Structure", "Review", "Prose"],
            index=0
        )
        
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
    if current_mode == "Idea Capture":
        idea_capture_tab()
    elif current_mode == "Structure":
        structure_tab()
    elif current_mode == "Review":
        review_tab()
    elif current_mode == "Prose":
        prose_tab()

if __name__ == "__main__":
    main()
