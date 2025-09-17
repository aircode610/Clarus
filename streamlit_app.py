"""
Clarus - Streamlit UI

A multi-tab interface for the Clarus document structuring system.
Features Idea Capture, Structure, Review, and Prose modes.
"""

import streamlit as st
import json
from typing import List, Dict, Any

import streamlit_voice
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
        if "transcript" not in st.session_state or st.session_state.transcript is None:
            st.session_state.transcript = ""

        if st.session_state.get("message_sent", False):
            st.session_state.transcript = ""
            st.session_state.message_sent = False

        streamlit_voice.whisper_voice_to_text(
            start_prompt="🎤 Voice Input",
            stop_prompt="🛑 Stop Recording",
        )
        # st.text_area("text", label_visibility="collapsed", key="transcript", height=200)
        # st.chat_input("Your thoughts and ideas", key="transcript")

        if prompt := st.chat_input("Your thoughts and ideas", key="transcript"):
            # Add user message to chat
            st.session_state.message_sent = True
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
    """Structure mode - Edit and rearrange assertions with relationship analysis."""
    st.header("🏗️ Structure Mode")
    st.markdown("Edit and rearrange your assertions. AI will help identify relationships between them.")
    
    if not st.session_state.assertions:
        st.info("No assertions available. Go to Idea Capture mode first to extract some assertions.")
        return
    
    # Initialize structure analysis if not done yet
    if "structure_analysis_done" not in st.session_state:
        st.session_state.structure_analysis_done = False
    
    if "relationships" not in st.session_state:
        st.session_state.relationships = []
    
    if "grouped_assertions" not in st.session_state:
        st.session_state.grouped_assertions = {}
    
    # Run structure analysis if not done yet
    if not st.session_state.structure_analysis_done:
        with st.spinner("Analyzing relationships between assertions..."):
            try:
                # Ensure ClarusApp has the current assertions from session state
                st.session_state.clarus_app.current_assertions = st.session_state.assertions
                
                result = st.session_state.clarus_app.start_structure_analysis()
                
                if "final_relationships" in result:
                    st.session_state.relationships = result["final_relationships"]
                    st.session_state.structure_analysis_done = True
                    st.success(f"✅ Relationship analysis complete! Found {len(result['final_relationships'])} relationships.")
                else:
                    st.warning("No relationships found between assertions.")
            except Exception as e:
                st.error(f"Error running structure analysis: {e}")
                return
    
    # Create two columns: relationships on left, grouped assertions on right
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🔗 Relationships Found")
        
        if st.session_state.relationships:
            # Group relationships by type
            relationships_by_type = {}
            for rel in st.session_state.relationships:
                if rel.relationship_type not in relationships_by_type:
                    relationships_by_type[rel.relationship_type] = []
                relationships_by_type[rel.relationship_type].append(rel)
            
            for rel_type, relationships in relationships_by_type.items():
                with st.expander(f"{rel_type.title()} Relationships ({len(relationships)})", expanded=True):
                    for i, rel in enumerate(relationships, 1):
                        # Find assertion contents
                        assertion1 = next((a for a in st.session_state.assertions if a.id == rel.assertion1_id), None)
                        assertion2 = next((a for a in st.session_state.assertions if a.id == rel.assertion2_id), None)
                        
                        if assertion1 and assertion2:
                            st.write(f"**{i}.** {assertion1.content[:50]}{'...' if len(assertion1.content) > 50 else ''}")
                            st.write(f"    → **{rel_type.upper()}** → {assertion2.content[:50]}{'...' if len(assertion2.content) > 50 else ''}")
                            st.write(f"    *{rel.explanation}*")
                            
                            # Action buttons for each relationship
                            col_edit, col_delete = st.columns(2)
                            with col_edit:
                                if st.button("✏️", key=f"edit_rel_{i}_{rel_type}", help="Edit relationship"):
                                    st.session_state[f"editing_relationship_{i}_{rel_type}"] = True
                            
                            with col_delete:
                                if st.button("🗑️", key=f"delete_rel_{i}_{rel_type}", help="Delete relationship"):
                                    st.session_state.relationships.remove(rel)
                                    st.rerun()
                            
                            st.markdown("---")
        else:
            st.info("No relationships found between assertions.")
    
    with col2:
        st.subheader("📋 Grouped Assertions")
        
        # Group assertions by relationships
        if st.session_state.relationships:
            # Create groups based on relationships
            groups = create_assertion_groups(st.session_state.assertions, st.session_state.relationships)
            
            for group_id, group_assertions in groups.items():
                with st.expander(f"Group {group_id} ({len(group_assertions)} assertions)", expanded=True):
                    for i, assertion in enumerate(group_assertions, 1):
                        st.write(f"**{i}.** {assertion.content}")
                        
                        # Action buttons for each assertion in group
                        col_edit, col_move = st.columns(2)
                        with col_edit:
                            if st.button("✏️", key=f"edit_group_{group_id}_{i}", help="Edit assertion"):
                                st.session_state[f"editing_group_assertion_{group_id}_{i}"] = True
                        
                        with col_move:
                            if st.button("↔️", key=f"move_group_{group_id}_{i}", help="Move to different group"):
                                st.session_state[f"moving_assertion_{group_id}_{i}"] = True
        else:
            # If no relationships, show all assertions in one group
            st.write("**All Assertions (No relationships found):**")
            for i, assertion in enumerate(st.session_state.assertions, 1):
                st.write(f"{i}. {assertion.content}")
    
    # Action buttons at the bottom
    st.markdown("---")
    col_rerun, col_export, col_reset = st.columns(3)
    
    with col_rerun:
        if st.button("🔄 Re-analyze Relationships", help="Run structure analysis again"):
            # Ensure ClarusApp has the current assertions from session state
            st.session_state.clarus_app.current_assertions = st.session_state.assertions
            st.session_state.structure_analysis_done = False
            st.session_state.relationships = []
            st.rerun()
    
    with col_export:
        if st.button("📤 Export Structure", help="Export relationships as JSON"):
            if st.session_state.relationships:
                relationships_data = [rel.model_dump() for rel in st.session_state.relationships]
                st.download_button(
                    label="Download Relationships",
                    data=json.dumps(relationships_data, indent=2),
                    file_name="relationships.json",
                    mime="application/json"
                )
            else:
                st.warning("No relationships to export")
    
    with col_reset:
        if st.button("🗑️ Reset Structure", help="Clear all relationships"):
            st.session_state.relationships = []
            st.session_state.structure_analysis_done = False
            st.rerun()


def create_assertion_groups(assertions, relationships):
    """Create groups of assertions based on their relationships."""
    from collections import defaultdict
    
    # Create a graph of connected assertions
    graph = defaultdict(set)
    assertion_ids = {a.id for a in assertions}
    
    for rel in relationships:
        if rel.assertion1_id in assertion_ids and rel.assertion2_id in assertion_ids:
            graph[rel.assertion1_id].add(rel.assertion2_id)
            graph[rel.assertion2_id].add(rel.assertion1_id)
    
    # Find connected components (groups)
    visited = set()
    groups = {}
    group_id = 1
    
    for assertion in assertions:
        if assertion.id not in visited:
            # Start a new group
            group_assertions = []
            stack = [assertion.id]
            
            while stack:
                current_id = stack.pop()
                if current_id not in visited:
                    visited.add(current_id)
                    current_assertion = next((a for a in assertions if a.id == current_id), None)
                    if current_assertion:
                        group_assertions.append(current_assertion)
                    
                    # Add connected assertions to stack
                    for connected_id in graph[current_id]:
                        if connected_id not in visited:
                            stack.append(connected_id)
            
            if group_assertions:
                groups[group_id] = group_assertions
                group_id += 1
    
    return groups

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
