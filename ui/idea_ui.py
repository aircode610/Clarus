"""
Idea Capture UI components for the Clarus application.

This module contains the UI components for the Idea Capture mode.
"""

import streamlit as st
from .common import display_assertions, export_assertions_button, clear_assertions_button, next_mode_button
import voice.streamlit_voice as streamlit_voice


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

        # Voice input
        streamlit_voice.whisper_voice_to_text(
            start_prompt="🎤 Voice Input",
            stop_prompt="🛑 Stop Recording",
        )

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
            st.rerun()
    
    with col2:
        st.subheader("📝 Extracted Assertions")
        
        # Display current assertions
        display_assertions(st.session_state.assertions)
        
        # Action buttons
        st.markdown("---")
        col_clear, col_export = st.columns(2)
        
        with col_clear:
            clear_assertions_button()
        
        with col_export:
            export_assertions_button()
        
        # Next button to go to Structure mode
        st.markdown("---")
        next_mode_button("Idea Capture", "Structure", "Move to Structure mode to organize your assertions")
