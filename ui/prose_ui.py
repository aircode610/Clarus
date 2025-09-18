"""
Prose UI components for the Clarus application.

This module contains the UI components for the Prose mode.
"""

import streamlit as st
from workflows.prose import ProseWorkflow


def prose_tab():
    """Prose mode - Transform structured paragraphs into fluent text."""
    st.header("📖 Prose Mode")
    st.markdown("Transform your structured paragraphs into fluent, reader-friendly text.")
    
    # Check if paragraphs are available from review mode
    if not st.session_state.get("paragraphs"):
        st.warning("⚠️ No paragraphs available. Please complete the Review mode first.")
        st.markdown("Go back to Review mode to create structured paragraphs from your assertions.")
        return
    
    # Check if issues have been reviewed
    if st.session_state.get("all_issues"):
        total_issues = len(st.session_state.all_issues)
        resolved_issues = len(st.session_state.accepted_issues) + len(st.session_state.declined_issues)
        
        if resolved_issues < total_issues:
            st.warning(f"⚠️ You have {total_issues - resolved_issues} unresolved issues in Review mode.")
            st.markdown("Please review and accept/decline all issue suggestions before generating prose.")
            return
    
    st.success("✅ Ready to generate prose from your structured paragraphs!")
    
    # Show summary of what will be used for generation
    st.subheader("📋 Generation Input Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Paragraphs", len(st.session_state.paragraphs))
    with col2:
        st.metric("Accepted Issues", len(st.session_state.accepted_issues))
    with col3:
        st.metric("Declined Issues", len(st.session_state.declined_issues))
    with col4:
        st.metric("Total Issues", len(st.session_state.all_issues))
    
    # Style and generation controls
    st.subheader("🎛️ Generation Settings")
    
    col1, col2, col3 = st.columns([1, 1.5, 1.5])
    with col1:
        temperature = st.slider(
            "Creativity", 
            0.0, 1.0, 
            st.session_state.get("prose_temperature", 0.3), 
            0.1, 
            help="Lower = more formal/precise, Higher = more creative"
        )
        st.session_state.prose_temperature = temperature
    
    with col2:
        style = st.selectbox(
            "Style", 
            ["Academic", "Technical"], 
            index=["Academic", "Technical"].index(st.session_state.get("prose_style", "Academic")), 
            help="Choose the writing style for the generated text"
        )
        st.session_state.prose_style = style
    
    with col3:
        add_headings = st.checkbox(
            "Include suggested headings", 
            value=st.session_state.get("prose_add_headings", False),
            help="Add section headings to the generated text"
        )
        st.session_state.prose_add_headings = add_headings
    
    # Generation button
    button_label = f"Generate {style} Text"
    spinner_label = f"Generating {style.lower()} text..."
    
    if st.button(button_label, type="primary", use_container_width=True):
        try:
            with st.spinner(spinner_label):
                # Initialize prose workflow
                prose_workflow = ProseWorkflow()
                
                # Run the prose workflow
                result = prose_workflow.run(
                    paragraphs=st.session_state.paragraphs,
                    all_issues=st.session_state.all_issues,
                    accepted_issues=st.session_state.accepted_issues,
                    declined_issues=st.session_state.declined_issues,
                    style=style,
                    temperature=temperature,
                    add_headings=add_headings
                )
                
                if "generated_text" in result:
                    st.session_state.generated_text = result["generated_text"]
                    st.success("✅ Prose generation complete!")
                    
                    # Show summary if available
                    if "chat_summary" in result:
                        st.info(result["chat_summary"])
                else:
                    st.error("Failed to generate prose text.")
                    
        except Exception as e:
            st.error(f"Error generating prose: {e}")
    
    # Show generated text if available
    if st.session_state.get("generated_text"):
        st.subheader(f"Generated {st.session_state.prose_style} Text")
        
        # Display the text in a text area
        st.text_area(
            f"{st.session_state.prose_style} Text", 
            value=st.session_state.generated_text, 
            height=420, 
            key="generated_text_display"
        )
        
        # Download button
        st.download_button(
            label=f"Download {st.session_state.prose_style} Text (.txt)",
            data=st.session_state.generated_text,
            file_name=f"{st.session_state.prose_style.lower()}_text.txt",
            mime="text/plain"
        )
    
    # Action buttons
    st.markdown("---")
    col_rerun, col_reset = st.columns(2)
    
    with col_rerun:
        if st.button("🔄 Regenerate Text", help="Generate new text with current settings"):
            # Clear the generated text to force regeneration
            st.session_state.generated_text = ""
            st.rerun()
    
    with col_reset:
        if st.button("🗑️ Reset Prose", help="Clear generated text and reset settings"):
            st.session_state.generated_text = ""
            st.session_state.prose_style = "Academic"
            st.session_state.prose_temperature = 0.3
            st.session_state.prose_add_headings = False
            st.rerun()
