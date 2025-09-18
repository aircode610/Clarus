"""
Common UI components and utilities for the Clarus application.

This module contains shared UI components that are used across multiple modes.
"""

import streamlit as st
import json
from typing import List
from models import Assertion


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


def export_assertions_button():
    """Create an export button for assertions."""
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


def clear_assertions_button():
    """Create a clear all button for assertions."""
    if st.button("🗑️ Clear All", help="Clear all assertions"):
        st.session_state.assertions = []
        st.session_state.clarus_app.current_assertions = []
        st.session_state.messages = []
        st.session_state.deleted_assertions = []
        st.rerun()


def next_mode_button(current_mode: str, next_mode: str, help_text: str):
    """Create a next mode button."""
    if st.button(f"➡️ Next: {next_mode}", help=help_text):
        st.session_state.current_mode = next_mode
        st.rerun()
