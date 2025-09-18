"""
Structure UI components for the Clarus application.

This module contains the UI components for the Structure mode.
"""

import streamlit as st
import json
import networkx as nx
import plotly.graph_objects as go
from workflows.conflict_resolving import GlobalGraph
from workflows.structure import evaluate_relationship_quality
from models import Relationship
from .common import next_mode_button


def _add_or_update_relationship(assertion1_id: str, assertion2_id: str, relationship_type: str, evaluation_key: str):
    """Helper function to add or update a relationship and update the global graph."""
    # Get assertion objects
    assertion1_obj = next((a for a in st.session_state.assertions if a.id == assertion1_id), None)
    assertion2_obj = next((a for a in st.session_state.assertions if a.id == assertion2_id), None)
    
    if not assertion1_obj or not assertion2_obj:
        st.error("Invalid assertion selection.")
        return
    
    # Get the cached evaluation results
    if evaluation_key in st.session_state:
        confidence, reason, suggestion = st.session_state[evaluation_key]
    else:
        confidence, reason, suggestion = 50, "No evaluation available", "ADD"
    
    # Check if relationship already exists
    existing_rel = None
    for rel in st.session_state.relationships:
        if ((rel.assertion1_id == assertion1_id and rel.assertion2_id == assertion2_id) or
            (rel.assertion1_id == assertion2_id and rel.assertion2_id == assertion1_id)):
            existing_rel = rel
            break
    
    if existing_rel:
        # Update existing relationship
        existing_rel.relationship_type = relationship_type
        existing_rel.confidence = confidence / 100.0  # Convert to 0-1 scale
        existing_rel.explanation = f"{relationship_type} relationship between assertions. LLM Evaluation: {reason} (Suggestion: {suggestion})"
        st.success("Relationship updated successfully!")
    else:
        # Create new relationship
        new_rel = Relationship(
            assertion1_id=assertion1_id,
            assertion2_id=assertion2_id,
            relationship_type=relationship_type,
            confidence=confidence / 100.0,  # Convert to 0-1 scale
            explanation=f"{relationship_type} relationship between assertions. LLM Evaluation: {reason} (Suggestion: {suggestion})"
        )
        st.session_state.relationships.append(new_rel)
        st.success("Relationship added successfully!")
    
    # Clear edit state after successful operation
    if 'edit_relationship' in st.session_state:
        del st.session_state['edit_relationship']
    
    # Update global graph with new relationships (same logic as conflict resolution)
    st.session_state.global_graph = GlobalGraph(st.session_state.relationships, st.session_state.assertions)
    st.session_state.conflicts_resolved = False  # Need to re-resolve conflicts
    st.session_state.chose_resolution_method = False
    st.session_state.ordered_graph_generated = False
    
    st.rerun()


def _create_graph_visualization():
    """Create the interactive graph visualization."""
    # Graph interaction controls (must be defined before graph creation)
    st.markdown("---")
    st.subheader("🎛️ Graph Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**View Options**")
        show_labels = st.checkbox("Show assertion content", value=False)
    
    with col2:
        st.markdown("**Filter Relationships**")
        filter_type = st.selectbox("Filter by relationship type:", 
                                 ["All"] + list(set(rel.relationship_type for rel in st.session_state.relationships)),
                                 key="graph_filter")
        
        # Assertion filter for graph
        assertion_options = {f"{a.id}": f"Assertion {st.session_state.assertions.index(a) + 1}: {a.content}" 
                           for a in st.session_state.assertions}
        assertion_filter_options = ["All"] + list(assertion_options.keys())
        selected_assertions_for_graph = st.selectbox(
            "Show only this assertion",
            options=assertion_filter_options,
            format_func=lambda x: "All" if x == "All" else assertion_options[x],
            index=0,
            key="graph_assertion_filter",
            help="Select assertion to display in the graph"
        )
    
    with col3:
        st.markdown("**Layout Options**")
        layout_type = st.selectbox("Graph layout:", ["Spring", "Circular", "Hierarchical"], key="graph_layout")
    
    # Add interaction mode selector
    st.markdown("**Interaction Mode**")
    interaction_mode = st.radio(
        "Graph interaction:",
        ["Select Mode (Click edges to jump to details)", "Pan Mode (Drag to move graph)"],
        key="interaction_mode",
        horizontal=True
    )
    
    # Create the graph data structure
    G = nx.DiGraph()
    
    # Add nodes (assertions) - filter based on user selection
    assertions_to_show = st.session_state.assertions
    if selected_assertions_for_graph != "All":
        # Find all assertions that are involved in relationships with the selected assertion
        related_assertion_ids = set()
        related_assertion_ids.add(selected_assertions_for_graph)
        
        # Add all assertions that have relationships with the selected assertion
        for rel in st.session_state.relationships:
            if rel.assertion1_id == selected_assertions_for_graph:
                related_assertion_ids.add(rel.assertion2_id)
            elif rel.assertion2_id == selected_assertions_for_graph:
                related_assertion_ids.add(rel.assertion1_id)
        
        assertions_to_show = [a for a in st.session_state.assertions if a.id in related_assertion_ids]
    
    for i, assertion in enumerate(st.session_state.assertions):
        # Only add node if it's in the filtered list
        if assertion.id in [a.id for a in assertions_to_show]:
            G.add_node(assertion.id, 
                      label=f"Assertion {i+1}",
                      content=assertion.content,
                      assertion_num=i+1,
                      confidence=assertion.confidence,
                      source=assertion.source)
    
    # Filter relationships based on user selection
    filtered_relationships = st.session_state.relationships
    if filter_type != "All":
        filtered_relationships = [rel for rel in st.session_state.relationships if rel.relationship_type == filter_type]
    
    # Further filter relationships based on assertion filter
    if selected_assertions_for_graph != "All":
        # Show all relationships between the filtered assertions
        filtered_assertion_ids = [a.id for a in assertions_to_show]
        filtered_relationships = [rel for rel in filtered_relationships 
                                if rel.assertion1_id in filtered_assertion_ids 
                                and rel.assertion2_id in filtered_assertion_ids]
    
    # Add edges (relationships) with colors based on type - high contrast colors
    relationship_colors = {
        "evidence": "#32CD32",      # Lime Green
        "background": "#00CED1",    # Dark Turquoise  
        "cause": "#1E90FF",         # Dodger Blue
        "contrast": "#FF8C00",      # Orange
        "condition": "#FFD700",     # Gold
        "contradiction": "#FF4444"  # Red
    }
    
    # Add edges for filtered relationships only
    for rel in filtered_relationships:
        if rel.assertion1_id in G.nodes and rel.assertion2_id in G.nodes:
            G.add_edge(rel.assertion1_id, rel.assertion2_id, 
                      relationship_type=rel.relationship_type,
                      confidence=rel.confidence,
                      explanation=rel.explanation)
    
    # Use different layouts based on user selection
    if layout_type == "Circular":
        pos = nx.circular_layout(G)
    elif layout_type == "Hierarchical":
        # Try to create a hierarchical layout
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        except:
            # Fallback to spring layout if graphviz is not available
            pos = nx.spring_layout(G, k=3, iterations=50)
    else:  # Spring layout
        pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Prepare data for Plotly - create separate traces for each relationship type
    edge_traces = []
    
    for rel_type, color in relationship_colors.items():
        edge_x = []
        edge_y = []
        edge_hover_text = []
        
        for edge in G.edges():
            edge_data = G[edge[0]][edge[1]]
            if edge_data.get('relationship_type') == rel_type:
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                
                confidence = edge_data.get('confidence', 0)
                explanation = edge_data.get('explanation', '')
                
                # Get assertion content for hover
                assertion1_data = G.nodes[edge[0]]
                assertion2_data = G.nodes[edge[1]]
                
                hover_text = f"<b>{rel_type.upper()}</b><br>"
                hover_text += f"<b>From:</b> Assertion {assertion1_data['assertion_num']}<br>"
                hover_text += f"<b>To:</b> Assertion {assertion2_data['assertion_num']}<br>"
                hover_text += f"<b>Explanation:</b> {explanation}"
                
                edge_hover_text.append(hover_text)
        
        # Always create trace for legend, even if no edges of this type exist
        # Create the visible line trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=3, color=color),
            hoverinfo='text',
            hovertext=edge_hover_text,
            mode='lines',
            name=rel_type.title(),
            showlegend=True,
            legendgroup=rel_type
        )
        edge_traces.append(edge_trace)
            
        # Only create clickable trace if there are actual edges of this type
        if edge_x:
            # Create invisible clickable points along the edges
            clickable_x = []
            clickable_y = []
            clickable_hover = []
            clickable_customdata = []
            
            for edge in G.edges():
                edge_data = G[edge[0]][edge[1]]
                if edge_data.get('relationship_type') == rel_type:
                    # Get edge coordinates
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    
                    # Create clickable points along the edge (start, middle, end)
                    clickable_x.extend([x0, (x0 + x1) / 2, x1])
                    clickable_y.extend([y0, (y0 + y1) / 2, y1])
                    
                    # Get assertion data for hover
                    assertion1_data = G.nodes[edge[0]]
                    assertion2_data = G.nodes[edge[1]]
                    confidence = edge_data.get('confidence', 0)
                    explanation = edge_data.get('explanation', '')
                    
                    hover_text = f"<b>{rel_type.upper()}</b><br>"
                    hover_text += f"<b>From:</b> Assertion {assertion1_data['assertion_num']}<br>"
                    hover_text += f"<b>To:</b> Assertion {assertion2_data['assertion_num']}<br>"
                    hover_text += f"<b>Explanation:</b> {explanation}"
                    
                    clickable_hover.extend([hover_text, hover_text, hover_text])
                    clickable_customdata.extend([
                        [edge[0], edge[1], rel_type],
                        [edge[0], edge[1], rel_type],
                        [edge[0], edge[1], rel_type]
                    ])
            
            # Create invisible clickable trace
            clickable_trace = go.Scatter(
                x=clickable_x, y=clickable_y,
                mode='markers',
                marker=dict(size=20, color='rgba(0,0,0,0)', line=dict(width=0)),
                hoverinfo='text',
                hovertext=clickable_hover,
                customdata=clickable_customdata,
                showlegend=False,
                legendgroup=rel_type
            )
            edge_traces.append(clickable_trace)
    
    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_hover_text = []
    node_colors = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        node_data = G.nodes[node]
        assertion_num = node_data['assertion_num']
        content = node_data['content']
        confidence = node_data.get('confidence', 0)
        source = node_data.get('source', '')
        
        # Node label
        if show_labels:
            # Truncate content for display
            display_content = content[:50] + "..." if len(content) > 50 else content
            node_text.append(f"<b>Assertion {assertion_num}</b><br>{display_content}")
        else:
            node_text.append(f"<b>Assertion {assertion_num}</b>")
        
        # Hover text with full content
        hover_text = f"<b>Assertion {assertion_num}</b><br><br>"
        hover_text += f"<b>Content:</b> {content}"
        
        node_hover_text.append(hover_text)
        node_colors.append('#4A90E2')  # Darker blue for better contrast
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="middle center",
        hovertext=node_hover_text,
        marker=dict(
            size=50,
            color=node_colors,
            line=dict(width=2, color='#4A90E2')  # Same color as node
        ),
        showlegend=False
    )
    
    # Create arrow annotations for directional edges
    arrow_annotations = []
    for edge in G.edges():
        edge_data = G[edge[0]][edge[1]]
        rel_type = edge_data.get('relationship_type')
        
        if rel_type in relationship_colors:
            # Get edge coordinates
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            # Calculate arrow position (closer to the target node)
            arrow_x = x0 + 0.8 * (x1 - x0)
            arrow_y = y0 + 0.8 * (y1 - y0)
            
            # Calculate arrow direction
            dx = x1 - x0
            dy = y1 - y0
            length = (dx**2 + dy**2)**0.5
            if length > 0:
                dx_norm = dx / length
                dy_norm = dy / length
                
                # Create arrow annotation
                arrow_annotations.append(dict(
                    x=arrow_x,
                    y=arrow_y,
                    ax=arrow_x - 0.1 * dx_norm,
                    ay=arrow_y - 0.1 * dy_norm,
                    xref='x',
                    yref='y',
                    axref='x',
                    ayref='y',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowwidth=2,
                    arrowcolor=relationship_colors[rel_type],
                    opacity=0.8
                ))
    
    # Create the figure with all traces
    all_traces = edge_traces + [node_trace]
    fig = go.Figure(data=all_traces,
                   layout=go.Layout(
                       title=dict(
                           text="Assertion Relationship Graph",
                           x=0.5,
                           font=dict(size=20, color='white')
                       ),
                       showlegend=True,
                       hovermode='closest',
                       dragmode='select' if "Select Mode" in interaction_mode else 'pan',  # Dynamic drag mode
                       margin=dict(b=20,l=5,r=5,t=40),
                       annotations=arrow_annotations + [ dict(
                           text="Hover for details" + (" • Click edges to jump to relationship details" if "Select Mode" in interaction_mode else " • Switch to Select Mode to click edges"),
                           showarrow=False,
                           xref="paper", yref="paper",
                           x=0.005, y=-0.002,
                           xanchor='left', yanchor='bottom',
                           font=dict(color='lightgray', size=12)
                       )],
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       plot_bgcolor='#1e1e1e',  # Dark background
                       paper_bgcolor='#1e1e1e',  # Dark paper background
                       font=dict(color='white'),  # White text for all elements
                       legend=dict(
                           bgcolor='rgba(0,0,0,0.5)',
                           bordercolor='white',
                           borderwidth=1,
                           font=dict(color='white')
                       )
                   ))
    
    # Display the graph with click events
    selected_data = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode=["points"])
    
    # Handle edge clicks to jump to relationship details
    if selected_data and 'selection' in selected_data and selected_data['selection']['points']:
        clicked_points = selected_data['selection']['points']
        if clicked_points:
            # Get the clicked point data
            point_data = clicked_points[0]
            
            # Debug: show what was clicked
            st.write("Debug - Clicked point data:", point_data)
            
            if 'customdata' in point_data and point_data['customdata']:
                # This is an edge click
                edge_info = point_data['customdata']
                st.write("Debug - Edge info:", edge_info)
                
                if len(edge_info) >= 2:
                    assertion1_id, assertion2_id = edge_info[0], edge_info[1]
                    # Find the relationship and scroll to it
                    for i, rel in enumerate(st.session_state.relationships):
                        if (rel.assertion1_id == assertion1_id and rel.assertion2_id == assertion2_id) or \
                           (rel.assertion1_id == assertion2_id and rel.assertion2_id == assertion1_id):
                            st.session_state['scroll_to_relationship'] = i
                            st.rerun()
                            break


def _create_relationship_management():
    """Create the relationship management interface."""
    st.markdown("---")
    st.subheader("🔗 Manage Relationships")
    
    # Create assertion selection interface
    if len(st.session_state.assertions) >= 2:
        st.markdown("### Create or Modify Relationship")
        
        # Create two main columns: left for selection, right for preview
        left_col, right_col = st.columns([1, 1])
        
        with left_col:
            if 'edit_relationship' in st.session_state:
                st.markdown("#### ✏️ Editing Relationship")
                st.info("You are currently editing a relationship. Make your changes and click Add/Change to save.")
            else:
                st.markdown("#### Selection")
            
            # First assertion selector
            assertion1_options = {f"{a.id}": f"Assertion {st.session_state.assertions.index(a) + 1}: {a.content}" 
                                for a in st.session_state.assertions}
            
            # Pre-fill if editing an existing relationship, otherwise use empty default
            default_assertion1 = None
            if 'edit_relationship' in st.session_state:
                default_assertion1 = st.session_state['edit_relationship']['assertion1']
            
            # Add empty option at the beginning
            assertion1_options_with_empty = {"": "Select first assertion..."}
            assertion1_options_with_empty.update(assertion1_options)
            
            selected_assertion1 = st.selectbox(
                "First Assertion",
                options=list(assertion1_options_with_empty.keys()),
                format_func=lambda x: assertion1_options_with_empty[x],
                index=list(assertion1_options_with_empty.keys()).index(default_assertion1) if default_assertion1 and default_assertion1 in assertion1_options_with_empty else 0,
                key="manage_rel_assertion1",
                help="Select the first assertion"
            )
            
            # Relationship type selector
            relationship_types = ["evidence", "background", "cause", "contrast", "condition", "contradiction"]
            relationship_labels = {
                "evidence": "Evidence - One assertion provides evidence/examples for another",
                "background": "Background - One assertion provides context/setting for another", 
                "cause": "Cause - One assertion causes/leads to another",
                "contrast": "Contrast - Assertions present different viewpoints/approaches",
                "condition": "Condition - One assertion is a prerequisite for another",
                "contradiction": "Contradiction - Assertions directly contradict/negate each other"
            }
            relationship_options = {rt: relationship_labels[rt] for rt in relationship_types}
            
            # Pre-fill if editing an existing relationship
            default_relationship = None
            if 'edit_relationship' in st.session_state:
                default_relationship = st.session_state['edit_relationship']['type']
            
            selected_relationship = st.selectbox(
                "Relationship Type",
                options=list(relationship_options.keys()),
                index=relationship_types.index(default_relationship) if default_relationship and default_relationship in relationship_types else 0,
                key="manage_rel_type",
                help="Select the type of relationship"
            )
            
            # Second assertion selector
            assertion2_options = {f"{a.id}": f"Assertion {st.session_state.assertions.index(a) + 1}: {a.content}" 
                                for a in st.session_state.assertions}
            
            # Pre-fill if editing an existing relationship
            default_assertion2 = None
            if 'edit_relationship' in st.session_state:
                default_assertion2 = st.session_state['edit_relationship']['assertion2']
            
            selected_assertion2 = st.selectbox(
                "Second Assertion",
                options=list(assertion2_options.keys()),
                format_func=lambda x: assertion2_options[x],
                index=list(assertion2_options.keys()).index(default_assertion2) if default_assertion2 and default_assertion2 in assertion2_options else 0,
                key="manage_rel_assertion2",
                help="Select the second assertion"
            )
            
            # Real-time evaluation when dropdowns change
            if selected_assertion1 and selected_assertion2 and selected_assertion1 != "":
                assertion1_obj = next((a for a in st.session_state.assertions if a.id == selected_assertion1), None)
                assertion2_obj = next((a for a in st.session_state.assertions if a.id == selected_assertion2), None)
                
                if assertion1_obj and assertion2_obj:
                    # Create a unique key for this combination to cache evaluation
                    evaluation_key = f"eval_{selected_assertion1}_{selected_assertion2}_{selected_relationship}"
                    
                    # Check if we already evaluated this combination
                    if evaluation_key not in st.session_state:
                        with st.spinner("Evaluating relationship quality..."):
                            confidence, reason, suggestion = evaluate_relationship_quality(
                                assertion1_obj.content, 
                                assertion2_obj.content, 
                                selected_relationship
                            )
                        st.session_state[evaluation_key] = (confidence, reason, suggestion)
                    else:
                        confidence, reason, suggestion = st.session_state[evaluation_key]
                    
                    # Display evaluation results
                    st.markdown("**Relationship Quality Evaluation:**")
                    
                    # Color code based on confidence and suggestion
                    if suggestion == "ADD":
                        st.success(f"✅ **{confidence}%** - {reason}")
                        st.info("💡 **Suggestion:** Add this relationship")
                    elif suggestion.startswith("MODIFY to"):
                        st.warning(f"⚠️ **{confidence}%** - {reason}")
                        st.info(f"💡 **Suggestion:** {suggestion}")
                    else:  # REMOVE
                        st.error(f"❌ **{confidence}%** - {reason}")
                        st.info("💡 **Suggestion:** Remove this relationship")
            
            # Check if we need to show confirmation buttons
            evaluation_key = f"eval_{selected_assertion1}_{selected_assertion2}_{selected_relationship}"
            needs_confirmation = False
            confirmation_type = None
            
            if (selected_assertion1 and selected_assertion2 and selected_assertion1 != "" and 
                evaluation_key in st.session_state):
                confidence, reason, suggestion = st.session_state[evaluation_key]
                if suggestion == "REMOVE" or suggestion.startswith("MODIFY to"):
                    needs_confirmation = True
                    confirmation_type = suggestion
            
            # Show confirmation buttons if needed
            if needs_confirmation:
                st.markdown("**⚠️ LLM Recommendation:**")
                if confirmation_type == "REMOVE":
                    st.warning("The LLM suggests removing this relationship. Are you sure you want to add it?")
                elif confirmation_type.startswith("MODIFY to"):
                    st.warning("The LLM suggests modifying this relationship. Are you sure you want to add it as-is?")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Yes, Add Anyway", key="confirm_add", type="primary"):
                        # Proceed with adding the relationship
                        _add_or_update_relationship(selected_assertion1, selected_assertion2, selected_relationship, evaluation_key)
                with col2:
                    if st.button("❌ Cancel", key="cancel_add"):
                        st.rerun()
            else:
                # Regular add/change button
                if st.button("➕ Add/Change", key="manage_relationship_btn", help="Add new or modify existing relationship", use_container_width=True):
                    # Validate selection
                    if not selected_assertion1 or selected_assertion1 == "":
                        st.error("Please select the first assertion.")
                        st.stop()
                    
                    # Proceed with adding the relationship
                    _add_or_update_relationship(selected_assertion1, selected_assertion2, selected_relationship, evaluation_key)
        
        with right_col:
            st.markdown("#### Interactive Preview")
            
            # Show preview of the relationship
            if selected_assertion1 and selected_assertion2 and selected_assertion1 != "":
                assertion1_obj = next((a for a in st.session_state.assertions if a.id == selected_assertion1), None)
                assertion2_obj = next((a for a in st.session_state.assertions if a.id == selected_assertion2), None)
                
                if assertion1_obj and assertion2_obj:
                    assertion1_num = st.session_state.assertions.index(assertion1_obj) + 1
                    assertion2_num = st.session_state.assertions.index(assertion2_obj) + 1
                    
                    # Create the formatted relationship text
                    relationship_verbs = {
                        "evidence": "serves as evidence for",
                        "background": "provides background for", 
                        "cause": "causes",
                        "contrast": "contrasts with",
                        "condition": "is a condition for",
                        "contradiction": "contradicts"
                    }
                    
                    verb = relationship_verbs.get(selected_relationship, selected_relationship)
                    
                    # Display in 3 rows format
                    st.markdown("**From:**")
                    st.info(f"Assertion {assertion1_num}: {assertion1_obj.content}")
                    
                    st.markdown("**Relationship:**")
                    st.success(f"{selected_relationship.upper()}: {verb}")
                    
                    st.markdown("**To:**")
                    st.info(f"Assertion {assertion2_num}: {assertion2_obj.content}")
                    
                    # Show the full formatted relationship
                    st.markdown("**Complete Relationship:**")
                    relationship_text = f"{assertion1_obj.content} (assertion {assertion1_num}) {verb} {assertion2_obj.content} (assertion {assertion2_num})"
                    st.markdown(f"*{relationship_text}*")
                else:
                    st.info("Please select both assertions to see the preview.")
            else:
                st.info("Please select both assertions to see the preview.")
    else:
        st.info("You need at least 2 assertions to create relationships.")


def _create_conflict_resolution():
    """Create the conflict resolution interface."""
    st.markdown("---")
    st.subheader("🔧 Conflict Resolution")
    st.markdown("Resolve any cycles or contradictions in your assertion relationships.")
    
    # Initialize global graph if not exists
    if not st.session_state.get("global_graph", None):
        st.session_state.global_graph = GlobalGraph(st.session_state.get("relationships", []), st.session_state.get("assertions", []))
    
    # Check if we need to resolve conflicts
    if not st.session_state.get("conflicts_resolved", False):
        # Method selection
        if not st.session_state.get("chose_resolution_method", False):
            st.markdown("**Choose resolution method:**")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🤖 Automatic Resolution", help="Automatically resolve conflicts using AI"):
                    st.session_state.automatic_resolution = True
                    st.session_state.chose_resolution_method = True
                    st.rerun()
            
            with col2:
                if st.button("👤 Manual Resolution", help="Manually choose which assertions to remove"):
                    st.session_state.automatic_resolution = False
                    st.session_state.chose_resolution_method = True
                    st.rerun()
        else:
            # Create assertions dictionary for UI
            assertions_dict = {a.id: a.content for a in st.session_state.assertions}
            
            # Run conflict resolution
            if st.session_state.global_graph.resolve_cycles_and_conflicts(
                st.session_state.automatic_resolution, 
                assertions_dict
            ):
                st.success("✅ All conflicts have been resolved!")
                
                # Generate ordered graph
                if not st.session_state.get("ordered_graph_generated", False):
                    st.session_state.global_graph.order_the_graph()
                    st.session_state.ordered_graph = st.session_state.global_graph.ordered_graph
                    
                    # Add any assertions not in the ordered graph
                    for assertion in st.session_state.assertions:
                        if assertion.id not in st.session_state.ordered_graph:
                            st.session_state.ordered_graph.append(assertion.id)
                    
                    st.session_state.ordered_graph_generated = True
                    st.session_state.conflicts_resolved = True
                    
                    # Update relationships in session state
                    st.session_state.relationships = st.session_state.global_graph.get_updated_relationships()
                    
                    st.rerun()
            else:
                st.info("🔄 Resolving conflicts... Please make your selections above.")
    else:
        st.success("✅ Conflicts have been resolved and graph has been ordered.")
        
        # Show summary
        if st.session_state.get("ordered_graph"):
            st.markdown(f"**Ordered assertions:** {len(st.session_state.ordered_graph)} assertions")
            st.markdown(f"**Remaining relationships:** {len(st.session_state.relationships)} relationships")


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
                
                if "evaluated_relationships" in result:
                    st.session_state.relationships = result["evaluated_relationships"]
                    st.session_state.structure_analysis_done = True
                    st.success(f"✅ Relationship analysis complete! Found {len(result['evaluated_relationships'])} relationships.")
                else:
                    st.warning("No relationships found between assertions.")
            except Exception as e:
                st.error(f"Error running structure analysis: {e}")
                return
    
    # Create assertions lookup for easy access
    assertions_dict = {a.id: a for a in st.session_state.assertions}
    
    # Main graph visualization interface
    st.subheader("🔗 Assertion Structure Graph")
    
    if st.session_state.relationships and len(st.session_state.assertions) > 1:
        _create_graph_visualization()
    
    # Relationship Management Section
    _create_relationship_management()
    
    # Action buttons at the bottom
    st.markdown("---")
    col_rerun, col_export, col_reset = st.columns(3)
    
    with col_rerun:
        if st.button("🔄 Re-analyze Relationships", help="Run structure analysis again"):
            # Ensure ClarusApp has the current assertions from session state
            st.session_state.clarus_app.current_assertions = st.session_state.assertions
            st.session_state.structure_analysis_done = False
            st.session_state.relationships = []
            # Reset conflict resolution state
            st.session_state.conflicts_resolved = False
            st.session_state.chose_resolution_method = False
            st.session_state.ordered_graph_generated = False
            st.session_state.global_graph = None
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
            # Reset conflict resolution state
            st.session_state.conflicts_resolved = False
            st.session_state.chose_resolution_method = False
            st.session_state.ordered_graph_generated = False
            st.session_state.global_graph = None
            st.rerun()
    
    # Conflict Resolution Section
    _create_conflict_resolution()
    
    # Next button to go to Review mode
    st.markdown("---")
    next_mode_button("Structure", "Review", "Move to Review mode to check for potential issues")
