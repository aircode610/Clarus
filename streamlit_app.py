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
    st.session_state.current_mode = "Idea Capture"

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
        
        # Next button to go to Structure mode
        st.markdown("---")
        if st.button("➡️ Next: Structure Mode", help="Move to Structure mode to organize your assertions"):
            st.session_state.current_mode = "Structure"
            st.rerun()

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
    
    # Create assertions lookup for easy access
    assertions_dict = {a.id: a for a in st.session_state.assertions}
    
    # Main graph visualization interface
    st.subheader("🔗 Assertion Structure Graph")
    
    if st.session_state.relationships and len(st.session_state.assertions) > 1:
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
    import networkx as nx
    import plotly.graph_objects as go
    import plotly.express as px
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes (assertions)
    for i, assertion in enumerate(st.session_state.assertions):
        G.add_node(assertion.id, 
                  label=f"Assertion {i+1}",
                  content=assertion.content,
                  assertion_num=i+1,
                  confidence=assertion.confidence,
                  source=assertion.source        )
    
    # Filter relationships based on user selection
    filtered_relationships = st.session_state.relationships
    if filter_type != "All":
        filtered_relationships = [rel for rel in st.session_state.relationships if rel.relationship_type == filter_type]
    
    # Add edges (relationships) with colors based on type - high contrast colors
    relationship_colors = {
        "evidence": "#32CD32",      # Lime Green
        "background": "#00CED1",    # Dark Turquoise  
        "cause": "#1E90FF",         # Dodger Blue
        "contrast": "#FF4444",      # Bright Red
        "condition": "#FFD700"      # Gold
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
        
        if edge_x:  # Only create trace if there are edges of this type
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
    
    # Relationship Management Section
    st.markdown("---")
    st.subheader("🔗 Manage Relationships")
    
    # Create assertion selection interface
    if len(st.session_state.assertions) >= 2:
        st.markdown("### Create or Modify Relationship")
        
        # Create two main columns: left for selection, right for preview
        left_col, right_col = st.columns([1, 1])
        
        with left_col:
            st.markdown("#### Selection")
            
            # First assertion selector
            assertion1_options = {f"{a.id}": f"Assertion {st.session_state.assertions.index(a) + 1}: {a.content}" 
                                for a in st.session_state.assertions}
            selected_assertion1 = st.selectbox(
                "First Assertion",
                options=list(assertion1_options.keys()),
                format_func=lambda x: assertion1_options[x],
                key="manage_rel_assertion1",
                help="Select the first assertion"
            )
            
            # Relationship type selector
            relationship_types = ["evidence", "background", "cause", "contrast", "condition"]
            relationship_labels = {
                "evidence": "Evidence - One assertion provides evidence/examples for another",
                "background": "Background - One assertion provides context/setting for another", 
                "cause": "Cause - One assertion causes/leads to another",
                "contrast": "Contrast - Assertions present opposing viewpoints",
                "condition": "Condition - One assertion is a prerequisite for another"
            }
            relationship_options = {rt: relationship_labels[rt] for rt in relationship_types}
            
            selected_relationship = st.selectbox(
                "Relationship Type",
                options=list(relationship_options.keys()),
                key="manage_rel_type",
                help="Select the type of relationship"
            )
            
            # Second assertion selector
            assertion2_options = {f"{a.id}": f"Assertion {st.session_state.assertions.index(a) + 1}: {a.content}" 
                                for a in st.session_state.assertions}
            selected_assertion2 = st.selectbox(
                "Second Assertion",
                options=list(assertion2_options.keys()),
                format_func=lambda x: assertion2_options[x],
                key="manage_rel_assertion2",
                help="Select the second assertion"
            )
            
            # Add/Change button
            if st.button("➕ Add/Change", key="manage_relationship_btn", help="Add new or modify existing relationship", use_container_width=True):
                # Check if relationship already exists
                existing_rel = None
                for rel in st.session_state.relationships:
                    if ((rel.assertion1_id == selected_assertion1 and rel.assertion2_id == selected_assertion2) or
                        (rel.assertion1_id == selected_assertion2 and rel.assertion2_id == selected_assertion1)):
                        existing_rel = rel
                        break
                
                if existing_rel:
                    # Update existing relationship
                    existing_rel.relationship_type = selected_relationship
                    existing_rel.explanation = f"{selected_relationship} relationship between assertions"
                    st.success("Relationship updated successfully!")
                else:
                    # Create new relationship
                    from models import Relationship
                    new_rel = Relationship(
                        assertion1_id=selected_assertion1,
                        assertion2_id=selected_assertion2,
                        relationship_type=selected_relationship,
                        confidence=0.8,
                        explanation=f"{selected_relationship} relationship between assertions"
                    )
                    st.session_state.relationships.append(new_rel)
                    st.success("Relationship added successfully!")
                st.rerun()
        
        with right_col:
            st.markdown("#### Interactive Preview")
            
            # Show preview of the relationship
            if selected_assertion1 and selected_assertion2:
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
                        "condition": "is a condition for"
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
    
    # Show all existing relationships
    st.markdown("---")
    st.subheader("📋 All Relationships")
    
    if st.session_state.relationships:
        for i, rel in enumerate(st.session_state.relationships):
            assertion1 = assertions_dict.get(rel.assertion1_id)
            assertion2 = assertions_dict.get(rel.assertion2_id)
            
            if assertion1 and assertion2:
                assertion1_num = st.session_state.assertions.index(assertion1) + 1
                assertion2_num = st.session_state.assertions.index(assertion2) + 1
                
                # Create the formatted relationship text
                relationship_verbs = {
                    "evidence": "serves as evidence for",
                    "background": "provides background for", 
                    "cause": "causes",
                    "contrast": "contrasts with",
                    "condition": "is a condition for"
                }
                
                verb = relationship_verbs.get(rel.relationship_type, rel.relationship_type)
                relationship_text = f"{assertion1.content} (assertion {assertion1_num}) {verb} {assertion2.content} (assertion {assertion2_num})"
                
                # Display relationship with edit and delete options
                col1, col2, col3 = st.columns([4, 1, 1])
                
                with col1:
                    st.markdown(f"**{rel.relationship_type.upper()}** - {relationship_text}")
                
                with col2:
                    if st.button("✏️", key=f"edit_rel_{i}", help="Edit this relationship"):
                        # Pre-fill the selection with this relationship's values
                        st.session_state['edit_relationship'] = {
                            'assertion1': rel.assertion1_id,
                            'assertion2': rel.assertion2_id,
                            'type': rel.relationship_type
                        }
                        st.rerun()
                
                with col3:
                    if st.button("🗑️", key=f"delete_rel_{i}", help="Delete this relationship"):
                        st.session_state.relationships.remove(rel)
                        st.rerun()
                
                st.markdown("---")
    else:
        st.info("No relationships created yet. Use the interface above to create relationships between assertions.")
    
    
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
    
    # Next button to go to Review mode
    st.markdown("---")
    if st.button("➡️ Next: Review Mode", help="Move to Review mode to check for potential issues"):
        st.session_state.current_mode = "Review"
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
    
    # Next button to go to Prose mode
    st.markdown("---")
    if st.button("➡️ Next: Prose Mode", help="Move to Prose mode to transform assertions into fluent text"):
        st.session_state.current_mode = "Prose"
        st.rerun()

def prose_tab():
    """Prose mode - Transform assertions into fluent text."""
    # Add green styling for Prose mode
    st.markdown("""
    <style>
    .prose-mode {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="prose-mode">', unsafe_allow_html=True)
    st.header("📖 Prose Mode")
    st.markdown("Transform your refined assertions into fluent, reader-friendly text.")
    
    st.info("🚧 Prose mode is coming soon! This will transform your assertions into well-structured prose.")
    st.markdown('</div>', unsafe_allow_html=True)

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
            index=["Idea Capture", "Structure", "Review", "Prose"].index(st.session_state.get("current_mode", "Idea Capture")),
            help="Click to select mode"
        )
        
        # Update session state when mode changes
        if current_mode != st.session_state.get("current_mode", "Idea Capture"):
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
    current_mode = st.session_state.get("current_mode", "Idea Capture")
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
