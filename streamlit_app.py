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
import json
from typing import List, Dict, Any

import voice.streamlit_voice as streamlit_voice
from workflows.conflict_resolving import GlobalGraph
from models import Assertion, Relationship
from app import ClarusApp, create_clarus_app
from workflows.structure import evaluate_relationship_quality
from workflows.review import ReviewWorkflow
from workflows.prose import ProseWorkflow

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
        from models import Relationship
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
    st.session_state.global_graph = GlobalGraph(st.session_state.relationships)
    st.session_state.conflicts_resolved = False  # Need to re-resolve conflicts
    st.session_state.chose_resolution_method = False
    st.session_state.ordered_graph_generated = False
    
    st.rerun()

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
            st.rerun()
    
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
        # Graph interaction controls (must be defined before graph creation)
        st.markdown("---")
        st.subheader("🎛️ Graph Controls")
        
        # col1, col2, col3 = st.columns(3)
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
    import networkx as nx
    import plotly.graph_objects as go
    import plotly.express as px
    
    # Create a directed graph
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
                      source=assertion.source        )
    
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
    
    # Relationship Management Section
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
    st.markdown("---")
    st.subheader("🔧 Conflict Resolution")
    st.markdown("Resolve any cycles or contradictions in your assertion relationships.")
    
    # Initialize global graph if not exists
    if not st.session_state.get("global_graph", None):
        st.session_state.global_graph = GlobalGraph(st.session_state.get("relationships", []))
    
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
    """Review mode - Create structured paragraphs from ordered assertions."""
    st.header("🔍 Review Mode")
    st.markdown("Transform your ordered assertions into structured paragraphs with optimal flow and organization.")

    # Check if conflicts have been resolved in structure mode
    if not st.session_state.get("conflicts_resolved", False):
        st.warning("⚠️ Please complete conflict resolution in Structure mode before proceeding to Review.")
        st.markdown("Go back to Structure mode to resolve any cycles or contradictions in your assertion relationships.")
        return
    
    if not st.session_state.get("ordered_graph"):
        st.warning("⚠️ No ordered assertions available. Please complete the Structure mode first.")
        return
    
    st.success("✅ Conflicts have been resolved in Structure mode.")
    
    # Show ordered graph summary
    st.markdown(f"**Ordered assertions:** {len(st.session_state.ordered_graph)} assertions")
    st.markdown(f"**Remaining relationships:** {len(st.session_state.relationships)} relationships")
    
    # Initialize review analysis if not done yet
    if "review_analysis_done" not in st.session_state:
        st.session_state.review_analysis_done = False
    
    if "paragraphs" not in st.session_state:
        st.session_state.paragraphs = []
    
    if "ordered_paragraphs" not in st.session_state:
        st.session_state.ordered_paragraphs = []
    
    if "all_issues" not in st.session_state:
        st.session_state.all_issues = []
    
    if "accepted_issues" not in st.session_state:
        st.session_state.accepted_issues = []
    
    if "declined_issues" not in st.session_state:
        st.session_state.declined_issues = []
    
    # Run review analysis if not done yet
    if not st.session_state.review_analysis_done:
        with st.spinner("Creating structured paragraphs from your assertions..."):
            try:
                # Initialize review workflow
                review_workflow = ReviewWorkflow()
                
                # Run the review workflow
                result = review_workflow.run(
                    assertions=st.session_state.assertions,
                    relationships=st.session_state.relationships,
                    ordered_assertion_ids=st.session_state.ordered_graph
                )
                
                if "extracted_paragraphs" in result:
                    st.session_state.paragraphs = result["extracted_paragraphs"]
                    st.session_state.review_analysis_done = True
                    st.success(f"✅ Paragraph extraction complete! Created {len(result['extracted_paragraphs'])} structured paragraphs.")
                else:
                    st.warning("No paragraphs were created from the assertions.")
                
                # Store issues if available
                if "all_issues" in result:
                    st.session_state.all_issues = result["all_issues"]
                    if st.session_state.all_issues:
                        st.info(f"🔍 Found {len(st.session_state.all_issues)} issues to review.")
                else:
                    st.session_state.all_issues = []
            except Exception as e:
                st.error(f"Error running review analysis: {e}")
                return
    
    # Display the ordered structure
    st.subheader("📋 Ordered Assertion Structure")
    assertions_dict = {a.id: a for a in st.session_state.assertions}
    
    with st.expander("View Ordered Assertions", expanded=False):
        for i, assertion_id in enumerate(st.session_state.ordered_graph, 1):
            assertion = assertions_dict.get(assertion_id)
            if assertion:
                st.markdown(f"{i}. **{assertion_id}**: {assertion.content}")
    
    # Display structured paragraphs
    if st.session_state.paragraphs:
        st.subheader("📝 Structured Paragraphs")
        
        # Show paragraph summary
        paragraph_types = {}
        for paragraph in st.session_state.paragraphs:
            paragraph_types[paragraph.paragraph_type] = paragraph_types.get(paragraph.paragraph_type, 0) + 1
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Paragraphs", len(st.session_state.paragraphs))
        with col2:
            st.metric("Body Paragraphs", paragraph_types.get("body", 0))
        with col3:
            st.metric("Other Types", sum(count for ptype, count in paragraph_types.items() if ptype != "body"))
        with col4:
            st.metric("Total Issues", len(st.session_state.all_issues))
        
        # Issue summary
        if st.session_state.all_issues:
            st.markdown("### 📊 Issue Summary")
            
            # Count issues by severity
            severity_counts = {"high": 0, "medium": 0, "low": 0}
            issue_type_counts = {}
            for issue in st.session_state.all_issues:
                severity_counts[issue.severity] += 1
                issue_type_counts[issue.issue_type] = issue_type_counts.get(issue.issue_type, 0) + 1
            
            # Count accepted/declined issues
            accepted_count = len(st.session_state.accepted_issues)
            declined_count = len(st.session_state.declined_issues)
            pending_count = len(st.session_state.all_issues) - accepted_count - declined_count
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**By Severity:**")
                st.write(f"🔴 High: {severity_counts['high']}")
                st.write(f"🟡 Medium: {severity_counts['medium']}")
                st.write(f"🟢 Low: {severity_counts['low']}")
            
            with col2:
                st.markdown("**By Type:**")
                for issue_type, count in issue_type_counts.items():
                    st.write(f"- {issue_type.replace('_', ' ').title()}: {count}")
            
            with col3:
                st.markdown("**By Status:**")
                st.write(f"✅ Accepted: {accepted_count}")
                st.write(f"❌ Declined: {declined_count}")
                st.write(f"⏳ Pending: {pending_count}")
            
            # Progress bar for issue resolution
            if len(st.session_state.all_issues) > 0:
                resolved_percentage = ((accepted_count + declined_count) / len(st.session_state.all_issues)) * 100
                st.progress(resolved_percentage / 100)
                st.caption(f"Issue Resolution Progress: {resolved_percentage:.1f}%")
        
        # Display each paragraph with issues
        for i, paragraph in enumerate(st.session_state.paragraphs, 1):
            # Get issues for this paragraph
            paragraph_issues = [issue for issue in st.session_state.all_issues if issue.paragraph_id == paragraph.id]
            
            # Create expander title with issue indicator
            expander_title = f"Paragraph {i}: {paragraph.title} ({paragraph.paragraph_type})"
            if paragraph_issues:
                high_issues = len([issue for issue in paragraph_issues if issue.severity == "high"])
                medium_issues = len([issue for issue in paragraph_issues if issue.severity == "medium"])
                low_issues = len([issue for issue in paragraph_issues if issue.severity == "low"])
                
                if high_issues > 0:
                    expander_title += f" 🔴 {high_issues}"
                if medium_issues > 0:
                    expander_title += f" 🟡 {medium_issues}"
                if low_issues > 0:
                    expander_title += f" 🟢 {low_issues}"
            else:
                expander_title += " ✅"
            
            with st.expander(expander_title, expanded=False):
                st.markdown(f"**Type:** {paragraph.paragraph_type}")
                st.markdown(f"**Confidence:** {paragraph.confidence:.2f}")
                st.markdown(f"**Order:** {paragraph.order}")
                
                st.markdown("**Content:**")
                st.write(paragraph.content)
                
                st.markdown("**Based on assertions:**")
                if paragraph.assertion_ids:
                    for assertion_id in paragraph.assertion_ids:
                        assertion = assertions_dict.get(assertion_id)
                        if assertion:
                            st.write(f"- {assertion.content}")
                        else:
                            st.write(f"- [Missing assertion: {assertion_id}]")
                else:
                    st.write("*No assertions linked to this paragraph*")
                
                # Display issues for this paragraph
                if paragraph_issues:
                    st.markdown("---")
                    st.markdown("### 🔍 Issues Found")
                    
                    for issue in paragraph_issues:
                        # Check if issue has been accepted or declined
                        issue_status = "pending"
                        if issue.id in st.session_state.accepted_issues:
                            issue_status = "accepted"
                        elif issue.id in st.session_state.declined_issues:
                            issue_status = "declined"
                        
                        # Color code based on severity
                        severity_colors = {
                            "high": "🔴",
                            "medium": "🟡", 
                            "low": "🟢"
                        }
                        
                        severity_emoji = severity_colors.get(issue.severity, "⚪")
                        
                        # Create issue display
                        if issue_status == "accepted":
                            st.success(f"{severity_emoji} **{issue.issue_type.replace('_', ' ').title()}** ({issue.severity}) - ✅ ACCEPTED")
                        elif issue_status == "declined":
                            st.info(f"{severity_emoji} **{issue.issue_type.replace('_', ' ').title()}** ({issue.severity}) - ❌ DECLINED")
                        else:
                            st.warning(f"{severity_emoji} **{issue.issue_type.replace('_', ' ').title()}** ({issue.severity})")
                        
                        st.write(f"**Description:** {issue.description}")
                        st.write(f"**Reason:** {issue.reason}")
                        st.write(f"**Suggestion:** {issue.suggestion}")
                        
                        # Accept/Decline buttons
                        if issue_status == "pending":
                            col_accept, col_decline = st.columns(2)
                            
                            with col_accept:
                                if st.button(f"✅ Accept Suggestion", key=f"accept_{issue.id}", help="Accept this issue and its suggestion"):
                                    if issue.id not in st.session_state.accepted_issues:
                                        st.session_state.accepted_issues.append(issue.id)
                                    st.rerun()
                            
                            with col_decline:
                                if st.button(f"❌ Decline", key=f"decline_{issue.id}", help="Decline this issue"):
                                    if issue.id not in st.session_state.declined_issues:
                                        st.session_state.declined_issues.append(issue.id)
                                    st.rerun()
                        
                        st.markdown("---")
                else:
                    st.markdown("---")
                    st.success("✅ **No issues found** - This paragraph looks good!")
    
    # Action buttons
    st.markdown("---")
    col_rerun, col_export, col_issues, col_reset = st.columns(4)
    
    with col_rerun:
        if st.button("🔄 Re-analyze Paragraphs", help="Run review analysis again"):
            st.session_state.review_analysis_done = False
            st.session_state.paragraphs = []
            st.session_state.ordered_paragraphs = []
            st.session_state.all_issues = []
            st.session_state.accepted_issues = []
            st.session_state.declined_issues = []
            st.rerun()
    
    with col_export:
        if st.button("📤 Export Paragraphs", help="Export paragraphs as JSON"):
            if st.session_state.paragraphs:
                paragraphs_data = [p.model_dump() for p in st.session_state.paragraphs]
                st.download_button(
                    label="Download Paragraphs",
                    data=json.dumps(paragraphs_data, indent=2),
                    file_name="paragraphs.json",
                    mime="application/json"
                )
            else:
                st.warning("No paragraphs to export")
    
    with col_issues:
        if st.button("📋 Export Issues", help="Export issues as JSON"):
            if st.session_state.all_issues:
                issues_data = [issue.model_dump() for issue in st.session_state.all_issues]
                # Add resolution status
                for issue_data in issues_data:
                    if issue_data["id"] in st.session_state.accepted_issues:
                        issue_data["resolution_status"] = "accepted"
                    elif issue_data["id"] in st.session_state.declined_issues:
                        issue_data["resolution_status"] = "declined"
                    else:
                        issue_data["resolution_status"] = "pending"
                
                st.download_button(
                    label="Download Issues",
                    data=json.dumps(issues_data, indent=2),
                    file_name="issues.json",
                    mime="application/json"
                )
            else:
                st.warning("No issues to export")
    
    with col_reset:
        if st.button("🗑️ Reset Review", help="Clear all paragraphs and issues"):
            st.session_state.paragraphs = []
            st.session_state.ordered_paragraphs = []
            st.session_state.all_issues = []
            st.session_state.accepted_issues = []
            st.session_state.declined_issues = []
            st.session_state.review_analysis_done = False
            st.rerun()
    
    # Next button to go to Prose mode
    st.markdown("---")
    if st.button("➡️ Next: Prose Mode", help="Move to Prose mode to transform paragraphs into fluent text"):
        st.session_state.current_mode = "Prose"
        st.rerun()

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
