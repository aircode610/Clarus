"""
Streamlit UI for Review Mode

This module provides a web interface for the Review workflow,
allowing users to see document plans and highlighted issues.
"""

import streamlit as st
import json
from typing import List, Dict, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from models import Assertion, Relationship, DocumentPlan, DocumentReview, ParagraphIssue
from review import ReviewWorkflow


def main():
    st.set_page_config(
        page_title="Clarus Review Mode",
        page_icon="📝",
        layout="wide"
    )
    
    st.title("📝 Clarus Review Mode")
    st.markdown("Generate document plans and detect potential issues from assertions and relationships.")
    
    # Initialize session state
    if 'review_workflow' not in st.session_state:
        st.session_state.review_workflow = ReviewWorkflow()
    
    if 'document_review' not in st.session_state:
        st.session_state.document_review = None
    
    if 'sample_data_loaded' not in st.session_state:
        st.session_state.sample_data_loaded = False
    
    # Sidebar for input
    with st.sidebar:
        st.header("Input Data")
        
        # Load sample data button
        if st.button("Load Sample Data", type="primary"):
            load_sample_data()
            st.session_state.sample_data_loaded = True
            st.rerun()
        
        # Manual input section
        st.subheader("Manual Input")
        
        # Assertions input
        st.write("**Assertions**")
        assertions_text = st.text_area(
            "Enter assertions (one per line):",
            height=200,
            placeholder="The current UI is confusing\nUsers can't find features\nMobile support is needed"
        )
        
        # Relationships input
        st.write("**Relationships**")
        relationships_text = st.text_area(
            "Enter relationships (format: assertion1_id --[type]--> assertion2_id):",
            height=150,
            placeholder="assertion_1 --[evidence]--> assertion_2\nassertion_3 --[background]--> assertion_1"
        )
        
        # Process button
        if st.button("Generate Review", disabled=not assertions_text.strip()):
            if process_input(assertions_text, relationships_text):
                st.rerun()
    
    # Main content area
    if st.session_state.document_review:
        display_review_results()
    else:
        display_welcome()


def load_sample_data():
    """Load sample assertions and relationships for demonstration."""
    sample_assertions = [
        Assertion(
            id="assertion_1",
            content="The current user interface is confusing and difficult to navigate",
            confidence=0.9,
            source="User feedback and usability testing"
        ),
        Assertion(
            id="assertion_2", 
            content="Users are having trouble finding the main features of the application",
            confidence=0.8,
            source="Analytics data and user interviews"
        ),
        Assertion(
            id="assertion_3",
            content="We need to prioritize mobile responsiveness in our design",
            confidence=0.7,
            source="Market research and device usage statistics"
        ),
        Assertion(
            id="assertion_4",
            content="Many users have requested dark mode support",
            confidence=0.9,
            source="Feature request surveys and user feedback"
        ),
        Assertion(
            id="assertion_5",
            content="The current design lacks accessibility features",
            confidence=0.6,
            source="Accessibility audit report"
        )
    ]
    
    sample_relationships = [
        Relationship(
            assertion1_id="assertion_2",
            assertion2_id="assertion_1",
            relationship_type="evidence",
            confidence=0.8,
            explanation="User difficulty finding features provides evidence for UI confusion"
        ),
        Relationship(
            assertion1_id="assertion_3",
            assertion2_id="assertion_1",
            relationship_type="background",
            confidence=0.6,
            explanation="Mobile responsiveness provides context for UI design considerations"
        ),
        Relationship(
            assertion1_id="assertion_4",
            assertion2_id="assertion_1",
            relationship_type="evidence",
            confidence=0.7,
            explanation="User requests for dark mode support evidence of UI issues"
        ),
        Relationship(
            assertion1_id="assertion_5",
            assertion2_id="assertion_1",
            relationship_type="evidence",
            confidence=0.5,
            explanation="Accessibility issues contribute to overall UI confusion"
        )
    ]
    
    # Run the review workflow
    with st.spinner("Generating document plan and checking for issues..."):
        result = st.session_state.review_workflow.run(sample_assertions, sample_relationships)
        st.session_state.document_review = result.get('document_review')


def process_input(assertions_text: str, relationships_text: str) -> bool:
    """Process manual input and generate review."""
    try:
        # Parse assertions
        assertions = []
        for i, line in enumerate(assertions_text.strip().split('\n')):
            if line.strip():
                assertions.append(Assertion(
                    id=f"assertion_{i+1}",
                    content=line.strip(),
                    confidence=0.8,
                    source="User input"
                ))
        
        # Parse relationships (simple format)
        relationships = []
        for line in relationships_text.strip().split('\n'):
            if line.strip() and '--[' in line and ']-->' in line:
                parts = line.split('--[')
                if len(parts) == 2:
                    assertion1_id = parts[0].strip()
                    rest = parts[1].split(']-->')
                    if len(rest) == 2:
                        rel_type = rest[0].strip()
                        assertion2_id = rest[1].strip()
                        
                        relationships.append(Relationship(
                            assertion1_id=assertion1_id,
                            assertion2_id=assertion2_id,
                            relationship_type=rel_type,
                            confidence=0.7,
                            explanation=f"User-defined {rel_type} relationship"
                        ))
        
        if not assertions:
            st.error("Please provide at least one assertion.")
            return False
        
        # Run the review workflow
        with st.spinner("Generating document plan and checking for issues..."):
            result = st.session_state.review_workflow.run(assertions, relationships)
            st.session_state.document_review = result.get('document_review')
        
        return True
        
    except Exception as e:
        st.error(f"Error processing input: {str(e)}")
        return False


def display_welcome():
    """Display welcome message and instructions."""
    st.markdown("""
    ## Welcome to Clarus Review Mode! 🎯
    
    This tool helps you create structured document plans from your assertions and relationships,
    then automatically detects potential issues like:
    
    - **Missing Justification**: Claims without sufficient evidence
    - **Vague Language**: Imprecise or unclear terms
    - **Unclear Flow**: Poor logical connections between paragraphs
    - **Weak Evidence**: Insufficient support for main assertions
    - **Logical Gaps**: Missing steps in reasoning
    
    ### How to use:
    
    1. **Load Sample Data**: Click the button in the sidebar to see a demonstration
    2. **Manual Input**: Enter your own assertions and relationships in the sidebar
    3. **Review Results**: See the generated plan and highlighted issues
    
    ### Input Format:
    
    **Assertions**: One assertion per line
    ```
    The current UI is confusing
    Users can't find features
    Mobile support is needed
    ```
    
    **Relationships**: Use the format `assertion1_id --[type]--> assertion2_id`
    ```
    assertion_1 --[evidence]--> assertion_2
    assertion_3 --[background]--> assertion_1
    ```
    
    Start by clicking "Load Sample Data" to see how it works!
    """)


def display_review_results():
    """Display the review results with interactive visualizations."""
    review = st.session_state.document_review
    
    if not review:
        st.error("No review results available.")
        return
    
    # Header with overall score
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Overall Quality Score",
            value=f"{review.overall_score:.2f}",
            delta=f"{review.overall_score - 0.5:.2f}" if review.overall_score > 0.5 else None
        )
    
    with col2:
        st.metric(
            label="Total Issues",
            value=len(review.issues),
            delta=f"-{len(review.issues)}" if len(review.issues) > 0 else "0"
        )
    
    with col3:
        st.metric(
            label="Paragraphs",
            value=len(review.document_plan.paragraphs)
        )
    
    with col4:
        st.metric(
            label="Document Type",
            value=review.document_plan.document_type.title()
        )
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Document Plan", "⚠️ Issues Analysis", "📊 Visualizations", "💡 Recommendations"])
    
    with tab1:
        display_document_plan(review)
    
    with tab2:
        display_issues_analysis(review)
    
    with tab3:
        display_visualizations(review)
    
    with tab4:
        display_recommendations(review)


def display_document_plan(review: DocumentReview):
    """Display the document plan structure."""
    st.subheader("📋 Document Plan")
    
    # Document metadata
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Title:** {review.document_plan.title}")
        st.write(f"**Target Audience:** {review.document_plan.target_audience}")
    
    with col2:
        st.write(f"**Document Type:** {review.document_plan.document_type}")
        st.write(f"**Overall Flow:** {review.document_plan.overall_flow}")
    
    st.divider()
    
    # Paragraphs
    for i, para in enumerate(review.document_plan.paragraphs, 1):
        with st.expander(f"Paragraph {i}: {para.topic}", expanded=True):
            # Find main assertion
            main_assertion = next(
                (a for a in st.session_state.document_review.document_plan.paragraphs[0].__dict__.get('_assertions', []) 
                 if hasattr(a, 'id') and a.id == para.main_assertion_id), 
                None
            )
            
            if main_assertion:
                st.write(f"**Main Assertion:** {main_assertion.content}")
            else:
                st.write(f"**Main Assertion ID:** {para.main_assertion_id}")
            
            st.write(f"**Topic:** {para.topic}")
            
            if para.supporting_assertions:
                st.write("**Supporting Assertions:**")
                for supp in para.supporting_assertions:
                    st.write(f"- {supp.role.title()}: {supp.explanation}")
            
            if para.transition_notes:
                st.write(f"**Transition:** {para.transition_notes}")
            
            # Show issues for this paragraph
            para_issues = [issue for issue in review.issues if issue.paragraph_id == para.paragraph_id]
            if para_issues:
                st.warning(f"⚠️ {len(para_issues)} issue(s) found in this paragraph")


def display_issues_analysis(review: DocumentReview):
    """Display detailed issues analysis."""
    st.subheader("⚠️ Issues Analysis")
    
    if not review.issues:
        st.success("🎉 No issues detected! Your document plan looks great!")
        return
    
    # Group issues by type
    issues_by_type = {}
    for issue in review.issues:
        issue_type = issue.issue_type.issue_type
        if issue_type not in issues_by_type:
            issues_by_type[issue_type] = []
        issues_by_type[issue_type].append(issue)
    
    # Display issues by type
    for issue_type, issues in issues_by_type.items():
        with st.expander(f"{issue_type.replace('_', ' ').title()} ({len(issues)} issues)", expanded=True):
            for issue in issues:
                # Color code by severity
                if issue.issue_type.severity == "high":
                    st.error(f"🔴 **High Severity** - {issue.issue_type.description}")
                elif issue.issue_type.severity == "medium":
                    st.warning(f"🟡 **Medium Severity** - {issue.issue_type.description}")
                else:
                    st.info(f"🟢 **Low Severity** - {issue.issue_type.description}")
                
                st.write(f"**Suggestion:** {issue.issue_type.suggestion}")
                st.write(f"**Location:** {issue.location}")
                st.write(f"**Confidence:** {issue.confidence:.2f}")
                
                if issue.affected_assertions:
                    st.write(f"**Affected Assertions:** {', '.join(issue.affected_assertions)}")
                
                st.divider()


def display_visualizations(review: DocumentReview):
    """Display visualizations of the review results."""
    st.subheader("📊 Visualizations")
    
    # Issues by severity
    if review.issues:
        severity_counts = {}
        for issue in review.issues:
            severity = issue.issue_type.severity
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Severity pie chart
            fig_severity = px.pie(
                values=list(severity_counts.values()),
                names=list(severity_counts.keys()),
                title="Issues by Severity",
                color_discrete_map={
                    "high": "#ff4444",
                    "medium": "#ffaa00", 
                    "low": "#44ff44"
                }
            )
            st.plotly_chart(fig_severity, use_container_width=True)
        
        with col2:
            # Issues by type
            type_counts = {}
            for issue in review.issues:
                issue_type = issue.issue_type.issue_type
                type_counts[issue_type] = type_counts.get(issue_type, 0) + 1
            
            fig_type = px.bar(
                x=list(type_counts.keys()),
                y=list(type_counts.values()),
                title="Issues by Type",
                labels={'x': 'Issue Type', 'y': 'Count'}
            )
            fig_type.update_xaxis(tickangle=45)
            st.plotly_chart(fig_type, use_container_width=True)
    
    # Quality score gauge
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = review.overall_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Overall Quality Score"},
        delta = {'reference': 0.5},
        gauge = {
            'axis': {'range': [None, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgray"},
                {'range': [0.3, 0.7], 'color': "yellow"},
                {'range': [0.7, 1], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.8
            }
        }
    ))
    
    fig_gauge.update_layout(height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)


def display_recommendations(review: DocumentReview):
    """Display recommendations for improvement."""
    st.subheader("💡 Recommendations")
    
    if not review.recommendations:
        st.info("No specific recommendations available.")
        return
    
    for i, rec in enumerate(review.recommendations, 1):
        st.write(f"{i}. {rec}")
    
    st.divider()
    
    # Summary
    st.subheader("📝 Review Summary")
    st.write(review.summary)


if __name__ == "__main__":
    main()
