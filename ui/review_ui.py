"""
Review UI components for the Clarus application.

This module contains the UI components for the Review mode.
"""

import streamlit as st
import json
from workflows.review import ReviewWorkflow
from .common import next_mode_button


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
    next_mode_button("Review", "Prose", "Move to Prose mode to transform paragraphs into fluent text")
