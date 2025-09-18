"""
Review Mode Workflow using LangGraph

This module implements the third mode of the Clarus project - Review.
Takes ordered assertions and relationships from Structure mode and creates
structured paragraphs, then orders them for optimal document flow.
"""

from typing import List, Dict, Any, Literal, Annotated, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json

from models import Assertion, Relationship, ReviewState, Paragraph, Issue


class ReviewWorkflow:
    """Main workflow class for Review mode."""
    
    def __init__(self, model_name: str = "gpt-4o-mini", config: dict = None):
        # Handle both direct instantiation and LangGraph Studio config
        if config is not None and isinstance(config, dict):
            # Extract model name from config if available
            model_name = config.get("model_name", "gpt-4o-mini")
        
        self.llm = ChatOpenAI(model=model_name, temperature=0.3)
        self.memory = MemorySaver()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow for review mode."""
        builder = StateGraph(ReviewState)
        
        # Add nodes
        builder.add_node("extract_paragraphs", self._extract_paragraphs_node)
        builder.add_node("order_paragraphs", self._order_paragraphs_node)
        builder.add_node("check_justification", self._check_justification_node)
        builder.add_node("check_vagueness", self._check_vagueness_node)
        builder.add_node("check_flow", self._check_flow_node)
        builder.add_node("merge_issues", self._merge_issues_node)
        builder.add_node("present_review", self._present_review_node)
        builder.add_node("summarize_review", self._summarize_review_node)
        
        # Add edges
        builder.add_edge(START, "extract_paragraphs")
        builder.add_edge("extract_paragraphs", "order_paragraphs")
        builder.add_edge("order_paragraphs", "check_justification")
        builder.add_edge("order_paragraphs", "check_vagueness")
        builder.add_edge("order_paragraphs", "check_flow")
        builder.add_edge("check_justification", "merge_issues")
        builder.add_edge("check_vagueness", "merge_issues")
        builder.add_edge("check_flow", "merge_issues")
        builder.add_edge("merge_issues", "present_review")
        builder.add_edge("present_review", "summarize_review")
        builder.add_edge("summarize_review", END)
        
        return builder.compile(checkpointer=self.memory)
    
    def _extract_paragraphs_node(self, state: ReviewState) -> Dict[str, Any]:
        """Extract structured paragraphs from assertions and relationships."""
        if not state.assertions or not state.ordered_assertion_ids:
            return {
                "messages": state.messages + [
                    AIMessage(content="No assertions or ordering information available for paragraph extraction.")
                ]
            }
        
        # Create assertions lookup
        assertions_dict = {a.id: a for a in state.assertions}
        
        # Create relationships lookup for context
        relationships_text = ""
        if state.relationships:
            relationship_lines = []
            for rel in state.relationships:
                assertion1 = assertions_dict.get(rel.assertion1_id)
                assertion2 = assertions_dict.get(rel.assertion2_id)
                content1 = assertion1.content if assertion1 else "Unknown"
                content2 = assertion2.content if assertion2 else "Unknown"
                relationship_lines.append(f"- {rel.relationship_type}: {content1} → {content2}")
            relationships_text = "\n".join(relationship_lines)
        
        # Format ordered assertions for the prompt
        ordered_assertions_text = ""
        for i, assertion_id in enumerate(state.ordered_assertion_ids):
            assertion = assertions_dict.get(assertion_id)
            if assertion:
                ordered_assertions_text += f"{i+1}. [{assertion_id}] {assertion.content}\n"
        
        # Create prompt for paragraph extraction
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at creating well-structured paragraphs from discrete assertions.

Your task is to:
1. Group related assertions into coherent paragraphs
2. Create smooth, flowing paragraph text that connects the assertions naturally
3. Give each paragraph a clear title/topic
4. Ensure logical flow within each paragraph
5. Consider the ordered structure provided

CRITICAL RULES:
- Each paragraph should contain 2-4 related assertions
- Create natural transitions between assertions within paragraphs
- Use the provided ordering as a guide for grouping
- Don't just concatenate assertions - create flowing prose
- Each paragraph should have a clear focus/topic
- Consider relationships between assertions when grouping

PARAGRAPH TYPES:
- introduction: Sets up the topic, provides context
- body: Main content paragraphs with supporting details
- conclusion: Summarizes key points, draws conclusions
- transition: Bridges between major sections

IMPORTANT: Return ONLY a valid JSON array. Do not include any other text.

CRITICAL: You MUST use the exact assertion IDs provided in the ORDERED ASSERTIONS section. Do not create new IDs or modify existing ones.

Return your response as a JSON list of paragraphs with this structure:
[
    {{
        "id": "paragraph_1",
        "title": "Clear paragraph title",
        "content": "Flowing paragraph text that naturally incorporates the assertions",
        "assertion_ids": ["1", "2", "3"],
        "paragraph_type": "body",
        "confidence": 0.8
    }}
]

The assertion_ids must be the exact IDs from the ORDERED ASSERTIONS list (like "1", "2", "3", etc.). The content should be well-written prose, not just a list of assertions."""),
            ("human", """Create structured paragraphs from these ordered assertions:

ORDERED ASSERTIONS:
{ordered_assertions}

RELATIONSHIPS (for context):
{relationships}

Create 3-6 well-structured paragraphs that group related assertions and create flowing prose.""")
        ])
        
        # Run LLM
        chain = prompt | self.llm
        response = chain.invoke({
            "ordered_assertions": ordered_assertions_text,
            "relationships": relationships_text
        })
        
        try:
            # Clean the response content
            content = response.content.strip()
            
            # Try to extract JSON from the response if it's wrapped in markdown
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            
            # Parse JSON response
            paragraphs_data = json.loads(content)
            
            # Ensure it's a list
            if not isinstance(paragraphs_data, list):
                paragraphs_data = [paragraphs_data]
            
            # Create paragraph objects
            paragraphs = []
            for i, paragraph_data in enumerate(paragraphs_data):
                if not isinstance(paragraph_data, dict):
                    continue
                
                # Ensure required fields exist
                paragraph_data.setdefault("id", f"paragraph_{i+1}")
                paragraph_data.setdefault("title", f"Paragraph {i+1}")
                paragraph_data.setdefault("content", "No content provided")
                paragraph_data.setdefault("assertion_ids", [])
                paragraph_data.setdefault("paragraph_type", "body")
                paragraph_data.setdefault("confidence", 0.8)
                paragraph_data.setdefault("order", i + 1)
                
                # Validate assertion IDs exist
                valid_assertion_ids = []
                for assertion_id in paragraph_data["assertion_ids"]:
                    if assertion_id in assertions_dict:
                        valid_assertion_ids.append(assertion_id)
                
                paragraph_data["assertion_ids"] = valid_assertion_ids
                
                paragraphs.append(Paragraph(**paragraph_data))
            
            # Create summary message
            paragraph_summary = f"I've extracted {len(paragraphs)} structured paragraphs from your assertions:\n\n"
            for i, paragraph in enumerate(paragraphs, 1):
                paragraph_summary += f"{i}. **{paragraph.title}** ({paragraph.paragraph_type})\n"
                paragraph_summary += f"   Contains {len(paragraph.assertion_ids)} assertions\n"
                paragraph_summary += f"   Content: {paragraph.content[:100]}{'...' if len(paragraph.content) > 100 else ''}\n\n"
            
            return {
                "extracted_paragraphs": paragraphs,
                "messages": state.messages + [
                    AIMessage(content=paragraph_summary)
                ]
            }
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Return error message with the actual response for debugging
            return {
                "messages": state.messages + [
                    AIMessage(content=f"I had trouble parsing the paragraphs. Error: {str(e)}. Response was: {response.content[:200]}...")
                ]
            }
    
    def _order_paragraphs_node(self, state: ReviewState) -> Dict[str, Any]:
        """Order paragraphs for optimal document flow."""
        if not state.extracted_paragraphs:
            return {
                "messages": state.messages + [
                    AIMessage(content="No paragraphs to order.")
                ]
            }
        
        # Create prompt for paragraph ordering
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at organizing paragraphs for optimal document flow and readability.

Your task is to:
1. Analyze the extracted paragraphs and their content
2. Determine the best order for logical flow
3. Consider the original assertion ordering as a guide
4. Ensure smooth transitions between paragraphs
5. Create a coherent narrative structure

ORDERING PRINCIPLES:
- Introduction paragraphs should come first
- Background/context paragraphs should come early
- Main argument/body paragraphs should be in logical sequence
- Supporting evidence should follow the claims they support
- Conclusion paragraphs should come last
- Transition paragraphs should bridge major sections

PARAGRAPH TYPES AND TYPICAL ORDER:
1. Introduction (sets up the topic)
2. Background/Context (provides necessary information)
3. Main Body paragraphs (core arguments, evidence)
4. Supporting paragraphs (examples, details)
5. Transition paragraphs (between major sections)
6. Conclusion (summarizes and concludes)

IMPORTANT: Return ONLY a valid JSON array. Do not include any other text.

Return your response as a JSON list with the paragraph IDs in the optimal order:
[
    "paragraph_1",
    "paragraph_3", 
    "paragraph_2",
    "paragraph_4"
]

The order should create the best logical flow for the document."""),
            ("human", """Order these paragraphs for optimal document flow:

PARAGRAPHS:
{paragraphs_text}

ORIGINAL ASSERTION ORDER (for reference):
{original_order}

Determine the best order for these paragraphs to create a coherent, well-flowing document.""")
        ])
        
        # Format paragraphs for the prompt
        paragraphs_text = ""
        for paragraph in state.extracted_paragraphs:
            paragraphs_text += f"ID: {paragraph.id}\n"
            paragraphs_text += f"Title: {paragraph.title}\n"
            paragraphs_text += f"Type: {paragraph.paragraph_type}\n"
            paragraphs_text += f"Content: {paragraph.content}\n"
            paragraphs_text += f"Assertions: {', '.join(paragraph.assertion_ids)}\n\n"
        
        # Format original order for reference
        original_order = " → ".join(state.ordered_assertion_ids)
        
        # Run LLM
        chain = prompt | self.llm
        response = chain.invoke({
            "paragraphs_text": paragraphs_text,
            "original_order": original_order
        })
        
        try:
            # Clean the response content
            content = response.content.strip()
            
            # Try to extract JSON from the response if it's wrapped in markdown
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            
            # Parse JSON response
            ordered_paragraph_ids = json.loads(content)
            
            # Ensure it's a list
            if not isinstance(ordered_paragraph_ids, list):
                ordered_paragraph_ids = [ordered_paragraph_ids]
            
            # Create paragraph lookup
            paragraphs_dict = {p.id: p for p in state.extracted_paragraphs}
            
            # Create ordered paragraphs list
            ordered_paragraphs = []
            for i, paragraph_id in enumerate(ordered_paragraph_ids):
                if paragraph_id in paragraphs_dict:
                    paragraph = paragraphs_dict[paragraph_id]
                    # Update the order field
                    paragraph.order = i + 1
                    ordered_paragraphs.append(paragraph)
            
            # Add any paragraphs not in the ordered list (fallback)
            for paragraph in state.extracted_paragraphs:
                if paragraph.id not in ordered_paragraph_ids:
                    paragraph.order = len(ordered_paragraphs) + 1
                    ordered_paragraphs.append(paragraph)
            
            # Create ordering summary
            ordering_summary = f"I've ordered the {len(ordered_paragraphs)} paragraphs for optimal flow:\n\n"
            for i, paragraph in enumerate(ordered_paragraphs, 1):
                ordering_summary += f"{i}. **{paragraph.title}** ({paragraph.paragraph_type})\n"
            
            return {
                "ordered_paragraphs": ordered_paragraphs,
                "messages": state.messages + [
                    AIMessage(content=ordering_summary)
                ]
            }
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Fallback: use original order
            ordered_paragraphs = state.extracted_paragraphs.copy()
            for i, paragraph in enumerate(ordered_paragraphs):
                paragraph.order = i + 1
            
            return {
                "ordered_paragraphs": ordered_paragraphs,
                "messages": state.messages + [
                    AIMessage(content=f"I had trouble parsing the paragraph order. Using original order. Error: {str(e)}")
                ]
            }
    
    def _check_justification_node(self, state: ReviewState) -> Dict[str, Any]:
        """Check paragraphs for missing justification issues."""
        if not state.ordered_paragraphs:
            return {"justification_issues": []}
        
        # Create prompt for justification checking
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at analyzing writing for missing justification and support.

Your task is to identify paragraphs that lack proper justification, evidence, or support for their claims.

JUSTIFICATION ISSUES TO LOOK FOR:
- Claims made without supporting evidence
- Statements that need examples or data
- Arguments without logical backing
- Assertions that require proof or validation
- General statements that need specific support

SEVERITY LEVELS:
- HIGH: Critical claims with no support that undermine credibility
- MEDIUM: Important points that would benefit from evidence
- LOW: Minor claims that could use additional support

IMPORTANT: Return ONLY a valid JSON array. Do not include any other text.

Return your response as a JSON list of issues with this structure:
[
    {{
        "id": "justification_issue_1",
        "paragraph_id": "paragraph_1",
        "issue_type": "missing_justification",
        "severity": "high|medium|low",
        "description": "Brief description of the missing justification",
        "reason": "Explanation of why this is a problem",
        "suggestion": "Specific suggestion for how to add justification",
        "confidence": 0.8
    }}
]

If no justification issues are found, return an empty array: []"""),
            ("human", """Analyze these paragraphs for missing justification issues:

PARAGRAPHS:
{paragraphs_text}

Look for claims, statements, or arguments that lack proper support, evidence, or justification.""")
        ])
        
        # Format paragraphs for the prompt
        paragraphs_text = ""
        for paragraph in state.ordered_paragraphs:
            paragraphs_text += f"ID: {paragraph.id}\n"
            paragraphs_text += f"Title: {paragraph.title}\n"
            paragraphs_text += f"Type: {paragraph.paragraph_type}\n"
            paragraphs_text += f"Content: {paragraph.content}\n\n"
        
        # Run LLM
        chain = prompt | self.llm
        response = chain.invoke({"paragraphs_text": paragraphs_text})
        
        try:
            # Clean and parse response
            content = response.content.strip()
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            
            issues_data = json.loads(content)
            if not isinstance(issues_data, list):
                issues_data = [issues_data]
            
            # Create issue objects
            issues = []
            for i, issue_data in enumerate(issues_data):
                if not isinstance(issue_data, dict):
                    continue
                
                issue_data.setdefault("id", f"justification_issue_{i+1}")
                issue_data.setdefault("issue_type", "missing_justification")
                issue_data.setdefault("confidence", 0.8)
                
                issues.append(Issue(**issue_data))
            
            return {"justification_issues": issues}
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            return {"justification_issues": []}
    
    def _check_vagueness_node(self, state: ReviewState) -> Dict[str, Any]:
        """Check paragraphs for vague language issues."""
        if not state.ordered_paragraphs:
            return {"vagueness_issues": []}
        
        # Create prompt for vagueness checking
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at identifying vague, imprecise, or unclear language in writing.

Your task is to identify paragraphs that contain vague language that could be made more specific and clear.

VAGUENESS ISSUES TO LOOK FOR:
- Unclear pronouns or references
- Vague quantifiers (many, some, few, most)
- Imprecise adjectives (good, bad, large, small)
- Ambiguous terms without definition
- Unclear relationships between ideas
- Missing specific details or examples

SEVERITY LEVELS:
- HIGH: Vague language that significantly impairs understanding
- MEDIUM: Unclear language that could confuse readers
- LOW: Minor vagueness that could be improved

IMPORTANT: Return ONLY a valid JSON array. Do not include any other text.

Return your response as a JSON list of issues with this structure:
[
    {{
        "id": "vagueness_issue_1",
        "paragraph_id": "paragraph_1",
        "issue_type": "vague_language",
        "severity": "high|medium|low",
        "description": "Brief description of the vague language",
        "reason": "Explanation of why this language is unclear",
        "suggestion": "Specific suggestion for making it clearer",
        "confidence": 0.8
    }}
]

If no vagueness issues are found, return an empty array: []"""),
            ("human", """Analyze these paragraphs for vague or unclear language:

PARAGRAPHS:
{paragraphs_text}

Look for imprecise, ambiguous, or unclear language that could be made more specific.""")
        ])
        
        # Format paragraphs for the prompt
        paragraphs_text = ""
        for paragraph in state.ordered_paragraphs:
            paragraphs_text += f"ID: {paragraph.id}\n"
            paragraphs_text += f"Title: {paragraph.title}\n"
            paragraphs_text += f"Type: {paragraph.paragraph_type}\n"
            paragraphs_text += f"Content: {paragraph.content}\n\n"
        
        # Run LLM
        chain = prompt | self.llm
        response = chain.invoke({"paragraphs_text": paragraphs_text})
        
        try:
            # Clean and parse response
            content = response.content.strip()
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            
            issues_data = json.loads(content)
            if not isinstance(issues_data, list):
                issues_data = [issues_data]
            
            # Create issue objects
            issues = []
            for i, issue_data in enumerate(issues_data):
                if not isinstance(issue_data, dict):
                    continue
                
                issue_data.setdefault("id", f"vagueness_issue_{i+1}")
                issue_data.setdefault("issue_type", "vague_language")
                issue_data.setdefault("confidence", 0.8)
                
                issues.append(Issue(**issue_data))
            
            return {"vagueness_issues": issues}
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            return {"vagueness_issues": []}
    
    def _check_flow_node(self, state: ReviewState) -> Dict[str, Any]:
        """Check paragraphs for unclear logical flow issues."""
        if not state.ordered_paragraphs:
            return {"flow_issues": []}
        
        # Create prompt for flow checking
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at analyzing logical flow and coherence in writing.

Your task is to identify paragraphs with unclear logical flow, poor transitions, or incoherent structure.

FLOW ISSUES TO LOOK FOR:
- Abrupt topic changes without transitions
- Ideas that don't logically connect
- Missing logical steps in arguments
- Poor paragraph structure or organization
- Ideas presented out of logical order
- Lack of clear progression from one idea to the next

SEVERITY LEVELS:
- HIGH: Major flow problems that significantly disrupt understanding
- MEDIUM: Flow issues that make reading difficult
- LOW: Minor flow improvements that could enhance clarity

IMPORTANT: Return ONLY a valid JSON array. Do not include any other text.

Return your response as a JSON list of issues with this structure:
[
    {{
        "id": "flow_issue_1",
        "paragraph_id": "paragraph_1",
        "issue_type": "unclear_flow",
        "severity": "high|medium|low",
        "description": "Brief description of the flow problem",
        "reason": "Explanation of why the flow is unclear",
        "suggestion": "Specific suggestion for improving flow",
        "confidence": 0.8
    }}
]

If no flow issues are found, return an empty array: []"""),
            ("human", """Analyze these paragraphs for unclear logical flow:

PARAGRAPHS:
{paragraphs_text}

Look for poor transitions, logical gaps, or unclear progression between ideas.""")
        ])
        
        # Format paragraphs for the prompt
        paragraphs_text = ""
        for paragraph in state.ordered_paragraphs:
            paragraphs_text += f"ID: {paragraph.id}\n"
            paragraphs_text += f"Title: {paragraph.title}\n"
            paragraphs_text += f"Type: {paragraph.paragraph_type}\n"
            paragraphs_text += f"Content: {paragraph.content}\n\n"
        
        # Run LLM
        chain = prompt | self.llm
        response = chain.invoke({"paragraphs_text": paragraphs_text})
        
        try:
            # Clean and parse response
            content = response.content.strip()
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            
            issues_data = json.loads(content)
            if not isinstance(issues_data, list):
                issues_data = [issues_data]
            
            # Create issue objects
            issues = []
            for i, issue_data in enumerate(issues_data):
                if not isinstance(issue_data, dict):
                    continue
                
                issue_data.setdefault("id", f"flow_issue_{i+1}")
                issue_data.setdefault("issue_type", "unclear_flow")
                issue_data.setdefault("confidence", 0.8)
                
                issues.append(Issue(**issue_data))
            
            return {"flow_issues": issues}
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            return {"flow_issues": []}
    
    def _merge_issues_node(self, state: ReviewState) -> Dict[str, Any]:
        """Merge all detected issues into a single list."""
        all_issues = []
        all_issues.extend(state.justification_issues)
        all_issues.extend(state.vagueness_issues)
        all_issues.extend(state.flow_issues)
        
        # Create summary message
        issue_summary = f"Issue analysis complete! Found {len(all_issues)} total issues:\n"
        issue_summary += f"- {len(state.justification_issues)} justification issues\n"
        issue_summary += f"- {len(state.vagueness_issues)} vagueness issues\n"
        issue_summary += f"- {len(state.flow_issues)} flow issues\n"
        
        # Count by severity
        severity_counts = {"high": 0, "medium": 0, "low": 0}
        for issue in all_issues:
            severity_counts[issue.severity] += 1
        
        issue_summary += f"\nSeverity breakdown:\n"
        issue_summary += f"- {severity_counts['high']} high priority issues\n"
        issue_summary += f"- {severity_counts['medium']} medium priority issues\n"
        issue_summary += f"- {severity_counts['low']} low priority issues\n"
        
        return {
            "all_issues": all_issues,
            "messages": state.messages + [
                AIMessage(content=issue_summary)
            ]
        }
    
    def _present_review_node(self, state: ReviewState) -> Dict[str, Any]:
        """Present the final review structure to the user."""
        if not state.ordered_paragraphs:
            return {
                "messages": state.messages + [
                    AIMessage(content="No ordered paragraphs to present.")
                ]
            }
        
        # Create the document structure presentation
        document_text = "Here's your structured document with issue analysis:\n\n"
        
        for i, paragraph in enumerate(state.ordered_paragraphs, 1):
            document_text += f"## {i}. {paragraph.title}\n"
            document_text += f"*Type: {paragraph.paragraph_type}*\n\n"
            document_text += f"{paragraph.content}\n\n"
            document_text += f"*Based on assertions: {', '.join(paragraph.assertion_ids)}*\n\n"
            
            # Show issues for this paragraph
            paragraph_issues = [issue for issue in state.all_issues if issue.paragraph_id == paragraph.id]
            if paragraph_issues:
                document_text += "**Issues found:**\n"
                for issue in paragraph_issues:
                    severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}[issue.severity]
                    document_text += f"- {severity_emoji} **{issue.issue_type.replace('_', ' ').title()}** ({issue.severity}): {issue.description}\n"
                    document_text += f"  - *Reason:* {issue.reason}\n"
                    document_text += f"  - *Suggestion:* {issue.suggestion}\n"
                document_text += "\n"
            else:
                document_text += "✅ **No issues found**\n\n"
            
            document_text += "---\n\n"
        
        # Add summary statistics
        document_text += "## Document Summary\n\n"
        document_text += f"- **Total paragraphs:** {len(state.ordered_paragraphs)}\n"
        document_text += f"- **Total assertions:** {len(state.assertions)}\n"
        document_text += f"- **Relationships used:** {len(state.relationships)}\n"
        document_text += f"- **Total issues found:** {len(state.all_issues)}\n\n"
        
        # Count paragraph types
        type_counts = {}
        for paragraph in state.ordered_paragraphs:
            type_counts[paragraph.paragraph_type] = type_counts.get(paragraph.paragraph_type, 0) + 1
        
        document_text += "**Paragraph breakdown:**\n"
        for ptype, count in type_counts.items():
            document_text += f"- {count} {ptype} paragraph(s)\n"
        
        # Count issues by type and severity
        if state.all_issues:
            document_text += "\n**Issue breakdown:**\n"
            issue_type_counts = {}
            severity_counts = {"high": 0, "medium": 0, "low": 0}
            
            for issue in state.all_issues:
                issue_type_counts[issue.issue_type] = issue_type_counts.get(issue.issue_type, 0) + 1
                severity_counts[issue.severity] += 1
            
            for issue_type, count in issue_type_counts.items():
                document_text += f"- {count} {issue_type.replace('_', ' ')} issue(s)\n"
            
            document_text += "\n**Severity breakdown:**\n"
            document_text += f"- 🔴 {severity_counts['high']} high priority issues\n"
            document_text += f"- 🟡 {severity_counts['medium']} medium priority issues\n"
            document_text += f"- 🟢 {severity_counts['low']} low priority issues\n"
        
        return {
            "messages": state.messages + [
                AIMessage(content=document_text)
            ]
        }
    
    def _summarize_review_node(self, state: ReviewState) -> Dict[str, Any]:
        """Summarize the review process."""
        if not state.ordered_paragraphs:
            summary = "No paragraphs were created during the review process."
        else:
            summary_parts = [f"Review process complete! Created {len(state.ordered_paragraphs)} structured paragraphs:"]
            
            # Count paragraph types
            type_counts = {}
            for paragraph in state.ordered_paragraphs:
                type_counts[paragraph.paragraph_type] = type_counts.get(paragraph.paragraph_type, 0) + 1
            
            for ptype, count in type_counts.items():
                summary_parts.append(f"- {count} {ptype} paragraph(s)")
            
            summary_parts.append(f"\nTotal assertions organized: {len(state.assertions)}")
            summary_parts.append(f"Relationships considered: {len(state.relationships)}")
            
            # Add issue summary
            if state.all_issues:
                summary_parts.append(f"\nIssue analysis completed:")
                summary_parts.append(f"- Total issues found: {len(state.all_issues)}")
                
                # Count by severity
                severity_counts = {"high": 0, "medium": 0, "low": 0}
                for issue in state.all_issues:
                    severity_counts[issue.severity] += 1
                
                summary_parts.append(f"- High priority: {severity_counts['high']}")
                summary_parts.append(f"- Medium priority: {severity_counts['medium']}")
                summary_parts.append(f"- Low priority: {severity_counts['low']}")
            else:
                summary_parts.append(f"\n✅ No issues found in the paragraphs!")
            
            summary = "\n".join(summary_parts)
        
        final_message = f"Review Mode Complete!\n\n{summary}\n\nThe structured paragraphs with issue analysis are now ready for the final Prose mode to create fluent, reader-friendly text."
        
        return {
            "chat_summary": summary,
            "messages": state.messages + [
                AIMessage(content=final_message)
            ]
        }
    
    def run(self, assertions: List[Assertion], relationships: List[Relationship], 
            ordered_assertion_ids: List[str], thread_id: str = "default") -> Dict[str, Any]:
        """Run the review workflow with assertions, relationships, and ordering."""
        config = {"configurable": {"thread_id": thread_id}}
        
        initial_state = ReviewState(
            assertions=assertions,
            relationships=relationships,
            ordered_assertion_ids=ordered_assertion_ids,
            messages=[HumanMessage(content=f"Starting review process for {len(assertions)} assertions with {len(relationships)} relationships.")]
        )
        
        result = self.graph.invoke(initial_state, config)
        return result


# Factory function for LangGraph Studio
def create_review_workflow(config: dict = None):
    """Factory function to create and return the compiled graph for LangGraph Studio."""
    workflow = ReviewWorkflow(config=config)
    return workflow.graph


# Alternative: Direct graph creation function
def create_review_graph(config: dict = None):
    """Create the review graph directly for LangGraph Studio."""
    # Extract model name from config if available
    model_name = "gpt-4o-mini"
    if config and isinstance(config, dict):
        model_name = config.get("model_name", "gpt-4o-mini")
    
    # Create LLM
    llm = ChatOpenAI(model=model_name, temperature=0.3)
    
    # Create memory
    memory = MemorySaver()
    
    # Build graph
    builder = StateGraph(ReviewState)
    
    # Create workflow instance for node methods
    workflow = ReviewWorkflow(model_name=model_name)
    
    # Add nodes
    builder.add_node("extract_paragraphs", workflow._extract_paragraphs_node)
    builder.add_node("order_paragraphs", workflow._order_paragraphs_node)
    builder.add_node("check_justification", workflow._check_justification_node)
    builder.add_node("check_vagueness", workflow._check_vagueness_node)
    builder.add_node("check_flow", workflow._check_flow_node)
    builder.add_node("merge_issues", workflow._merge_issues_node)
    builder.add_node("present_review", workflow._present_review_node)
    builder.add_node("summarize_review", workflow._summarize_review_node)
    
    # Add edges
    builder.add_edge(START, "extract_paragraphs")
    builder.add_edge("extract_paragraphs", "order_paragraphs")
    builder.add_edge("order_paragraphs", "check_justification")
    builder.add_edge("order_paragraphs", "check_vagueness")
    builder.add_edge("order_paragraphs", "check_flow")
    builder.add_edge("check_justification", "merge_issues")
    builder.add_edge("check_vagueness", "merge_issues")
    builder.add_edge("check_flow", "merge_issues")
    builder.add_edge("merge_issues", "present_review")
    builder.add_edge("present_review", "summarize_review")
    builder.add_edge("summarize_review", END)
    
    return builder.compile(checkpointer=memory)


# Example usage
if __name__ == "__main__":
    # Initialize the workflow
    workflow = ReviewWorkflow()
    
    # Example data (normally these would come from Structure mode)
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
        )
    ]
    
    sample_relationships = [
        Relationship(
            assertion1_id="assertion_1",
            assertion2_id="assertion_2",
            relationship_type="evidence",
            confidence=0.8,
            explanation="The confusing UI provides evidence for users having trouble finding features"
        )
    ]
    
    sample_ordered_ids = ["assertion_1", "assertion_2", "assertion_3", "assertion_4"]
    
    # Run the workflow
    result = workflow.run(sample_assertions, sample_relationships, sample_ordered_ids)
    print("Final result:", result)
