"""
Review Mode Workflow using LangGraph

This module implements the third mode of the Clarus project - Review.
Takes assertions and their relationships from Structure mode and creates a document plan,
then reviews the plan for potential issues like missing justification, vague language,
unclear logical flow, weak evidence, and logical gaps.
"""

from typing import List, Dict, Any, Literal, Annotated, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json
import uuid
from datetime import datetime

from models import (
    Assertion, Relationship, DocumentPlan, DocumentReview, ParagraphIssue,
    SupportingAssertion, Paragraph, IssueType, ReviewState
)


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
        builder.add_node("generate_plan", self._generate_plan_node)
        builder.add_node("check_ordering", self._check_ordering_node)
        builder.add_node("refine_plan", self._refine_plan_node)
        builder.add_node("start_issue_detection", self._start_issue_detection_node)
        builder.add_node("check_missing_justification", self._check_missing_justification_node)
        builder.add_node("check_vague_language", self._check_vague_language_node)
        builder.add_node("check_unclear_flow", self._check_unclear_flow_node)
        builder.add_node("check_weak_evidence", self._check_weak_evidence_node)
        builder.add_node("check_logical_gaps", self._check_logical_gaps_node)
        builder.add_node("merge_issues", self._merge_issues_node)
        builder.add_node("present_review", self._present_review_node)
        builder.add_node("summarize_review", self._summarize_review_node)
        
        # Add edges - plan generation first
        builder.add_edge(START, "generate_plan")
        builder.add_edge("generate_plan", "check_ordering")
        
        # Conditional edge for plan refinement - only goes to refinement or issue detection
        builder.add_conditional_edges(
            "check_ordering",
            self._should_refine_plan,
            {
                "refine": "refine_plan",
                "review": "start_issue_detection"
            }
        )
        builder.add_edge("refine_plan", "check_ordering")
        
        # Start issue detection phase - all issue checking nodes run in parallel
        builder.add_edge("start_issue_detection", "check_missing_justification")
        builder.add_edge("start_issue_detection", "check_vague_language")
        builder.add_edge("start_issue_detection", "check_unclear_flow")
        builder.add_edge("start_issue_detection", "check_weak_evidence")
        builder.add_edge("start_issue_detection", "check_logical_gaps")
        
        # All issue checking nodes feed into merge
        builder.add_edge("check_missing_justification", "merge_issues")
        builder.add_edge("check_vague_language", "merge_issues")
        builder.add_edge("check_unclear_flow", "merge_issues")
        builder.add_edge("check_weak_evidence", "merge_issues")
        builder.add_edge("check_logical_gaps", "merge_issues")
        
        # Merge feeds into presentation and summary
        builder.add_edge("merge_issues", "present_review")
        builder.add_edge("present_review", "summarize_review")
        builder.add_edge("summarize_review", END)
        
        return builder.compile(checkpointer=self.memory)
    
    def _generate_plan_node(self, state: ReviewState) -> Dict[str, Any]:
        """Generate a document plan from assertions and relationships."""
        if not state.assertions or not state.relationships:
            return {
                "messages": state.messages + [
                    AIMessage(content="No assertions or relationships available to create a plan.")
                ]
            }
        
        # Create assertions lookup
        assertions_dict = {a.id: a for a in state.assertions}
        
        # Format assertions and relationships for the prompt
        assertions_text = "\n".join([
            f"ID: {assertion.id}\nContent: {assertion.content}\nConfidence: {assertion.confidence:.2f}\n"
            for assertion in state.assertions
        ])
        
        relationships_text = "\n".join([
            f"{rel.assertion1_id} --[{rel.relationship_type}]--> {rel.assertion2_id} (confidence: {rel.confidence:.2f})\nExplanation: {rel.explanation}\n"
            for rel in state.relationships
        ])
        
        # Create prompt for plan generation
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at creating document plans from assertions and their relationships.

Your task is to create a coherent document plan that:
1. Groups related assertions into logical paragraphs
2. Orders paragraphs for optimal flow
3. Identifies main assertions and supporting assertions for each paragraph
4. Considers the relationships between assertions

CRITICAL RULES:
- Each paragraph should have ONE main assertion and supporting assertions
- Supporting assertions should have clear roles (evidence, background, cause, contrast, condition, example)
- Paragraphs should be ordered logically (background → main points → conclusions)
- Use the relationships to determine which assertions support which main assertions
- Consider the confidence scores when determining importance

IMPORTANT: Return ONLY a valid JSON object. Do not include any other text.

Return your plan as JSON with this structure:
{{
    "title": "Proposed document title",
    "document_type": "essay|report|article|analysis",
    "target_audience": "intended audience",
    "overall_flow": "description of document flow",
    "paragraphs": [
        {{
            "paragraph_id": "para_1",
            "main_assertion_id": "assertion_id",
            "topic": "brief topic description",
            "order": 1,
            "supporting_assertions": [
                {{
                    "assertion_id": "assertion_id",
                    "role": "evidence|background|cause|contrast|condition|example",
                    "explanation": "how this supports the main assertion"
                }}
            ],
            "transition_notes": "how this connects to next paragraph"
        }}
    ]
}}"""),
            ("human", """Create a document plan from these assertions and relationships:

ASSERTIONS:
{assertions_text}

RELATIONSHIPS:
{relationships_text}

Generate a coherent plan that groups assertions into logical paragraphs with clear main points and supporting evidence.""")
        ])
        
        # Run LLM
        chain = prompt | self.llm
        response = chain.invoke({
            "assertions_text": assertions_text,
            "relationships_text": relationships_text
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
            plan_data = json.loads(content)
            
            # Create document plan object
            paragraphs = []
            for para_data in plan_data.get("paragraphs", []):
                supporting_assertions = []
                for supp_data in para_data.get("supporting_assertions", []):
                    supporting_assertions.append(SupportingAssertion(**supp_data))
                
                paragraph = Paragraph(
                    paragraph_id=para_data.get("paragraph_id", f"para_{len(paragraphs) + 1}"),
                    main_assertion_id=para_data.get("main_assertion_id"),
                    supporting_assertions=supporting_assertions,
                    order=para_data.get("order", len(paragraphs) + 1),
                    topic=para_data.get("topic", ""),
                    transition_notes=para_data.get("transition_notes")
                )
                paragraphs.append(paragraph)
            
            document_plan = DocumentPlan(
                plan_id=f"plan_{uuid.uuid4().hex[:8]}",
                title=plan_data.get("title", "Document Plan"),
                paragraphs=paragraphs,
                overall_flow=plan_data.get("overall_flow", ""),
                target_audience=plan_data.get("target_audience", "General audience"),
                document_type=plan_data.get("document_type", "essay")
            )
            
            # Create plan summary message
            plan_summary = f"I've created a document plan with {len(paragraphs)} paragraphs:\n\n"
            for i, para in enumerate(paragraphs, 1):
                main_assertion = assertions_dict.get(para.main_assertion_id)
                if main_assertion:
                    plan_summary += f"**Paragraph {i}:** {main_assertion.content[:100]}...\n"
                    plan_summary += f"  - Topic: {para.topic}\n"
                    plan_summary += f"  - Supporting assertions: {len(para.supporting_assertions)}\n\n"
            
            return {
                "document_plan": document_plan,
                "plan_iteration": state.plan_iteration + 1,
                "messages": state.messages + [
                    AIMessage(content=plan_summary)
                ]
            }
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            return {
                "messages": state.messages + [
                    AIMessage(content=f"I had trouble creating the document plan. Error: {str(e)}. Response was: {response.content[:200]}...")
                ]
            }
    
    def _check_ordering_node(self, state: ReviewState) -> Dict[str, Any]:
        """Check if the paragraph ordering makes logical sense."""
        if not state.document_plan or not state.document_plan.paragraphs:
            return {}
        
        # Create assertions lookup
        assertions_dict = {a.id: a for a in state.assertions}
        
        # Format paragraphs for analysis
        paragraphs_text = ""
        for para in state.document_plan.paragraphs:
            main_assertion = assertions_dict.get(para.main_assertion_id)
            if main_assertion:
                paragraphs_text += f"Paragraph {para.order}: {main_assertion.content}\n"
                paragraphs_text += f"  Topic: {para.topic}\n"
                paragraphs_text += f"  Supporting: {len(para.supporting_assertions)} assertions\n\n"
        
        # Create prompt for ordering check
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at analyzing document structure and logical flow.

Your task is to evaluate whether the paragraph ordering makes logical sense for the given document type and audience.

Consider:
1. Does the flow follow logical progression (background → main points → conclusions)?
2. Are there any paragraphs that should come earlier or later?
3. Do the transitions between paragraphs make sense?
4. Is the document type appropriate for the content?

IMPORTANT: Return ONLY a valid JSON object. Do not include any other text.

Return your analysis as JSON:
{{
    "ordering_score": 0.0-1.0,
    "issues_found": [
        {{
            "paragraph_order": 1,
            "issue": "description of the ordering issue",
            "suggestion": "how to fix it"
        }}
    ],
    "needs_refinement": true/false,
    "overall_assessment": "brief assessment of the ordering"
}}"""),
            ("human", """Analyze the ordering of these paragraphs:

Document Type: {document_type}
Target Audience: {target_audience}

PARAGRAPHS:
{paragraphs_text}

Evaluate the logical flow and ordering of these paragraphs.""")
        ])
        
        # Run LLM
        chain = prompt | self.llm
        response = chain.invoke({
            "document_type": state.document_plan.document_type,
            "target_audience": state.document_plan.target_audience,
            "paragraphs_text": paragraphs_text
        })
        
        try:
            # Clean and parse response
            content = response.content.strip()
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            
            ordering_analysis = json.loads(content)
            
            # Create ordering message
            ordering_message = f"Ordering Analysis:\n"
            ordering_message += f"Score: {ordering_analysis.get('ordering_score', 0.5):.2f}/1.0\n"
            ordering_message += f"Assessment: {ordering_analysis.get('overall_assessment', 'No assessment provided')}\n"
            
            if ordering_analysis.get('issues_found'):
                ordering_message += f"\nIssues found: {len(ordering_analysis['issues_found'])}\n"
                for issue in ordering_analysis['issues_found'][:3]:  # Show first 3 issues
                    ordering_message += f"- Paragraph {issue.get('paragraph_order', '?')}: {issue.get('issue', 'Unknown issue')}\n"
            
            return {
                "messages": state.messages + [
                    AIMessage(content=ordering_message)
                ]
            }
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            return {
                "messages": state.messages + [
                    AIMessage(content=f"I had trouble analyzing the paragraph ordering. Error: {str(e)}")
                ]
            }
    
    def _should_refine_plan(self, state: ReviewState) -> Literal["refine", "review"]:
        """Determine if the plan needs refinement based on ordering analysis."""
        # For now, always proceed to review after first iteration
        # In a more sophisticated version, this could analyze the ordering score
        if state.plan_iteration < 2:  # Allow one refinement iteration
            return "review"
        return "review"
    
    def _refine_plan_node(self, state: ReviewState) -> Dict[str, Any]:
        """Refine the document plan based on ordering feedback."""
        # This would implement plan refinement logic
        # For now, just return the current plan
        return {
            "messages": state.messages + [
                AIMessage(content="Plan refinement completed. Proceeding to issue detection.")
            ]
        }
    
    def _start_issue_detection_node(self, state: ReviewState) -> Dict[str, Any]:
        """Start the issue detection phase."""
        print(f"DEBUG: Starting issue detection - Document plan has {len(state.document_plan.paragraphs) if state.document_plan else 0} paragraphs")
        return {
            "messages": state.messages + [
                AIMessage(content="Starting parallel issue detection across all paragraphs...")
            ]
        }
    
    def _check_missing_justification_node(self, state: ReviewState) -> Dict[str, Any]:
        """Check for missing justification in assertions."""
        return self._check_issue_type(state, "missing_justification")
    
    def _check_vague_language_node(self, state: ReviewState) -> Dict[str, Any]:
        """Check for vague language in assertions."""
        return self._check_issue_type(state, "vague_language")
    
    def _check_unclear_flow_node(self, state: ReviewState) -> Dict[str, Any]:
        """Check for unclear logical flow between paragraphs."""
        return self._check_issue_type(state, "unclear_flow")
    
    def _check_weak_evidence_node(self, state: ReviewState) -> Dict[str, Any]:
        """Check for weak evidence supporting main assertions."""
        return self._check_issue_type(state, "weak_evidence")
    
    def _check_logical_gaps_node(self, state: ReviewState) -> Dict[str, Any]:
        """Check for logical gaps in the argument."""
        return self._check_issue_type(state, "logical_gaps")
    
    def _check_issue_type(self, state: ReviewState, issue_type: str) -> Dict[str, Any]:
        """Generic method to check for a specific type of issue."""
        if not state.document_plan or not state.document_plan.paragraphs:
            return {f"{issue_type}_issues": []}
        
        # Debug: Check if we have a document plan
        print(f"DEBUG: Checking {issue_type} - Document plan has {len(state.document_plan.paragraphs)} paragraphs")
        
        # Create assertions lookup
        assertions_dict = {a.id: a for a in state.assertions}
        
        # Format paragraphs for analysis
        paragraphs_text = ""
        for para in state.document_plan.paragraphs:
            main_assertion = assertions_dict.get(para.main_assertion_id)
            if main_assertion:
                paragraphs_text += f"Paragraph {para.order} (ID: {para.paragraph_id}):\n"
                paragraphs_text += f"  Main: {main_assertion.content}\n"
                paragraphs_text += f"  Topic: {para.topic}\n"
                for supp in para.supporting_assertions:
                    supp_assertion = assertions_dict.get(supp.assertion_id)
                    if supp_assertion:
                        paragraphs_text += f"  Supporting ({supp.role}): {supp_assertion.content}\n"
                paragraphs_text += "\n"
        
        # Create issue-specific prompts
        issue_prompts = {
            "missing_justification": {
                "description": "Missing justification or evidence for claims",
                "criteria": "Look for assertions that make claims without sufficient evidence, data, or reasoning to support them. Be aggressive - flag any claim that lacks specific data, examples, or detailed reasoning."
            },
            "vague_language": {
                "description": "Vague or imprecise language",
                "criteria": "Look for words like 'some', 'many', 'often', 'usually', 'generally', 'probably', 'might', 'could', 'seems', 'appears' without quantification, or unclear terms. Flag any language that is not specific or measurable."
            },
            "unclear_flow": {
                "description": "Unclear logical flow between paragraphs",
                "criteria": "Look for paragraphs that don't connect well to each other or lack clear transitions. Flag any paragraph that doesn't clearly relate to the previous one or doesn't build logically on previous points."
            },
            "weak_evidence": {
                "description": "Weak or insufficient evidence",
                "criteria": "Look for supporting assertions that don't adequately support the main assertion. Flag any supporting evidence that is too weak, irrelevant, or doesn't directly support the main claim."
            },
            "logical_gaps": {
                "description": "Logical gaps or missing steps in reasoning",
                "criteria": "Look for missing logical steps, unsupported leaps in reasoning, or contradictions. Flag any argument that jumps to conclusions without proper reasoning or has internal contradictions."
            }
        }
        
        issue_info = issue_prompts.get(issue_type, {"description": "Unknown issue", "criteria": "No criteria"})
        
        # Create prompt for issue detection
        system_message = """You are an expert at detecting {issue_description} in document plans.

Your task is to identify {issue_description} in the given paragraphs.

{criteria}

Be thorough and aggressive in your analysis - flag any potential issues that could impact document quality. Don't be overly conservative - if there's any doubt, flag it as an issue. Look for subtle problems that might not be immediately obvious.

IMPORTANT: Return ONLY a valid JSON object. Do not include any other text.

Return your analysis as JSON:
{{
    "issues": [
        {{
            "paragraph_id": "para_1",
            "issue_description": "specific description of the issue",
            "severity": "low|medium|high",
            "affected_assertions": ["assertion_id1", "assertion_id2"],
            "location": "specific location in paragraph",
            "suggestion": "how to fix this issue",
            "confidence": 0.0-1.0
        }}
    ]
}}

If no issues are found, return: {{"issues": []}}"""

        human_message = """Analyze these paragraphs for {issue_description}:

{{paragraphs_text}}

Identify any {issue_description} issues in these paragraphs."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ])
        
        # Run LLM
        chain = prompt | self.llm
        response = chain.invoke({
            "paragraphs_text": paragraphs_text,
            "issue_description": issue_info['description'],
            "criteria": issue_info['criteria']
        })
        
        # Debug: Print what we're analyzing
        print(f"DEBUG: {issue_type} - Analyzing {len(paragraphs_text.split('Paragraph'))} paragraphs")
        print(f"DEBUG: {issue_type} - LLM response: {response.content[:200]}...")
        
        try:
            # Clean and parse response
            content = response.content.strip()
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            
            issue_analysis = json.loads(content)
            
            # Create issue objects
            issues = []
            for issue_data in issue_analysis.get("issues", []):
                issue_type_obj = IssueType(
                    issue_type=issue_type,
                    severity=issue_data.get("severity", "medium"),
                    description=issue_data.get("issue_description", ""),
                    suggestion=issue_data.get("suggestion", "")
                )
                
                paragraph_issue = ParagraphIssue(
                    issue_id=f"{issue_type}_{uuid.uuid4().hex[:8]}",
                    paragraph_id=issue_data.get("paragraph_id", ""),
                    issue_type=issue_type_obj,
                    affected_assertions=issue_data.get("affected_assertions", []),
                    location=issue_data.get("location", ""),
                    confidence=issue_data.get("confidence", 0.7)
                )
                issues.append(paragraph_issue)
            
            return {f"{issue_type}_issues": issues}
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            return {f"{issue_type}_issues": []}
    
    def _merge_issues_node(self, state: ReviewState) -> Dict[str, Any]:
        """Merge all issue detection results."""
        # Collect all issues from different checks
        all_issues = []
        missing_issues = getattr(state, "missing_justification_issues", [])
        vague_issues = getattr(state, "vague_language_issues", [])
        flow_issues = getattr(state, "unclear_flow_issues", [])
        evidence_issues = getattr(state, "weak_evidence_issues", [])
        gaps_issues = getattr(state, "logical_gaps_issues", [])
        
        print(f"DEBUG: Merge issues - missing: {len(missing_issues)}, vague: {len(vague_issues)}, flow: {len(flow_issues)}, evidence: {len(evidence_issues)}, gaps: {len(gaps_issues)}")
        
        all_issues.extend(missing_issues)
        all_issues.extend(vague_issues)
        all_issues.extend(flow_issues)
        all_issues.extend(evidence_issues)
        all_issues.extend(gaps_issues)
        
        # Calculate overall score based on issues
        if all_issues:
            # Calculate average confidence and adjust for severity
            total_score = 0
            for issue in all_issues:
                severity_multiplier = {"low": 0.1, "medium": 0.3, "high": 0.5}.get(issue.issue_type.severity, 0.3)
                total_score += issue.confidence * severity_multiplier
            
            overall_score = max(0, 1 - (total_score / len(all_issues)))
        else:
            overall_score = 1.0
        
        # Create document review
        document_review = DocumentReview(
            review_id=f"review_{uuid.uuid4().hex[:8]}",
            document_plan=state.document_plan,
            issues=all_issues,
            overall_score=overall_score,
            summary=f"Found {len(all_issues)} issues across {len(set(issue.paragraph_id for issue in all_issues))} paragraphs",
            recommendations=self._generate_recommendations(all_issues)
        )
        
        # Create merge message
        merge_message = f"Issue Detection Complete!\n\n"
        merge_message += f"Total issues found: {len(all_issues)}\n"
        merge_message += f"Overall quality score: {overall_score:.2f}/1.0\n\n"
        
        if all_issues:
            # Group issues by type
            issues_by_type = {}
            for issue in all_issues:
                issue_type = issue.issue_type.issue_type
                if issue_type not in issues_by_type:
                    issues_by_type[issue_type] = []
                issues_by_type[issue_type].append(issue)
            
            merge_message += "Issues by type:\n"
            for issue_type, issues in issues_by_type.items():
                merge_message += f"- {issue_type.replace('_', ' ').title()}: {len(issues)} issues\n"
        else:
            merge_message += "No significant issues detected! The document plan looks good."
        
        return {
            "document_review": document_review,
            "messages": state.messages + [AIMessage(content=merge_message)]
        }
    
    def _generate_recommendations(self, issues: List[ParagraphIssue]) -> List[str]:
        """Generate high-level recommendations based on detected issues."""
        recommendations = []
        
        # Group issues by type
        issues_by_type = {}
        for issue in issues:
            issue_type = issue.issue_type.issue_type
            if issue_type not in issues_by_type:
                issues_by_type[issue_type] = []
            issues_by_type[issue_type].append(issue)
        
        # Generate recommendations for each issue type
        if "missing_justification" in issues_by_type:
            recommendations.append("Add more evidence and justification for your main claims")
        
        if "vague_language" in issues_by_type:
            recommendations.append("Use more specific and precise language throughout the document")
        
        if "unclear_flow" in issues_by_type:
            recommendations.append("Improve transitions and logical flow between paragraphs")
        
        if "weak_evidence" in issues_by_type:
            recommendations.append("Strengthen the evidence supporting your main assertions")
        
        if "logical_gaps" in issues_by_type:
            recommendations.append("Fill in logical gaps and ensure complete reasoning chains")
        
        if not recommendations:
            recommendations.append("The document plan is well-structured with no major issues detected")
        
        return recommendations
    
    def _present_review_node(self, state: ReviewState) -> Dict[str, Any]:
        """Present the review results to the user."""
        if not state.document_review:
            return {
                "messages": state.messages + [
                    AIMessage(content="No review results to present.")
                ]
            }
        
        review = state.document_review
        
        # Create detailed presentation
        presentation = f"# Document Review Results\n\n"
        presentation += f"**Overall Quality Score:** {review.overall_score:.2f}/1.0\n\n"
        presentation += f"**Document Title:** {review.document_plan.title}\n"
        presentation += f"**Document Type:** {review.document_plan.document_type}\n"
        presentation += f"**Target Audience:** {review.document_plan.target_audience}\n\n"
        
        presentation += f"## Document Plan\n\n"
        for para in review.document_plan.paragraphs:
            presentation += f"**Paragraph {para.order}:** {para.topic}\n"
            # Find main assertion
            main_assertion = next((a for a in state.assertions if a.id == para.main_assertion_id), None)
            if main_assertion:
                presentation += f"- Main: {main_assertion.content}\n"
            presentation += f"- Supporting assertions: {len(para.supporting_assertions)}\n\n"
        
        if review.issues:
            presentation += f"## Issues Found ({len(review.issues)} total)\n\n"
            
            # Group issues by paragraph
            issues_by_paragraph = {}
            for issue in review.issues:
                if issue.paragraph_id not in issues_by_paragraph:
                    issues_by_paragraph[issue.paragraph_id] = []
                issues_by_paragraph[issue.paragraph_id].append(issue)
            
            for para_id, para_issues in issues_by_paragraph.items():
                # Find paragraph info
                para = next((p for p in review.document_plan.paragraphs if p.paragraph_id == para_id), None)
                if para:
                    presentation += f"### Paragraph {para.order}: {para.topic}\n"
                    for issue in para_issues:
                        presentation += f"- **{issue.issue_type.issue_type.replace('_', ' ').title()}** ({issue.issue_type.severity} severity)\n"
                        presentation += f"  - Issue: {issue.issue_type.description}\n"
                        presentation += f"  - Suggestion: {issue.issue_type.suggestion}\n"
                        presentation += f"  - Location: {issue.location}\n\n"
        else:
            presentation += "## Issues Found\n\nNo significant issues detected! 🎉\n\n"
        
        presentation += f"## Recommendations\n\n"
        for i, rec in enumerate(review.recommendations, 1):
            presentation += f"{i}. {rec}\n"
        
        return {
            "messages": state.messages + [AIMessage(content=presentation)]
        }
    
    def _summarize_review_node(self, state: ReviewState) -> Dict[str, Any]:
        """Summarize the review process."""
        if not state.document_review:
            return {
                "chat_summary": "Review process completed but no results available.",
                "review_complete": True
            }
        
        review = state.document_review
        
        summary = f"Review Complete!\n\n"
        summary += f"Document: {review.document_plan.title}\n"
        summary += f"Quality Score: {review.overall_score:.2f}/1.0\n"
        summary += f"Issues Found: {len(review.issues)}\n"
        summary += f"Paragraphs: {len(review.document_plan.paragraphs)}\n\n"
        summary += f"Summary: {review.summary}\n\n"
        summary += "The document plan is ready for writing or further refinement."
        
        return {
            "chat_summary": summary,
            "review_complete": True,
            "messages": state.messages + [AIMessage(content=summary)]
        }
    
    def run(self, assertions: List[Assertion], relationships: List[Relationship], thread_id: str = "default") -> Dict[str, Any]:
        """Run the review workflow with assertions and relationships."""
        config = {"configurable": {"thread_id": thread_id}}
        
        initial_state = ReviewState(
            assertions=assertions,
            relationships=relationships,
            messages=[HumanMessage(content=f"Starting review process for {len(assertions)} assertions and {len(relationships)} relationships.")]
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
    builder.add_node("generate_plan", workflow._generate_plan_node)
    builder.add_node("check_ordering", workflow._check_ordering_node)
    builder.add_node("refine_plan", workflow._refine_plan_node)
    builder.add_node("start_issue_detection", workflow._start_issue_detection_node)
    builder.add_node("check_missing_justification", workflow._check_missing_justification_node)
    builder.add_node("check_vague_language", workflow._check_vague_language_node)
    builder.add_node("check_unclear_flow", workflow._check_unclear_flow_node)
    builder.add_node("check_weak_evidence", workflow._check_weak_evidence_node)
    builder.add_node("check_logical_gaps", workflow._check_logical_gaps_node)
    builder.add_node("merge_issues", workflow._merge_issues_node)
    builder.add_node("present_review", workflow._present_review_node)
    builder.add_node("summarize_review", workflow._summarize_review_node)
    
    # Add edges
    builder.add_edge(START, "generate_plan")
    builder.add_edge("generate_plan", "check_ordering")
    
    # Conditional edge for plan refinement
    builder.add_conditional_edges(
        "check_ordering",
        workflow._should_refine_plan,
        {
            "refine": "refine_plan",
            "review": "start_issue_detection"
        }
    )
    builder.add_edge("refine_plan", "check_ordering")
    
    # Start issue detection phase - all issue checking nodes run in parallel
    builder.add_edge("start_issue_detection", "check_missing_justification")
    builder.add_edge("start_issue_detection", "check_vague_language")
    builder.add_edge("start_issue_detection", "check_unclear_flow")
    builder.add_edge("start_issue_detection", "check_weak_evidence")
    builder.add_edge("start_issue_detection", "check_logical_gaps")
    
    # All issue checking nodes feed into merge
    builder.add_edge("check_missing_justification", "merge_issues")
    builder.add_edge("check_vague_language", "merge_issues")
    builder.add_edge("check_unclear_flow", "merge_issues")
    builder.add_edge("check_weak_evidence", "merge_issues")
    builder.add_edge("check_logical_gaps", "merge_issues")
    
    # Merge feeds into presentation and summary
    builder.add_edge("merge_issues", "present_review")
    builder.add_edge("present_review", "summarize_review")
    builder.add_edge("summarize_review", END)
    
    return builder.compile(checkpointer=memory)


# Example usage
if __name__ == "__main__":
    # Initialize the workflow
    workflow = ReviewWorkflow()
    
    # Example assertions and relationships (normally these would come from previous modes)
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
        )
    ]
    
    # Run the workflow
    result = workflow.run(sample_assertions, sample_relationships)
    print("Final result:", result)
