"""
Prose Workflow - Transform structured paragraphs into fluent text

This module implements the final step of the Clarus workflow, converting
structured paragraphs and accepted issue suggestions into coherent prose.
"""

from typing import List, Dict, Any, Literal
from pydantic import Field
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

from models import Assertion, Relationship, Paragraph, Issue, ProseState


class ProseWorkflow:
    """Workflow for generating final prose from structured paragraphs."""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    def _build_graph(self) -> StateGraph:
        """Build the prose generation graph."""
        builder = StateGraph(ProseState)
        
        # Add nodes
        builder.add_node("prepare_input", self._prepare_input_node)
        builder.add_node("generate_prose", self._generate_prose_node)
        builder.add_node("finalize_text", self._finalize_text_node)
        
        # Add edges
        builder.add_edge(START, "prepare_input")
        builder.add_edge("prepare_input", "generate_prose")
        builder.add_edge("generate_prose", "finalize_text")
        builder.add_edge("finalize_text", END)
        
        return builder.compile()
    
    def _prepare_input_node(self, state: ProseState) -> ProseState:
        """Prepare the input for prose generation."""
        # Create a comprehensive input that includes:
        # 1. All paragraphs with their content
        # 2. Accepted issue suggestions
        # 3. Declined issue suggestions (for context)
        # 4. Original assertions and relationships for reference
        
        paragraphs_text = "STRUCTURED PARAGRAPHS:\n"
        paragraphs_text += "=" * 50 + "\n\n"
        
        for i, paragraph in enumerate(state.ordered_paragraphs, 1):
            paragraphs_text += f"Paragraph {i} ({paragraph.paragraph_type}):\n"
            paragraphs_text += f"Title: {paragraph.title}\n"
            paragraphs_text += f"Content: {paragraph.content}\n"
            paragraphs_text += f"Confidence: {paragraph.confidence:.2f}\n"
            paragraphs_text += f"Order: {paragraph.order}\n\n"
        
        # Add accepted issue suggestions
        accepted_issues_text = "ACCEPTED ISSUE SUGGESTIONS:\n"
        accepted_issues_text += "=" * 50 + "\n\n"
        
        accepted_issues = [issue for issue in state.all_issues if issue.id in state.accepted_issues]
        for issue in accepted_issues:
            accepted_issues_text += f"Issue: {issue.issue_type.replace('_', ' ').title()}\n"
            accepted_issues_text += f"Severity: {issue.severity}\n"
            accepted_issues_text += f"Description: {issue.description}\n"
            accepted_issues_text += f"Suggestion: {issue.suggestion}\n\n"
        
        # Add declined issues for context
        declined_issues_text = "DECLINED ISSUE SUGGESTIONS (for context):\n"
        declined_issues_text += "=" * 50 + "\n\n"
        
        declined_issues = [issue for issue in state.all_issues if issue.id in state.declined_issues]
        for issue in declined_issues:
            declined_issues_text += f"Issue: {issue.issue_type.replace('_', ' ').title()}\n"
            declined_issues_text += f"Description: {issue.description}\n"
            declined_issues_text += f"Reason for declining: {issue.reason}\n\n"
        
        # Create the comprehensive input
        comprehensive_input = f"{paragraphs_text}\n{accepted_issues_text}\n{declined_issues_text}"
        
        # Update state
        state.current_input = comprehensive_input
        
        return state
    
    def _generate_prose_node(self, state: ProseState) -> ProseState:
        """Generate the final prose text."""
        # Set LLM temperature based on user preference
        self.llm.temperature = state.temperature
        
        # Create style-specific instructions
        if state.style == "Academic":
            instruction = (
                "You are an expert academic writer. Transform the provided structured paragraphs into a coherent, formal academic text suitable for a paper or report. "
                "Requirements: 1) Maintain logical flow and argumentative coherence; 2) Use precise, objective, and formal tone; 3) Add connective tissue (transitions, definitions, brief context) as needed; "
                "4) Avoid bullet points; write continuous prose; 5) Do not invent unsupported claims, but you may elaborate reasoning from the paragraphs; "
                "6) Structure content with an introduction, logically ordered paragraphs, and a concise concluding synthesis; "
                "7) Remove meta-commentary about the task itself; 8) Incorporate accepted issue suggestions naturally into the text; "
                "9) Ignore declined issue suggestions but use them for context about what to avoid."
            )
            if state.add_headings:
                instruction += " 10) Include concise section headings appropriate for academic writing."
        else:  # Technical
            instruction = (
                "You are a senior technical writer. Transform the provided structured paragraphs into clear, precise, implementation-oriented technical text suitable for engineering documentation or a design note. "
                "Requirements: 1) Prioritize clarity, unambiguous terminology, and actionability; 2) Use concise sentences and active voice; 3) Where helpful, include numbered steps, bullet lists, or short code-style blocks; "
                "4) Provide definitions and assumptions upfront; 5) Include explicit inputs, outputs, constraints, and edge cases when implied by the content; 6) Avoid marketing language and rhetorical flourish; "
                "7) Keep sections well-scoped and skimmable; 8) Incorporate accepted issue suggestions naturally into the text; "
                "9) Ignore declined issue suggestions but use them for context about what to avoid."
            )
            if state.add_headings:
                instruction += " 10) Include concise section headings appropriate for technical documentation."
        
        # Create the prompt
        prompt = (
            f"{instruction}\n\n"
            "STRUCTURED CONTENT TO TRANSFORM:\n"
            "-------------------------------\n"
            f"{state.current_input}\n\n"
            "Please generate the final prose text:"
        )
        
        # Generate the text
        message = HumanMessage(content=prompt)
        response = self.llm.invoke([message])
        generated_text = getattr(response, "content", str(response))
        
        # Update state
        state.generated_text = generated_text
        
        return state
    
    def _finalize_text_node(self, state: ProseState) -> ProseState:
        """Finalize the generated text."""
        # Add a summary message
        summary_message = f"I've successfully generated {state.style.lower()} prose from your structured paragraphs. "
        summary_message += f"The text incorporates {len(state.accepted_issues)} accepted issue suggestions and "
        summary_message += f"ignores {len(state.declined_issues)} declined suggestions."
        
        state.chat_summary = summary_message
        
        return state
    
    def run(self, 
            paragraphs: List[Paragraph],
            all_issues: List[Issue],
            accepted_issues: List[str],
            declined_issues: List[str],
            style: Literal["Academic", "Technical"] = "Academic",
            temperature: float = 0.3,
            add_headings: bool = False) -> Dict[str, Any]:
        """Run the prose generation workflow."""
        
        # Create initial state
        initial_state = ProseState(
            assertions=[],  # Not needed for prose generation
            relationships=[],  # Not needed for prose generation
            ordered_assertion_ids=[],  # Not needed for prose generation
            extracted_paragraphs=paragraphs,
            ordered_paragraphs=paragraphs,  # Use the same paragraphs for now
            all_issues=all_issues,
            accepted_issues=accepted_issues,
            declined_issues=declined_issues,
            style=style,
            temperature=temperature,
            add_headings=add_headings,
            current_input="",
            chat_summary=""
        )
        
        # Run the workflow
        graph = self._build_graph()
        final_state = graph.invoke(initial_state)
        
        # Handle both state object and dictionary returns
        if hasattr(final_state, 'generated_text'):
            # It's a state object
            return {
                "generated_text": final_state.generated_text,
                "chat_summary": final_state.chat_summary,
                "style": final_state.style,
                "temperature": final_state.temperature,
                "add_headings": final_state.add_headings
            }
        else:
            # It's already a dictionary
            return final_state


def create_prose_workflow() -> ProseWorkflow:
    """Create a prose workflow instance."""
    return ProseWorkflow()
