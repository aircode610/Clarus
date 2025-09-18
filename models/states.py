"""
State models for LangGraph workflows.

This module defines the state classes used by the different workflow modes.
"""

from typing import List, Annotated, Optional, Literal
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from datetime import datetime

from .assertions import Assertion, Relationship


class ChangeRecord(BaseModel):
    """Record of a single change to assertions."""
    change_id: str = Field(description="Unique identifier for this change")
    change_type: Literal["add", "remove", "modify"] = Field(description="Type of change")
    timestamp: datetime = Field(default_factory=datetime.now)
    assertion: Optional[Assertion] = Field(default=None, description="The assertion that was added/modified")
    original_assertion: Optional[Assertion] = Field(default=None, description="Original assertion before modification")
    assertion_index: Optional[int] = Field(default=None, description="Index of the assertion in the list")
    user_request: str = Field(description="The user's request that caused this change")
    description: str = Field(description="Human-readable description of the change")


class ChangeHistory(BaseModel):
    """History of all changes made to assertions."""
    changes: List[ChangeRecord] = Field(default_factory=list)
    current_version: int = Field(default=0, description="Current version number")
    
    def add_change(self, change: ChangeRecord) -> None:
        """Add a new change to the history."""
        self.changes.append(change)
        self.current_version += 1
    
    def get_last_change(self) -> Optional[ChangeRecord]:
        """Get the most recent change."""
        return self.changes[-1] if self.changes else None
    
    def get_changes_by_type(self, change_type: str) -> List[ChangeRecord]:
        """Get all changes of a specific type."""
        return [change for change in self.changes if change.change_type == change_type]
    
    def get_removed_assertions(self) -> List[Assertion]:
        """Get all assertions that were removed."""
        return [change.assertion for change in self.get_changes_by_type("remove") if change.assertion]
    
    def find_assertion_by_content(self, content: str) -> Optional[Assertion]:
        """Find an assertion by its content in the history."""
        for change in reversed(self.changes):  # Search from most recent
            if change.assertion and change.assertion.content == content:
                return change.assertion
        return None


class IdeaCaptureState(BaseModel):
    """State for the Idea Capture workflow."""
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
    assertions: List[Assertion] = Field(default_factory=list)
    current_input: str = Field(default="")
    iteration_count: int = Field(default=0)
    chat_summary: str = Field(default="")
    change_history: ChangeHistory = Field(default_factory=ChangeHistory)


class StructureState(BaseModel):
    """State for the Structure workflow."""
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
    assertions: List[Assertion] = Field(default_factory=list)
    evidence_relationships: List[Relationship] = Field(default_factory=list)
    background_relationships: List[Relationship] = Field(default_factory=list)
    cause_relationships: List[Relationship] = Field(default_factory=list)
    contrast_relationships: List[Relationship] = Field(default_factory=list)
    condition_relationships: List[Relationship] = Field(default_factory=list)
    contradiction_relationships: List[Relationship] = Field(default_factory=list)
    final_relationships: List[Relationship] = Field(default_factory=list)
    evaluated_relationships: List[Relationship] = Field(default_factory=list)
    current_input: str = Field(default="")
    chat_summary: str = Field(default="")


class Paragraph(BaseModel):
    """Represents a structured paragraph containing multiple assertions."""
    id: str = Field(description="Unique identifier for the paragraph")
    title: str = Field(description="Title or topic of the paragraph")
    content: str = Field(description="The actual paragraph text")
    assertion_ids: List[str] = Field(description="IDs of assertions included in this paragraph")
    order: int = Field(description="Order of this paragraph in the document")
    paragraph_type: Literal["introduction", "body", "conclusion", "transition"] = Field(
        default="body", 
        description="Type of paragraph"
    )
    confidence: float = Field(default=0.8, description="Confidence in paragraph quality")


class Issue(BaseModel):
    """Represents an issue found in a paragraph."""
    id: str = Field(description="Unique identifier for the issue")
    paragraph_id: str = Field(description="ID of the paragraph with the issue")
    issue_type: Literal["missing_justification", "vague_language", "unclear_flow"] = Field(
        description="Type of issue found"
    )
    severity: Literal["low", "medium", "high"] = Field(description="Severity of the issue")
    description: str = Field(description="Description of the issue")
    reason: str = Field(description="Explanation of why this is an issue")
    suggestion: str = Field(description="Suggestion for how to fix the issue")
    confidence: float = Field(default=0.8, description="Confidence in the issue detection")


class ReviewState(BaseModel):
    """State for the Review workflow."""
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
    assertions: List[Assertion] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    ordered_assertion_ids: List[str] = Field(default_factory=list, description="Ordered list of assertion IDs from structure mode")
    extracted_paragraphs: List[Paragraph] = Field(default_factory=list)
    ordered_paragraphs: List[Paragraph] = Field(default_factory=list)
    justification_issues: List[Issue] = Field(default_factory=list)
    vagueness_issues: List[Issue] = Field(default_factory=list)
    flow_issues: List[Issue] = Field(default_factory=list)
    all_issues: List[Issue] = Field(default_factory=list)
    current_input: str = Field(default="")
    chat_summary: str = Field(default="")