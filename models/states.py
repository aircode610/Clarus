"""
State models for LangGraph workflows.

This module defines the state classes used by the different workflow modes.
"""

from typing import List, Annotated, Optional, Literal
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from datetime import datetime

from .assertions import Assertion, Relationship, DocumentPlan, DocumentReview, ParagraphIssue


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


class ReviewState(BaseModel):
    """State for the Review workflow."""
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
    assertions: List[Assertion] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    document_plan: Optional[DocumentPlan] = Field(default=None)
    document_review: Optional[DocumentReview] = Field(default=None)
    current_input: str = Field(default="")
    chat_summary: str = Field(default="")
    plan_iteration: int = Field(default=0, description="Number of plan refinement iterations")
    review_complete: bool = Field(default=False, description="Whether the review process is complete")
