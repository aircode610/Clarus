"""
State models for LangGraph workflows.

This module defines the state classes used by the different workflow modes.
"""

from typing import List, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

from .assertions import Assertion, Relationship


class IdeaCaptureState(BaseModel):
    """State for the Idea Capture workflow."""
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
    assertions: List[Assertion] = Field(default_factory=list)
    current_input: str = Field(default="")
    iteration_count: int = Field(default=0)
    chat_summary: str = Field(default="")


class StructureState(BaseModel):
    """State for the Structure workflow."""
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
    assertions: List[Assertion] = Field(default_factory=list)
    evidence_relationships: List[Relationship] = Field(default_factory=list)
    background_relationships: List[Relationship] = Field(default_factory=list)
    cause_relationships: List[Relationship] = Field(default_factory=list)
    contrast_relationships: List[Relationship] = Field(default_factory=list)
    condition_relationships: List[Relationship] = Field(default_factory=list)
    final_relationships: List[Relationship] = Field(default_factory=list)
    current_input: str = Field(default="")
    chat_summary: str = Field(default="")
