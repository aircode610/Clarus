"""
Data models for assertions and relationships.

This module defines the core data structures used throughout the Clarus application.
"""

from typing import Literal
from pydantic import BaseModel, Field


class Assertion(BaseModel):
    """Represents a discrete, atomic assertion extracted from user input."""
    id: str = Field(description="Unique identifier for the assertion")
    content: str = Field(description="The actual assertion text")
    confidence: float = Field(description="Confidence score (0-1) for this assertion")
    source: str = Field(description="Source text that led to this assertion")


class Relationship(BaseModel):
    """Represents a relationship between two assertions."""
    assertion1_id: str = Field(description="ID of the first assertion")
    assertion2_id: str = Field(description="ID of the second assertion")
    relationship_type: Literal["evidence", "background", "cause", "contrast", "condition", "contradiction"] = Field(
        description="Type of relationship between the assertions"
    )
    confidence: float = Field(description="Confidence score (0-1) for this relationship")
    explanation: str = Field(description="Brief explanation of why this relationship exists")
