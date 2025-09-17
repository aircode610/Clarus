"""
Data models for assertions and relationships.

This module defines the core data structures used throughout the Clarus application.
"""

from typing import Literal, List, Optional
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


class SupportingAssertion(BaseModel):
    """Represents a supporting assertion in a paragraph with its role."""
    assertion_id: str = Field(description="ID of the supporting assertion")
    role: Literal["evidence", "background", "cause", "contrast", "condition", "example"] = Field(
        description="Role of this assertion in supporting the main assertion"
    )
    explanation: str = Field(description="How this assertion supports the main one")


class Paragraph(BaseModel):
    """Represents a paragraph in the document plan."""
    paragraph_id: str = Field(description="Unique identifier for the paragraph")
    main_assertion_id: str = Field(description="ID of the main assertion for this paragraph")
    supporting_assertions: List[SupportingAssertion] = Field(default_factory=list)
    order: int = Field(description="Order of this paragraph in the document")
    topic: str = Field(description="Brief topic or theme of this paragraph")
    transition_notes: Optional[str] = Field(default=None, description="Notes about transition to next paragraph")


class DocumentPlan(BaseModel):
    """Represents the complete document plan."""
    plan_id: str = Field(description="Unique identifier for the plan")
    title: str = Field(description="Proposed title for the document")
    paragraphs: List[Paragraph] = Field(default_factory=list)
    overall_flow: str = Field(description="Description of the overall document flow")
    target_audience: str = Field(description="Intended audience for the document")
    document_type: str = Field(description="Type of document (essay, report, etc.)")


class IssueType(BaseModel):
    """Represents a type of issue that can be detected."""
    issue_type: Literal["missing_justification", "vague_language", "unclear_flow", "weak_evidence", "logical_gaps"] = Field(
        description="Type of issue detected"
    )
    severity: Literal["low", "medium", "high"] = Field(description="Severity of the issue")
    description: str = Field(description="Description of the issue")
    suggestion: str = Field(description="Suggestion for improvement")


class ParagraphIssue(BaseModel):
    """Represents an issue found in a specific paragraph."""
    issue_id: str = Field(description="Unique identifier for the issue")
    paragraph_id: str = Field(description="ID of the paragraph with the issue")
    issue_type: IssueType = Field(description="Type and details of the issue")
    affected_assertions: List[str] = Field(default_factory=list, description="IDs of assertions affected by this issue")
    location: str = Field(description="Specific location in the paragraph where the issue occurs")
    confidence: float = Field(description="Confidence score (0-1) for this issue detection")


class DocumentReview(BaseModel):
    """Represents the complete review of a document plan."""
    review_id: str = Field(description="Unique identifier for the review")
    document_plan: DocumentPlan = Field(description="The document plan being reviewed")
    issues: List[ParagraphIssue] = Field(default_factory=list)
    overall_score: float = Field(description="Overall quality score (0-1)")
    summary: str = Field(description="Summary of the review findings")
    recommendations: List[str] = Field(default_factory=list, description="High-level recommendations for improvement")
