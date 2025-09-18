"""
Workflows package for Clarus application.

This package contains all the LangGraph workflows for the different modes:
- Idea Capture: Extract assertions from user input
- Structure: Analyze relationships between assertions  
- Review: Create structured paragraphs and identify issues
- Prose: Generate final fluent text
"""

from .idea_capture import IdeaCaptureWorkflow, create_idea_capture_workflow, create_idea_capture_graph
from .structure import StructureWorkflow, create_structure_workflow, create_structure_graph, evaluate_relationship_quality
from .review import ReviewWorkflow, create_review_workflow, create_review_graph
from .prose import ProseWorkflow, create_prose_workflow

__all__ = [
    "IdeaCaptureWorkflow",
    "create_idea_capture_workflow", 
    "create_idea_capture_graph",
    "StructureWorkflow",
    "create_structure_workflow",
    "create_structure_graph",
    "evaluate_relationship_quality",
    "ReviewWorkflow",
    "create_review_workflow",
    "create_review_graph",
    "ProseWorkflow",
    "create_prose_workflow"
]
