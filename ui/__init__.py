"""
UI package for Clarus application.

This package contains all the Streamlit UI components organized by functionality:
- common: Shared UI components and utilities
- idea_ui: Idea capture mode UI components
- structure_ui: Structure mode UI components  
- review_ui: Review mode UI components
- prose_ui: Prose mode UI components
"""

from .common import display_assertions, create_assertion_groups
from .structure_ui import structure_tab
from .review_ui import review_tab
from .prose_ui import prose_tab

# Import idea_ui only if voice dependencies are available
try:
    from .idea_ui import idea_capture_tab
    _IDEA_UI_AVAILABLE = True
except ImportError:
    _IDEA_UI_AVAILABLE = False
    idea_capture_tab = None

__all__ = [
    "display_assertions",
    "create_assertion_groups", 
    "structure_tab",
    "review_tab",
    "prose_tab"
]

if _IDEA_UI_AVAILABLE:
    __all__.append("idea_capture_tab")
