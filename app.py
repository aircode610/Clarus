"""
Main Clarus Application

This module provides the main application class that orchestrates both Idea Capture
and Structure modes, allowing users to seamlessly move from brainstorming to 
structured document creation.
"""

from typing import List, Dict, Any, Optional
import os
from models import Assertion
from idea_capture import IdeaCaptureWorkflow
from structure import StructureWorkflow


class ClarusApp:
    """
    Main application class that orchestrates the Clarus workflow.
    
    This class manages the transition between Idea Capture and Structure modes,
    maintaining state and providing a unified interface for the entire process.
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini", config: dict = None):
        """
        Initialize the Clarus application.
        
        Args:
            model_name: The OpenAI model to use for both workflows
            config: Optional configuration dictionary
        """
        self.model_name = model_name
        self.config = config or {}
        
        # Initialize workflows
        self.idea_capture_workflow = IdeaCaptureWorkflow(model_name, config)
        self.structure_workflow = StructureWorkflow(model_name, config)
        
        # Application state
        self.current_assertions: List[Assertion] = []
        self.current_mode: str = "idea_capture"  # or "structure"
        self.session_id: str = "default"
    
    def start_idea_capture(self, initial_input: str, session_id: str = None) -> Dict[str, Any]:
        """
        Start the Idea Capture workflow.
        
        Args:
            initial_input: The initial user input to extract assertions from
            session_id: Optional session ID for conversation continuity
            
        Returns:
            Dictionary containing the workflow result
        """
        if session_id:
            self.session_id = session_id
            
        self.current_mode = "idea_capture"
        
        result = self.idea_capture_workflow.run(initial_input, self.session_id)
        
        # Extract assertions from the result
        if "assertions" in result:
            self.current_assertions = result["assertions"]
        
        return result
    
    def continue_idea_capture(self, user_input: str) -> Dict[str, Any]:
        """
        Continue the Idea Capture workflow with additional user input.
        
        Args:
            user_input: Additional user input for the conversation
            
        Returns:
            Dictionary containing the workflow result
        """
        if self.current_mode != "idea_capture":
            raise ValueError("Not currently in idea capture mode")
        
        # Create a new state with the user input
        from models import IdeaCaptureState
        from langchain_core.messages import HumanMessage
        
        # Get the current state from the workflow
        config = {"configurable": {"thread_id": self.session_id}}
        current_state = self.idea_capture_workflow.graph.get_state(config)
        
        # Add the new user message
        new_messages = current_state.values.get("messages", []) + [HumanMessage(content=user_input)]
        
        # Create new state
        new_state = IdeaCaptureState(
            messages=new_messages,
            assertions=self.current_assertions,
            current_input=user_input
        )
        
        # Continue the workflow
        result = self.idea_capture_workflow.graph.invoke(new_state, config)
        
        # Update current assertions
        if "assertions" in result:
            self.current_assertions = result["assertions"]
        
        return result
    
    def start_structure_analysis(self, assertions: List[Assertion] = None, session_id: str = None) -> Dict[str, Any]:
        """
        Start the Structure analysis workflow.
        
        Args:
            assertions: List of assertions to analyze. If None, uses current assertions
            session_id: Optional session ID for the analysis
            
        Returns:
            Dictionary containing the structure analysis result
        """
        if session_id:
            self.session_id = session_id
            
        self.current_mode = "structure"
        
        # Use provided assertions or current ones
        assertions_to_analyze = assertions or self.current_assertions
        
        if not assertions_to_analyze:
            raise ValueError("No assertions available for structure analysis")
        
        result = self.structure_workflow.run(assertions_to_analyze, self.session_id)
        return result
    
    def get_current_assertions(self) -> List[Assertion]:
        """Get the current list of assertions."""
        return self.current_assertions.copy()
    
    def get_current_mode(self) -> str:
        """Get the current workflow mode."""
        return self.current_mode
    
    def reset_session(self):
        """Reset the application state for a new session."""
        self.current_assertions = []
        self.current_mode = "idea_capture"
        self.session_id = "default"
    
    def export_assertions(self) -> List[Dict[str, Any]]:
        """
        Export current assertions as dictionaries.
        
        Returns:
            List of assertion dictionaries
        """
        return [assertion.model_dump() for assertion in self.current_assertions]
    
    def import_assertions(self, assertions_data: List[Dict[str, Any]]):
        """
        Import assertions from dictionaries.
        
        Args:
            assertions_data: List of assertion dictionaries
        """
        self.current_assertions = [Assertion(**data) for data in assertions_data]
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """
        Get the current status of the application.
        
        Returns:
            Dictionary containing status information
        """
        return {
            "current_mode": self.current_mode,
            "assertion_count": len(self.current_assertions),
            "session_id": self.session_id,
            "model_name": self.model_name
        }


# Convenience functions for direct usage
def create_clarus_app(model_name: str = "gpt-4o-mini", config: dict = None) -> ClarusApp:
    """
    Create a new Clarus application instance.
    
    Args:
        model_name: The OpenAI model to use
        config: Optional configuration dictionary
        
    Returns:
        New ClarusApp instance
    """
    return ClarusApp(model_name, config)


def run_full_workflow(initial_input: str, model_name: str = "gpt-4o-mini") -> Dict[str, Any]:
    """
    Run the complete Clarus workflow from idea capture to structure analysis.
    
    Args:
        initial_input: Initial user input for idea capture
        model_name: The OpenAI model to use
        
    Returns:
        Dictionary containing both idea capture and structure analysis results
    """
    app = create_clarus_app(model_name)
    
    # Run idea capture
    idea_result = app.start_idea_capture(initial_input)
    
    # Run structure analysis
    structure_result = app.start_structure_analysis()
    
    return {
        "idea_capture": idea_result,
        "structure_analysis": structure_result,
        "final_assertions": app.get_current_assertions(),
        "status": app.get_workflow_status()
    }


# Example usage
if __name__ == "__main__":
    # Create application
    app = create_clarus_app()
    
    # Example input
    sample_input = """
    I think we should focus on building a better user interface for our application. 
    The current design is confusing and users are having trouble finding the main features. 
    We need to prioritize mobile responsiveness and make sure the navigation is intuitive. 
    Also, I've been thinking about adding dark mode support since many users have requested it.
    """
    
    # Run idea capture
    print("Running Idea Capture...")
    idea_result = app.start_idea_capture(sample_input)
    print(f"Extracted {len(app.get_current_assertions())} assertions")
    
    # Run structure analysis
    print("\nRunning Structure Analysis...")
    structure_result = app.start_structure_analysis()
    print("Structure analysis complete!")
    
    # Print final status
    print(f"\nFinal Status: {app.get_workflow_status()}")
