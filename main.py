"""
Main entry point for the Clarus application.

This file provides a simple command-line interface to run the Clarus workflow.
"""

import sys
import os
from app import ClarusApp, create_clarus_app, run_full_workflow


def main():
    """Main entry point for the Clarus application."""
    print("Clarus - Intelligent Document Structuring System")
    print("=" * 50)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key before running Clarus.")
        sys.exit(1)
    
    # Get user input
    print("\nEnter your ideas or thoughts (type 'quit' to exit):")
    user_input = input("> ")
    
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        sys.exit(0)
    
    if not user_input.strip():
        print("No input provided. Exiting.")
        sys.exit(0)
    
    try:
        # Run the full workflow
        print("\nRunning Clarus workflow...")
        result = run_full_workflow(user_input)
        
        # Display results
        print("\n" + "=" * 50)
        print("RESULTS")
        print("=" * 50)
        
        # Show assertions
        assertions = result["final_assertions"]
        print(f"\nExtracted {len(assertions)} assertions:")
        for i, assertion in enumerate(assertions, 1):
            print(f"{i}. {assertion.content} (confidence: {assertion.confidence:.2f})")
        
        # Show structure analysis
        structure_messages = result["structure_analysis"].get("messages", [])
        if structure_messages:
            print(f"\nStructure Analysis:")
            for message in structure_messages:
                if hasattr(message, 'content'):
                    print(f"  {message.content}")
        
        print(f"\nWorkflow completed successfully!")
        
    except Exception as e:
        print(f"Error running Clarus workflow: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
