"""
Demo script for the Review Mode workflow.

This script demonstrates how to use the Review workflow with sample data
and shows the integration with the existing Idea Capture and Structure workflows.
"""

import os
import sys
from typing import List

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import Assertion, Relationship
from review import ReviewWorkflow


def create_sample_data():
    """Create sample assertions and relationships for demonstration."""
    
    # Sample assertions from a hypothetical user interface improvement project
    assertions = [
        Assertion(
            id="assertion_1",
            content="The current user interface is confusing and difficult to navigate",
            confidence=0.9,
            source="User feedback and usability testing"
        ),
        Assertion(
            id="assertion_2", 
            content="Users are having trouble finding the main features of the application",
            confidence=0.8,
            source="Analytics data and user interviews"
        ),
        Assertion(
            id="assertion_3",
            content="We need to prioritize mobile responsiveness in our design",
            confidence=0.7,
            source="Market research and device usage statistics"
        ),
        Assertion(
            id="assertion_4",
            content="Many users have requested dark mode support",
            confidence=0.9,
            source="Feature request surveys and user feedback"
        ),
        Assertion(
            id="assertion_5",
            content="The current design lacks accessibility features",
            confidence=0.6,
            source="Accessibility audit report"
        ),
        Assertion(
            id="assertion_6",
            content="Performance issues are causing user frustration",
            confidence=0.8,
            source="Performance monitoring and user complaints"
        ),
        Assertion(
            id="assertion_7",
            content="The onboarding process is too complex for new users",
            confidence=0.7,
            source="User onboarding analytics and feedback"
        ),
        Assertion(
            id="assertion_8",
            content="Search functionality is not working effectively",
            confidence=0.6,
            source="Search analytics and user reports"
        )
    ]
    
    # Sample relationships between assertions
    relationships = [
        Relationship(
            assertion1_id="assertion_2",
            assertion2_id="assertion_1",
            relationship_type="evidence",
            confidence=0.8,
            explanation="User difficulty finding features provides evidence for UI confusion"
        ),
        Relationship(
            assertion1_id="assertion_3",
            assertion2_id="assertion_1",
            relationship_type="background",
            confidence=0.6,
            explanation="Mobile responsiveness provides context for UI design considerations"
        ),
        Relationship(
            assertion1_id="assertion_4",
            assertion2_id="assertion_1",
            relationship_type="evidence",
            confidence=0.7,
            explanation="User requests for dark mode support evidence of UI issues"
        ),
        Relationship(
            assertion1_id="assertion_5",
            assertion2_id="assertion_1",
            relationship_type="evidence",
            confidence=0.5,
            explanation="Accessibility issues contribute to overall UI confusion"
        ),
        Relationship(
            assertion1_id="assertion_6",
            assertion2_id="assertion_1",
            relationship_type="cause",
            confidence=0.7,
            explanation="Performance issues can cause UI confusion and navigation problems"
        ),
        Relationship(
            assertion1_id="assertion_7",
            assertion2_id="assertion_1",
            relationship_type="evidence",
            confidence=0.6,
            explanation="Complex onboarding indicates UI confusion"
        ),
        Relationship(
            assertion1_id="assertion_8",
            assertion2_id="assertion_2",
            relationship_type="evidence",
            confidence=0.8,
            explanation="Poor search functionality directly causes user difficulty finding features"
        ),
        Relationship(
            assertion1_id="assertion_3",
            assertion2_id="assertion_4",
            relationship_type="contrast",
            confidence=0.4,
            explanation="Mobile responsiveness and dark mode are different design priorities"
        )
    ]
    
    return assertions, relationships


def print_assertions(assertions: List[Assertion]):
    """Print assertions in a formatted way."""
    print("\n" + "="*80)
    print("ASSERTIONS")
    print("="*80)
    
    for i, assertion in enumerate(assertions, 1):
        print(f"{i}. {assertion.content}")
        print(f"   ID: {assertion.id}")
        print(f"   Confidence: {assertion.confidence:.2f}")
        print(f"   Source: {assertion.source}")
        print()


def print_relationships(relationships: List[Relationship]):
    """Print relationships in a formatted way."""
    print("\n" + "="*80)
    print("RELATIONSHIPS")
    print("="*80)
    
    for i, rel in enumerate(relationships, 1):
        print(f"{i}. {rel.assertion1_id} --[{rel.relationship_type}]--> {rel.assertion2_id}")
        print(f"   Confidence: {rel.confidence:.2f}")
        print(f"   Explanation: {rel.explanation}")
        print()


def print_document_plan(document_plan):
    """Print the document plan in a formatted way."""
    print("\n" + "="*80)
    print("DOCUMENT PLAN")
    print("="*80)
    
    print(f"Title: {document_plan.title}")
    print(f"Document Type: {document_plan.document_type}")
    print(f"Target Audience: {document_plan.target_audience}")
    print(f"Overall Flow: {document_plan.overall_flow}")
    print()
    
    for i, para in enumerate(document_plan.paragraphs, 1):
        print(f"Paragraph {i}: {para.topic}")
        print(f"  Main Assertion ID: {para.main_assertion_id}")
        print(f"  Order: {para.order}")
        
        if para.supporting_assertions:
            print("  Supporting Assertions:")
            for supp in para.supporting_assertions:
                print(f"    - {supp.role}: {supp.assertion_id} ({supp.explanation})")
        
        if para.transition_notes:
            print(f"  Transition: {para.transition_notes}")
        print()


def print_review_results(document_review):
    """Print the review results in a formatted way."""
    print("\n" + "="*80)
    print("REVIEW RESULTS")
    print("="*80)
    
    print(f"Overall Quality Score: {document_review.overall_score:.2f}/1.0")
    print(f"Total Issues Found: {len(document_review.issues)}")
    print(f"Summary: {document_review.summary}")
    print()
    
    if document_review.issues:
        print("ISSUES BY TYPE:")
        issues_by_type = {}
        for issue in document_review.issues:
            issue_type = issue.issue_type.issue_type
            if issue_type not in issues_by_type:
                issues_by_type[issue_type] = []
            issues_by_type[issue_type].append(issue)
        
        for issue_type, issues in issues_by_type.items():
            print(f"\n{issue_type.replace('_', ' ').title()} ({len(issues)} issues):")
            for issue in issues:
                print(f"  - Paragraph {issue.paragraph_id}: {issue.issue_type.description}")
                print(f"    Severity: {issue.issue_type.severity}")
                print(f"    Suggestion: {issue.issue_type.suggestion}")
                print(f"    Confidence: {issue.confidence:.2f}")
    
    print("\nRECOMMENDATIONS:")
    for i, rec in enumerate(document_review.recommendations, 1):
        print(f"{i}. {rec}")


def main():
    """Main demo function."""
    print("Clarus Review Mode Demo")
    print("="*50)
    
    # Create sample data
    print("Creating sample assertions and relationships...")
    assertions, relationships = create_sample_data()
    
    # Print input data
    print_assertions(assertions)
    print_relationships(relationships)
    
    # Initialize the review workflow
    print("Initializing Review Workflow...")
    workflow = ReviewWorkflow()
    
    # Run the review workflow
    print("Running Review Workflow...")
    print("This may take a moment as the LLM analyzes the data...")
    
    try:
        result = workflow.run(assertions, relationships)
        
        # Extract results
        document_review = result.get('document_review')
        
        if document_review:
            # Print results
            print_document_plan(document_review.document_plan)
            print_review_results(document_review)
            
            print("\n" + "="*80)
            print("DEMO COMPLETE!")
            print("="*80)
            print("The review workflow has successfully:")
            print("1. Generated a document plan from assertions and relationships")
            print("2. Checked paragraph ordering and flow")
            print("3. Detected potential issues in parallel")
            print("4. Provided recommendations for improvement")
            print("\nYou can now use this system with your own assertions and relationships!")
            
        else:
            print("Error: No document review was generated.")
            
    except Exception as e:
        print(f"Error running review workflow: {str(e)}")
        print("Make sure you have the required dependencies installed and API keys configured.")


if __name__ == "__main__":
    main()
