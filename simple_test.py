"""
Simple test to verify issue detection is working
"""

import json
from models import Assertion, Relationship
from review import ReviewWorkflow

def test_issue_detection():
    """Test issue detection with obvious issues"""
    
    # Create assertions with obvious issues
    assertions = [
        Assertion(
            id="assertion_1",
            content="Our product is amazing and everyone loves it",
            confidence=0.9,
            source="Internal assessment"
        ),
        Assertion(
            id="assertion_2", 
            content="Many users think the interface is good",
            confidence=0.8,
            source="Some feedback"
        )
    ]
    
    relationships = [
        Relationship(
            assertion1_id="assertion_2",
            assertion2_id="assertion_1",
            relationship_type="evidence",
            confidence=0.8,
            explanation="User feedback provides evidence for product quality"
        )
    ]
    
    # Run the workflow
    workflow = ReviewWorkflow()
    result = workflow.run(assertions, relationships)
    
    # Check results
    if 'document_review' in result:
        review = result['document_review']
        print(f"Total issues found: {len(review.issues)}")
        for issue in review.issues:
            print(f"- {issue.issue_type.issue_type}: {issue.issue_type.description}")
    else:
        print("No document review found in result")
    
    return result

if __name__ == "__main__":
    test_issue_detection()

