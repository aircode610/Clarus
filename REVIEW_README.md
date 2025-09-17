# Clarus Review Mode

The Review Mode is the third component of the Clarus project, designed to take assertions and their relationships from the Structure mode and create a comprehensive document plan with issue detection.

## Overview

The Review Mode performs two main functions:

1. **Document Plan Generation**: Creates a structured document plan from assertions and relationships
2. **Issue Detection**: Identifies potential problems in the document plan using parallel LLM analysis

## Architecture

### Data Structures

The review system introduces several new data models:

- **`DocumentPlan`**: Complete document structure with title, paragraphs, and flow
- **`Paragraph`**: Individual paragraph with main assertion and supporting assertions
- **`SupportingAssertion`**: Supporting assertion with its role (evidence, background, etc.)
- **`DocumentReview`**: Complete review results with issues and recommendations
- **`ParagraphIssue`**: Specific issue found in a paragraph
- **`IssueType`**: Type of issue (missing justification, vague language, etc.)

### Workflow

The Review workflow uses LangGraph with the following nodes:

1. **`generate_plan`**: Creates document plan from assertions and relationships
2. **`check_ordering`**: Analyzes paragraph ordering and logical flow
3. **`refine_plan`**: Refines the plan based on ordering feedback
4. **Parallel Issue Detection**:
   - `check_missing_justification`
   - `check_vague_language`
   - `check_unclear_flow`
   - `check_weak_evidence`
   - `check_logical_gaps`
5. **`merge_issues`**: Combines all issue detection results
6. **`present_review`**: Formats results for display
7. **`summarize_review`**: Creates final summary

## Usage

### Basic Usage

```python
from review import ReviewWorkflow
from models import Assertion, Relationship

# Create workflow
workflow = ReviewWorkflow()

# Run with assertions and relationships
result = workflow.run(assertions, relationships)
document_review = result['document_review']
```

### Integration with ClarusApp

```python
from app import ClarusApp

# Create app
app = ClarusApp()

# Run idea capture
idea_result = app.start_idea_capture("Your initial input...")

# Run structure analysis
structure_result = app.start_structure_analysis()

# Run review analysis
review_result = app.start_review_analysis()
```

### Using the Streamlit UI

```bash
streamlit run streamlit_review.py
```

The UI provides:
- Interactive plan visualization
- Issue highlighting by severity
- Quality score gauges
- Recommendations display

## Issue Types

The system detects five types of issues:

1. **Missing Justification**: Claims without sufficient evidence
2. **Vague Language**: Imprecise or unclear terms
3. **Unclear Flow**: Poor logical connections between paragraphs
4. **Weak Evidence**: Insufficient support for main assertions
5. **Logical Gaps**: Missing steps in reasoning

Each issue includes:
- Severity level (low, medium, high)
- Specific description
- Improvement suggestions
- Confidence score
- Affected assertions

## Document Plan Structure

The generated document plan includes:

- **Title**: Proposed document title
- **Document Type**: Essay, report, article, etc.
- **Target Audience**: Intended readers
- **Overall Flow**: Description of document structure
- **Paragraphs**: Each with:
  - Main assertion
  - Supporting assertions with roles
  - Topic description
  - Transition notes

## Example Output

```json
{
  "title": "Improving User Interface Design",
  "document_type": "report",
  "target_audience": "development team",
  "paragraphs": [
    {
      "paragraph_id": "para_1",
      "main_assertion_id": "assertion_1",
      "topic": "Current UI Problems",
      "supporting_assertions": [
        {
          "assertion_id": "assertion_2",
          "role": "evidence",
          "explanation": "Provides evidence for UI confusion"
        }
      ]
    }
  ]
}
```

## Quality Scoring

The system provides an overall quality score (0-1) based on:
- Number of issues detected
- Severity of issues
- Confidence in issue detection
- Document structure quality

## Recommendations

The system generates actionable recommendations such as:
- "Add more evidence and justification for your main claims"
- "Use more specific and precise language throughout the document"
- "Improve transitions and logical flow between paragraphs"

## Files

- `review.py`: Main workflow implementation
- `streamlit_review.py`: Web UI for review mode
- `run_review_demo.py`: Demo script with sample data
- `models/assertions.py`: Data models for review system
- `models/states.py`: State management for review workflow

## Dependencies

- `langgraph`: Workflow orchestration
- `langchain`: LLM integration
- `streamlit`: Web UI
- `plotly`: Visualizations
- `pydantic`: Data validation

## Configuration

The review system uses the same configuration as other Clarus components:

```python
config = {
    "model_name": "gpt-4o-mini",  # or "gpt-4o"
    "temperature": 0.3
}
```

## Future Enhancements

Potential improvements include:
- Custom issue types
- User-defined quality criteria
- Integration with document writing tools
- Export to various formats (Word, LaTeX, etc.)
- Collaborative review features
