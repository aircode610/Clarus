"""
Structure Mode Workflow using LangGraph

This module implements the second mode of the Clarus project - Structure.
Takes assertions from Idea Capture mode and analyzes relationships between them
using Rhetorical Structure Theory (RST) relationships.
"""

from typing import List, Dict, Any, Literal, Annotated, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json

from models import Assertion, Relationship, StructureState


def evaluate_relationship_quality(assertion1_content: str, assertion2_content: str, relationship_type: str) -> tuple[int, str, str]:
    """
    Evaluate the quality of a relationship using LLM.
    
    Args:
        assertion1_content: Content of the first assertion
        assertion2_content: Content of the second assertion
        relationship_type: Type of relationship (evidence, background, cause, contrast, condition)
    
    Returns:
        tuple: (confidence_score, reason, suggestion) where:
            - confidence_score is 0-100
            - reason is a short explanation of why it's good or bad
            - suggestion is one of: "ADD", "MODIFY", "REMOVE"
    """
    try:
        # Initialize LLM
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        
        # Create evaluation prompt
        prompt = f"""
You are an expert at analyzing relationships between assertions in academic and analytical writing. 
Evaluate the quality of the proposed relationship between these two assertions.

Assertion 1: "{assertion1_content}"
Assertion 2: "{assertion2_content}"
Proposed Relationship Type: {relationship_type}

Relationship Types:
- evidence: One assertion provides evidence, examples, or support for another
- background: One assertion provides context, setting, or foundational information for another
- cause: One assertion directly causes or leads to another
- contrast: Assertions present opposing viewpoints, contradictions, or different perspectives
- condition: One assertion is a prerequisite or condition for another

Rate this relationship on a scale of 0-100 and provide structured feedback.

Be strict but fair in your evaluation. Consider:
1. Logical coherence of the relationship
2. Strength of the connection
3. Clarity of the relationship type
4. Whether the assertions actually relate in the proposed way

Respond in this exact format:
CONFIDENCE: [0-100]
REASON: [One short sentence explaining why it's good or bad]
SUGGESTION: [One of: "ADD" if good, "MODIFY to [relationship_type]" if needs different relationship type, "REMOVE" if poor]

Examples of good relationships:
- Evidence: "AI can detect cancer" → "AI is useful in healthcare" (CONFIDENCE: 85-95, SUGGESTION: ADD)
- Cause: "Increased CO2 levels" → "Global warming" (CONFIDENCE: 90-100, SUGGESTION: ADD)
- Contrast: "AI will replace humans" vs "AI will augment humans" (CONFIDENCE: 80-95, SUGGESTION: ADD)

Examples of relationships needing modification:
- "AI algorithms improved" → "AI use in healthcare increased" (CONFIDENCE: 60-70, SUGGESTION: MODIFY to cause)
- "Some studies show benefits" → "Technology is advancing" (CONFIDENCE: 40-60, SUGGESTION: MODIFY to evidence)

Examples of poor relationships:
- "It's sunny today" → "Technology is advancing" (CONFIDENCE: 0-20, SUGGESTION: REMOVE)
- "Apples are red" vs "Oranges are orange" (CONFIDENCE: 10-30, SUGGESTION: REMOVE)
"""

        # Get LLM response
        response = llm.invoke([HumanMessage(content=prompt)])
        response_text = response.content
        
        # Parse response
        confidence = 50  # Default
        reason = "Unable to evaluate relationship quality."
        suggestion = "MODIFY"
        
        if "CONFIDENCE:" in response_text and "REASON:" in response_text and "SUGGESTION:" in response_text:
            try:
                confidence_line = [line for line in response_text.split('\n') if 'CONFIDENCE:' in line][0]
                confidence = int(confidence_line.split('CONFIDENCE:')[1].strip())
                confidence = max(0, min(100, confidence))  # Clamp between 0-100
                
                reason_line = [line for line in response_text.split('\n') if 'REASON:' in line][0]
                reason = reason_line.split('REASON:')[1].strip()
                
                suggestion_line = [line for line in response_text.split('\n') if 'SUGGESTION:' in line][0]
                suggestion = suggestion_line.split('SUGGESTION:')[1].strip().upper()
                
                # Parse suggestion and extract relationship type if MODIFY
                if suggestion.startswith('MODIFY TO '):
                    # Extract the suggested relationship type
                    suggested_type = suggestion.replace('MODIFY TO ', '').strip()
                    suggestion = f"MODIFY to {suggested_type}"
                elif suggestion not in ['ADD', 'REMOVE']:
                    suggestion = 'MODIFY'
                    
            except (ValueError, IndexError):
                pass
        
        return confidence, reason, suggestion
        
    except Exception as e:
        return 50, f"Error evaluating relationship: {str(e)}", "MODIFY"


class StructureWorkflow:
    """Main workflow class for Structure mode."""
    
    def __init__(self, model_name: str = "gpt-4o-mini", config: dict = None):
        # Handle both direct instantiation and LangGraph Studio config
        if config is not None and isinstance(config, dict):
            # Extract model name from config if available
            model_name = config.get("model_name", "gpt-4o-mini")
        
        self.llm = ChatOpenAI(model=model_name, temperature=0.3)
        self.memory = MemorySaver()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow with parallel relationship analysis."""
        builder = StateGraph(StructureState)
        
        # Add nodes
        builder.add_node("analyze_evidence", self._analyze_evidence_node)
        builder.add_node("analyze_background", self._analyze_background_node)
        builder.add_node("analyze_cause", self._analyze_cause_node)
        builder.add_node("analyze_contrast", self._analyze_contrast_node)
        builder.add_node("analyze_condition", self._analyze_condition_node)
        builder.add_node("analyze_contradiction", self._analyze_contradiction_node)
        builder.add_node("merge_relationships", self._merge_relationships_node)
        builder.add_node("evaluate_relationships", self._evaluate_relationships_node)
        builder.add_node("present_structure", self._present_structure_node)
        builder.add_node("summarize_structure", self._summarize_structure_node)
        
        # Add edges - all relationship analysis nodes run in parallel
        builder.add_edge(START, "analyze_evidence")
        builder.add_edge(START, "analyze_background")
        builder.add_edge(START, "analyze_cause")
        builder.add_edge(START, "analyze_contrast")
        builder.add_edge(START, "analyze_condition")
        builder.add_edge(START, "analyze_contradiction")
        
        # All parallel nodes feed into merge
        builder.add_edge("analyze_evidence", "merge_relationships")
        builder.add_edge("analyze_background", "merge_relationships")
        builder.add_edge("analyze_cause", "merge_relationships")
        builder.add_edge("analyze_contrast", "merge_relationships")
        builder.add_edge("analyze_condition", "merge_relationships")
        builder.add_edge("analyze_contradiction", "merge_relationships")
        
        # Merge feeds into evaluation, then presentation
        builder.add_edge("merge_relationships", "evaluate_relationships")
        builder.add_edge("evaluate_relationships", "present_structure")
        builder.add_edge("present_structure", "summarize_structure")
        builder.add_edge("summarize_structure", END)
        
        return builder.compile(checkpointer=self.memory)
    
    def _create_relationship_prompt(self, relationship_type: str) -> ChatPromptTemplate:
        """Create a prompt template for analyzing a specific relationship type."""
        
        relationship_descriptions = {
            "evidence": {
                "description": "Evidence relationships where one assertion provides evidence, examples, support, or substantiation for another",
                "examples": "A provides evidence for B, A is an example of B, A supports B's claim, A demonstrates B, A illustrates B"
            },
            "background": {
                "description": "Background relationships where one assertion provides context, setting, background information, or historical context for another",
                "examples": "A provides background for B, A sets the context for B, A gives necessary information for understanding B, A explains the history of B"
            },
            "cause": {
                "description": "Cause relationships where one assertion causes, leads to, results in, or contributes to another",
                "examples": "A causes B, A leads to B, A results in B, A is the reason for B, A contributes to B, A triggers B"
            },
            "contrast": {
                "description": "Contrast relationships where assertions present different viewpoints, perspectives, or approaches (not contradictory)",
                "examples": "A contrasts with B, A differs from B, A presents alternative to B, A shows different approach than B"
            },
            "condition": {
                "description": "Condition relationships where one assertion is a prerequisite, condition, requirement, or enabling factor for another",
                "examples": "A is a condition for B, A is required for B, A must happen before B, A enables B, A determines B"
            },
            "contradiction": {
                "description": "Contradiction relationships where assertions directly contradict, negate, or are mutually exclusive with each other",
                "examples": "A contradicts B, A negates B, A is mutually exclusive with B, A directly opposes B's claim"
            }
        }
        
        rel_info = relationship_descriptions[relationship_type]
        
        return ChatPromptTemplate.from_messages([
            ("system", f"""You are an expert at analyzing rhetorical relationships between assertions using Rhetorical Structure Theory.

Your task is to find {relationship_type} relationships between the given assertions.

{relationship_type.upper()} RELATIONSHIPS:
{rel_info['description']}

Examples: {rel_info['examples']}

CRITICAL RULES:
- Look for relationships that fit the {relationship_type} pattern, even if subtle
- Be thorough - include relationships you can reasonably identify
- Each relationship should be between exactly 2 assertions
- Provide confidence scores (0-1) for each relationship
- Include brief explanations for why the relationship exists
- Consider temporal, causal, logical, and semantic connections

IMPORTANT: Return ONLY a valid JSON array. Do not include any other text.

Return your response as a JSON list of relationships. Each relationship should have:
- assertion1_id: ID of the first assertion
- assertion2_id: ID of the second assertion  
- relationship_type: "{relationship_type}"
- confidence: number between 0 and 1
- explanation: brief explanation of the relationship

EXAMPLES OF {relationship_type.upper()} RELATIONSHIPS:
- Historical events providing background for current situations
- Data points providing evidence for broader claims
- Causes leading to effects or consequences
- Conditions that must be met for outcomes
- Contrasting viewpoints or different perspectives

If no {relationship_type} relationships are found, return an empty array: []"""),
            ("human", "Analyze these assertions for {relationship_type} relationships:\n\n{assertions_text}")
        ])
    
    def _analyze_evidence_node(self, state: StructureState) -> Dict[str, Any]:
        """Analyze evidence relationships between assertions."""
        return self._analyze_relationships(state, "evidence")
    
    def _analyze_background_node(self, state: StructureState) -> Dict[str, Any]:
        """Analyze background relationships between assertions."""
        return self._analyze_relationships(state, "background")
    
    def _analyze_cause_node(self, state: StructureState) -> Dict[str, Any]:
        """Analyze cause relationships between assertions."""
        return self._analyze_relationships(state, "cause")
    
    def _analyze_contrast_node(self, state: StructureState) -> Dict[str, Any]:
        """Analyze contrast relationships between assertions."""
        return self._analyze_relationships(state, "contrast")
    
    def _analyze_condition_node(self, state: StructureState) -> Dict[str, Any]:
        """Analyze condition relationships between assertions."""
        return self._analyze_relationships(state, "condition")
    
    def _analyze_contradiction_node(self, state: StructureState) -> Dict[str, Any]:
        """Analyze contradiction relationships between assertions."""
        return self._analyze_relationships(state, "contradiction")
    
    def _analyze_relationships(self, state: StructureState, relationship_type: str) -> Dict[str, Any]:
        """Generic method to analyze relationships of a specific type."""
        if not state.assertions or len(state.assertions) < 2:
            return {}
        
        # Format assertions for the prompt
        assertions_text = "\n".join([
            f"ID: {assertion.id}\nContent: {assertion.content}\n"
            for assertion in state.assertions
        ])
        
        # Create and run the prompt
        prompt = self._create_relationship_prompt(relationship_type)
        chain = prompt | self.llm
        response = chain.invoke({
            "assertions_text": assertions_text,
            "relationship_type": relationship_type
        })
        
        try:
            # Clean the response content
            content = response.content.strip()
            
            # Try to extract JSON from the response if it's wrapped in markdown
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            
            # Parse JSON response
            relationships_data = json.loads(content)
            
            # Ensure it's a list
            if not isinstance(relationships_data, list):
                relationships_data = [relationships_data]
            
            # Create relationship objects
            relationships = []
            for rel_data in relationships_data:
                if not isinstance(rel_data, dict):
                    continue
                
                # Ensure required fields exist
                rel_data.setdefault("relationship_type", relationship_type)
                rel_data.setdefault("confidence", 0.8)
                rel_data.setdefault("explanation", f"{relationship_type} relationship")
                
                # Validate that both assertion IDs exist
                assertion1_id = rel_data.get("assertion1_id")
                assertion2_id = rel_data.get("assertion2_id")
                
                # Convert to strings if they're integers (LLM sometimes returns integers)
                if isinstance(assertion1_id, int):
                    assertion1_id = str(assertion1_id)
                if isinstance(assertion2_id, int):
                    assertion2_id = str(assertion2_id)
                
                if (assertion1_id and assertion2_id and 
                    assertion1_id != assertion2_id and
                    any(a.id == assertion1_id for a in state.assertions) and
                    any(a.id == assertion2_id for a in state.assertions)):
                    
                    # Update the relationship data with converted IDs
                    rel_data["assertion1_id"] = assertion1_id
                    rel_data["assertion2_id"] = assertion2_id
                    
                    relationships.append(Relationship(**rel_data))
            
            # Return the relationships for the specific type
            field_name = f"{relationship_type}_relationships"
            return {field_name: relationships}
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Return empty relationships on error
            field_name = f"{relationship_type}_relationships"
            return {field_name: []}
    
    def _merge_relationships_node(self, state: StructureState) -> Dict[str, Any]:
        """Merge all relationship types and resolve conflicts."""
        # Collect all relationships
        all_relationships = []
        all_relationships.extend(state.evidence_relationships)
        all_relationships.extend(state.background_relationships)
        all_relationships.extend(state.cause_relationships)
        all_relationships.extend(state.contrast_relationships)
        all_relationships.extend(state.condition_relationships)
        all_relationships.extend(state.contradiction_relationships)
        
        if not all_relationships:
            return {
                "final_relationships": [],
                "messages": state.messages + [
                    AIMessage(content="I analyzed all the assertions but didn't find any clear relationships between them.")
                ]
            }
        
        # Find duplicate pairs (same assertion IDs but different relationship types)
        relationship_pairs = {}
        for rel in all_relationships:
            # Create a consistent pair key (sorted IDs)
            pair_key = tuple(sorted([rel.assertion1_id, rel.assertion2_id]))
            
            if pair_key not in relationship_pairs:
                relationship_pairs[pair_key] = []
            relationship_pairs[pair_key].append(rel)
        
        # Resolve conflicts for pairs with multiple relationship types
        final_relationships = []
        conflicts_to_resolve = []
        
        for pair_key, relationships in relationship_pairs.items():
            if len(relationships) == 1:
                # No conflict, add the relationship
                final_relationships.append(relationships[0])
            else:
                # Multiple relationship types for the same pair - need to resolve
                conflicts_to_resolve.append(relationships)
        
        # If there are conflicts, use LLM to resolve them
        if conflicts_to_resolve:
            resolved_relationships = self._resolve_relationship_conflicts(
                conflicts_to_resolve, state.assertions
            )
            final_relationships.extend(resolved_relationships)
        
        return {
            "final_relationships": final_relationships,
            "messages": state.messages + [
                AIMessage(content=f"I found {len(all_relationships)} potential relationships and resolved any conflicts. Final count: {len(final_relationships)} relationships.")
            ]
        }
    
    def _resolve_relationship_conflicts(self, conflicts: List[List[Relationship]], assertions: List[Assertion]) -> List[Relationship]:
        """Use LLM to resolve conflicts when multiple relationship types exist for the same assertion pair."""
        if not conflicts:
            return []
        
        # Create assertions lookup
        assertions_dict = {a.id: a for a in assertions}
        
        resolved_relationships = []
        
        for conflict_group in conflicts:
            if not conflict_group:
                continue
            
            # Get the assertion pair
            rel = conflict_group[0]
            assertion1 = assertions_dict.get(rel.assertion1_id)
            assertion2 = assertions_dict.get(rel.assertion2_id)
            
            if not assertion1 or not assertion2:
                continue
            
            # Create prompt for conflict resolution
            conflict_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert at analyzing rhetorical relationships between assertions.

You need to choose the BEST relationship type for a pair of assertions that could have multiple relationship types.

Given these assertions and the possible relationship types, choose the ONE relationship type that best describes their relationship.

Consider:
1. Which relationship is most fundamental/primary?
2. Which relationship best captures the semantic connection?
3. Which relationship would be most useful for structuring a document?

IMPORTANT: Return ONLY a valid JSON object. Do not include any other text.

Return your choice as JSON with these fields:
- chosen_relationship: one of "evidence", "background", "cause", "contrast", "condition", or "contradiction"
- confidence: number between 0 and 1
- explanation: brief explanation of why this relationship type is best"""),
                ("human", """Assertion 1 (ID: {assertion1_id}): {assertion1_content}

Assertion 2 (ID: {assertion2_id}): {assertion2_content}

Possible relationship types:
{possible_relationships}

Choose the best relationship type for this pair.""")
            ])
            
            # Format possible relationships
            possible_rels = "\n".join([
                f"- {rel.relationship_type}: {rel.explanation} (confidence: {rel.confidence:.2f})"
                for rel in conflict_group
            ])
            
            # Run the conflict resolution
            chain = conflict_prompt | self.llm
            response = chain.invoke({
                "assertion1_id": assertion1.id,
                "assertion1_content": assertion1.content,
                "assertion2_id": assertion2.id,
                "assertion2_content": assertion2.content,
                "possible_relationships": possible_rels
            })
            
            try:
                # Clean and parse response
                content = response.content.strip()
                if content.startswith("```json"):
                    content = content.split("```json")[1].split("```")[0].strip()
                elif content.startswith("```"):
                    content = content.split("```")[1].split("```")[0].strip()
                
                resolution = json.loads(content)
                chosen_type = resolution.get("chosen_relationship")
                confidence = resolution.get("confidence", 0.8)
                explanation = resolution.get("explanation", f"Resolved conflict: {chosen_type}")
                
                # Find the original relationship with the chosen type
                chosen_rel = None
                for rel in conflict_group:
                    if rel.relationship_type == chosen_type:
                        chosen_rel = rel
                        break
                
                if chosen_rel:
                    # Update with resolved confidence and explanation
                    resolved_rel = Relationship(
                        assertion1_id=chosen_rel.assertion1_id,
                        assertion2_id=chosen_rel.assertion2_id,
                        relationship_type=chosen_rel.relationship_type,
                        confidence=confidence,
                        explanation=explanation
                    )
                    resolved_relationships.append(resolved_rel)
                else:
                    # Fallback to highest confidence relationship
                    best_rel = max(conflict_group, key=lambda r: r.confidence)
                    resolved_relationships.append(best_rel)
                    
            except (json.JSONDecodeError, KeyError, TypeError):
                # Fallback to highest confidence relationship
                best_rel = max(conflict_group, key=lambda r: r.confidence)
                resolved_relationships.append(best_rel)
        
        return resolved_relationships
    
    def _evaluate_relationships_node(self, state: StructureState) -> Dict[str, Any]:
        """Evaluate all relationships and filter based on quality assessment."""
        if not state.final_relationships:
            return {
                "evaluated_relationships": [],
                "messages": state.messages + [
                    AIMessage(content="No relationships to evaluate.")
                ]
            }
        
        # Create assertions lookup for evaluation
        assertions_dict = {a.id: a for a in state.assertions}
        
        evaluated_relationships = []
        evaluation_results = []
        
        for rel in state.final_relationships:
            assertion1 = assertions_dict.get(rel.assertion1_id)
            assertion2 = assertions_dict.get(rel.assertion2_id)
            
            if not assertion1 or not assertion2:
                continue
            
            # Evaluate the relationship quality
            confidence, reason, suggestion = evaluate_relationship_quality(
                assertion1.content,
                assertion2.content,
                rel.relationship_type
            )
            
            evaluation_results.append({
                "relationship": rel,
                "confidence": confidence,
                "reason": reason,
                "suggestion": suggestion
            })
            
            # Decide whether to keep the relationship based on evaluation
            if suggestion == "ADD" or (suggestion.startswith("MODIFY") and confidence >= 60):
                # Keep the relationship, potentially with modification
                if suggestion.startswith("MODIFY to "):
                    # Extract the suggested relationship type
                    suggested_type = suggestion.replace("MODIFY to ", "").strip()
                    if suggested_type in ["evidence", "background", "cause", "contrast", "condition", "contradiction"]:
                        # Update the relationship type
                        updated_rel = Relationship(
                            assertion1_id=rel.assertion1_id,
                            assertion2_id=rel.assertion2_id,
                            relationship_type=suggested_type,
                            confidence=confidence / 100.0,  # Convert to 0-1 scale
                            explanation=f"Modified from {rel.relationship_type}: {reason}"
                        )
                        evaluated_relationships.append(updated_rel)
                    else:
                        # Keep original if suggested type is invalid
                        evaluated_relationships.append(rel)
                else:
                    # Keep as is
                    evaluated_relationships.append(rel)
            elif suggestion == "REMOVE" or confidence < 50:
                # Remove the relationship
                continue
            else:
                # Keep with lower confidence
                updated_rel = Relationship(
                    assertion1_id=rel.assertion1_id,
                    assertion2_id=rel.assertion2_id,
                    relationship_type=rel.relationship_type,
                    confidence=confidence / 100.0,
                    explanation=f"Evaluated: {reason}"
                )
                evaluated_relationships.append(updated_rel)
        
        # Create evaluation summary
        kept_count = len(evaluated_relationships)
        removed_count = len(state.final_relationships) - kept_count
        
        evaluation_message = f"Evaluated {len(state.final_relationships)} relationships. Kept {kept_count} high-quality relationships and removed {removed_count} low-quality ones."
        
        return {
            "evaluated_relationships": evaluated_relationships,
            "messages": state.messages + [AIMessage(content=evaluation_message)]
        }
    
    def _present_structure_node(self, state: StructureState) -> Dict[str, Any]:
        """Present the structured relationships to the user."""
        if not state.evaluated_relationships:
            return {
                "messages": state.messages + [
                    AIMessage(content="I didn't find any clear relationships between your assertions. This might mean they are independent ideas or the relationships are too subtle to detect automatically.")
                ]
            }
        
        # Group relationships by type
        relationships_by_type = {}
        for rel in state.evaluated_relationships:
            if rel.relationship_type not in relationships_by_type:
                relationships_by_type[rel.relationship_type] = []
            relationships_by_type[rel.relationship_type].append(rel)
        
        # Create assertions lookup for display
        assertions_dict = {a.id: a for a in state.assertions}
        
        # Format the structure presentation
        structure_text = "Here's the structured analysis of your assertions:\n\n"
        
        for rel_type, relationships in relationships_by_type.items():
            structure_text += f"**{rel_type.upper()} RELATIONSHIPS:**\n"
            for i, rel in enumerate(relationships, 1):
                assertion1 = assertions_dict.get(rel.assertion1_id)
                assertion2 = assertions_dict.get(rel.assertion2_id)
                
                if assertion1 and assertion2:
                    structure_text += f"{i}. {assertion1.content}\n   → {rel_type.upper()} → {assertion2.content}\n"
                    structure_text += f"   (Confidence: {rel.confidence:.2f}) - {rel.explanation}\n\n"
        
        structure_text += "\nThis structure shows how your assertions relate to each other. You can use this to organize your ideas into a coherent document or presentation."
        
        return {
            "messages": state.messages + [AIMessage(content=structure_text)]
        }
    
    
    def _summarize_structure_node(self, state: StructureState) -> Dict[str, Any]:
        """Summarize the structure analysis."""
        if not state.evaluated_relationships:
            summary = "No relationships were found between the assertions."
        else:
            # Count relationships by type
            rel_counts = {}
            for rel in state.evaluated_relationships:
                rel_counts[rel.relationship_type] = rel_counts.get(rel.relationship_type, 0) + 1
            
            summary_parts = [f"Found {len(state.evaluated_relationships)} high-quality relationships:"]
            for rel_type, count in rel_counts.items():
                summary_parts.append(f"- {count} {rel_type} relationships")
            
            summary = "\n".join(summary_parts)
        
        final_message = f"Structure Analysis Complete!\n\n{summary}\n\nThe structured relationships can now be used to organize your assertions into a coherent document or presentation."
        
        return {
            "chat_summary": summary,
            "messages": state.messages + [AIMessage(content=final_message)]
        }
    
    def run(self, assertions: List[Assertion], thread_id: str = "default") -> Dict[str, Any]:
        """Run the structure workflow with a list of assertions."""
        config = {"configurable": {"thread_id": thread_id}}
        
        initial_state = StructureState(
            assertions=assertions,
            messages=[HumanMessage(content=f"Analyzing structure for {len(assertions)} assertions.")]
        )
        
        result = self.graph.invoke(initial_state, config)
        return result


# Factory function for LangGraph Studio
def create_structure_workflow(config: dict = None):
    """Factory function to create and return the compiled graph for LangGraph Studio."""
    workflow = StructureWorkflow(config=config)
    return workflow.graph


# Alternative: Direct graph creation function
def create_structure_graph(config: dict = None):
    """Create the structure graph directly for LangGraph Studio."""
    # Extract model name from config if available
    model_name = "gpt-4o-mini"
    if config and isinstance(config, dict):
        model_name = config.get("model_name", "gpt-4o-mini")
    
    # Create LLM
    llm = ChatOpenAI(model=model_name, temperature=0.3)
    
    # Create memory
    memory = MemorySaver()
    
    # Build graph
    builder = StateGraph(StructureState)
    
    # Create workflow instance for node methods
    workflow = StructureWorkflow(model_name=model_name)
    
    # Add nodes
    builder.add_node("analyze_evidence", workflow._analyze_evidence_node)
    builder.add_node("analyze_background", workflow._analyze_background_node)
    builder.add_node("analyze_cause", workflow._analyze_cause_node)
    builder.add_node("analyze_contrast", workflow._analyze_contrast_node)
    builder.add_node("analyze_condition", workflow._analyze_condition_node)
    builder.add_node("analyze_contradiction", workflow._analyze_contradiction_node)
    builder.add_node("merge_relationships", workflow._merge_relationships_node)
    builder.add_node("evaluate_relationships", workflow._evaluate_relationships_node)
    builder.add_node("present_structure", workflow._present_structure_node)
    builder.add_node("summarize_structure", workflow._summarize_structure_node)
    
    # Add edges - all relationship analysis nodes run in parallel
    builder.add_edge(START, "analyze_evidence")
    builder.add_edge(START, "analyze_background")
    builder.add_edge(START, "analyze_cause")
    builder.add_edge(START, "analyze_contrast")
    builder.add_edge(START, "analyze_condition")
    builder.add_edge(START, "analyze_contradiction")
    
    # All parallel nodes feed into merge
    builder.add_edge("analyze_evidence", "merge_relationships")
    builder.add_edge("analyze_background", "merge_relationships")
    builder.add_edge("analyze_cause", "merge_relationships")
    builder.add_edge("analyze_contrast", "merge_relationships")
    builder.add_edge("analyze_condition", "merge_relationships")
    builder.add_edge("analyze_contradiction", "merge_relationships")
    
    # Merge feeds into evaluation, then presentation
    builder.add_edge("merge_relationships", "evaluate_relationships")
    builder.add_edge("evaluate_relationships", "present_structure")
    builder.add_edge("present_structure", "summarize_structure")
    builder.add_edge("summarize_structure", END)
    
    return builder.compile(checkpointer=memory)


# Example usage
if __name__ == "__main__":
    # Initialize the workflow
    workflow = StructureWorkflow()
    
    # Example assertions (normally these would come from Idea Capture mode)
    sample_assertions = [
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
        )
    ]
    
    # Run the workflow
    result = workflow.run(sample_assertions)
    print("Final result:", result)
