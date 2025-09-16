"""
Idea Capture Workflow using LangGraph

This module implements the first mode of the Clarus project - Idea Capture.
Users input raw, unstructured thoughts which are converted into discrete, atomic assertions.
"""

from typing import List, Dict, Any, Literal, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json

from models import Assertion, IdeaCaptureState


class IdeaCaptureWorkflow:
    """Main workflow class for Idea Capture."""
    
    def __init__(self, model_name: str = "gpt-4o-mini", config: dict = None):
        # Handle both direct instantiation and LangGraph Studio config
        if config is not None and isinstance(config, dict):
            # Extract model name from config if available
            model_name = config.get("model_name", "gpt-4o-mini")
        
        self.llm = ChatOpenAI(model=model_name, temperature=0.3)
        self.memory = MemorySaver()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        builder = StateGraph(IdeaCaptureState)
        
        # Add nodes
        builder.add_node("extract_assertions", self._extract_assertions_node)
        builder.add_node("present_assertions", self._present_assertions_node)
        builder.add_node("user_confirmation", self._user_confirmation_node)
        builder.add_node("route_decision", self._route_decision_node)
        builder.add_node("update_assertions", self._update_assertions_node)
        builder.add_node("summarize_chat", self._summarize_chat_node)
        
        # Add edges
        builder.add_edge(START, "extract_assertions")
        builder.add_edge("extract_assertions", "present_assertions")
        builder.add_edge("present_assertions", "user_confirmation")
        builder.add_edge("user_confirmation", "route_decision")
        builder.add_conditional_edges(
            "route_decision",
            self._route_after_confirmation,
            {
                "update": "update_assertions",
                "end": "summarize_chat"
            }
        )
        builder.add_edge("update_assertions", "extract_assertions")
        builder.add_edge("summarize_chat", END)
        
        return builder.compile(checkpointer=self.memory)
    
    def _extract_assertions_node(self, state: IdeaCaptureState) -> Dict[str, Any]:
        """Extract assertions from user input using LLM."""
        # Get input text from current_input or from the latest human message
        input_text = state.current_input
        if not input_text and state.messages:
            # Find the latest human message
            for message in reversed(state.messages):
                if isinstance(message, HumanMessage):
                    input_text = message.content
                    break
        
        if not input_text:
            return {}
        
        # Create prompt for assertion extraction
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at extracting discrete, atomic assertions from raw, unstructured text.

Your task is to:
1. Analyze the user's input text
2. Extract clear, specific assertions that are factual statements or claims
3. Each assertion should be atomic (one clear idea)
4. Provide confidence scores (0-1) for each assertion
5. Include the source text that led to each assertion

CRITICAL RULES:
- Only extract assertions that are factual statements or claims about the world
- Do NOT extract user instructions, commands, or requests (like "remove X", "add Y", "I want to...")
- Do NOT extract questions or uncertain statements
- Focus on statements that could be building blocks for a document

IMPORTANT: Return ONLY a valid JSON array. Do not include any other text.

Return your response as a JSON list of assertions with this structure:
[
    {{
        "id": "unique_id",
        "content": "the assertion text",
        "confidence": 0.8,
        "source": "excerpt from original text"
    }}
]

If the input contains no extractable assertions (only instructions/questions), return an empty array: []"""),
            ("human", "Extract assertions from this text: {input_text}")
        ])
        
        # Run LLM
        chain = prompt | self.llm
        response = chain.invoke({
            "input_text": input_text
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
            assertions_data = json.loads(content)
            
            # Ensure it's a list
            if not isinstance(assertions_data, list):
                assertions_data = [assertions_data]
            
            # Create assertions with proper IDs
            new_assertions = []
            for i, assertion_data in enumerate(assertions_data):
                if not isinstance(assertion_data, dict):
                    continue
                    
                # Ensure required fields exist
                assertion_data.setdefault("id", f"assertion_{len(state.assertions) + i + 1}")
                assertion_data.setdefault("confidence", 0.8)
                assertion_data.setdefault("source", input_text[:100] + "...")
                
                new_assertions.append(Assertion(**assertion_data))
            
            # Add to existing assertions
            all_assertions = state.assertions + new_assertions
            
            if len(new_assertions) > 0:
                return {
                    "assertions": all_assertions,
                    "messages": state.messages + [
                        AIMessage(content=f"I've extracted {len(new_assertions)} new assertions from your input.")
                    ]
                }
            else:
                # No new assertions extracted - this might be an instruction
                return {
                    "messages": state.messages + [
                        AIMessage(content="I didn't extract any new assertions from your input. This might be an instruction rather than new content to extract assertions from.")
                    ]
                }
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Return error message with the actual response for debugging
            return {
                "messages": state.messages + [
                    AIMessage(content=f"I had trouble parsing the assertions. Error: {str(e)}. Response was: {response.content[:200]}...")
                ]
            }
    
    def _present_assertions_node(self, state: IdeaCaptureState) -> Dict[str, Any]:
        """Present current assertions to the user."""
        if not state.assertions:
            return {
                "messages": state.messages + [
                    AIMessage(content="No assertions have been extracted yet.")
                ]
            }
        
        # Format assertions for display
        assertions_text = "\n".join([
            f"{i+1}. {assertion.content} (confidence: {assertion.confidence:.2f})"
            for i, assertion in enumerate(state.assertions)
        ])
        
        message = f"Here are the current assertions:\n\n{assertions_text}\n\nWould you like to modify any of these assertions or add new ones?"
        
        return {
            "messages": state.messages + [AIMessage(content=message)]
        }
    
    def _user_confirmation_node(self, state: IdeaCaptureState) -> Command:
        """Get user feedback on assertions in a conversational way."""
        # Format assertions for display
        assertions_text = "\n".join([
            f"{i+1}. {assertion.content} (confidence: {assertion.confidence:.2f})"
            for i, assertion in enumerate(state.assertions)
        ])
        
        user_response = interrupt({
            "type": "user_feedback",
            "message": f"Here are the current assertions:\n\n{assertions_text}\n\nPlease let me know what you'd like to do with these assertions. You can:\n- Accept them as they are\n- Ask me to remove specific ones\n- Add new assertions\n- Modify existing ones\n- Or just tell me what you think!\n\nWhat would you like to do?",
            "current_assertions": [a.model_dump() for a in state.assertions]
        })
        
        return Command(
            update={
                "messages": state.messages + [HumanMessage(content=user_response.get("response", ""))],
                "current_input": user_response.get("response", "")
            },
            goto="route_decision"
        )
    
    def _route_decision_node(self, state: IdeaCaptureState) -> Dict[str, Any]:
        """Route decision node - just passes through the state."""
        # This node doesn't modify state, just serves as a routing point
        return {}
    
    def _update_assertions_node(self, state: IdeaCaptureState) -> Dict[str, Any]:
        """Update assertions based on user feedback using LLM to understand intent."""
        user_input = state.current_input
        
        if not user_input:
            return {}
        
        
        # First, let the LLM understand what the user wants to do
        intent_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are analyzing user feedback about assertions to understand their intent.

Current assertions:
{current_assertions}

User feedback: {user_feedback}

CRITICAL RULES:
1. If the user says "add" or "no add" followed by content, they want to ADD new assertions
2. Only remove assertions if the user explicitly says "remove", "delete", "get rid of", "don't want", "take out"
3. Be conservative with removals - only remove if the intent is clearly to remove
4. Support both index-based removal (e.g., "remove 1, 3, 5") and content-based removal (e.g., "remove assertions about machine learning")

Analyze the user's intent and determine:
1. Do they want to accept all assertions as-is? (return "accept") - Look for: "yes", "good", "perfect", "keep them", "accept", "fine", "ok", "sounds good", "looks good", "that works", "i'm satisfied"
2. Do they want to remove specific assertions? (return "remove") - ONLY if they explicitly say to remove/delete
   - If they specify indices/numbers: use "remove_indices" with 1-based indices
   - If they specify content/keywords: use "remove_content" with keywords or phrases to match
3. Do they want to add new assertions? (return "add" and provide the new assertions) - Look for: "add", "no add", "also", "and", "plus", or when they provide new content
4. Do they want to modify existing assertions? (return "modify" and provide changes)
5. Do they want to continue the conversation? (return "continue")

EXAMPLES:
- "no add this assertion" → intent: "add", new_assertions: ["this assertion"]
- "remove the first one" → intent: "remove", remove_indices: [1]
- "remove assertions 1, 3, 5" → intent: "remove", remove_indices: [1, 3, 5]
- "remove assertions about machine learning" → intent: "remove", remove_content: ["machine learning"]
- "remove the one about AI bias" → intent: "remove", remove_content: ["AI bias", "bias"]
- "add nuclear power debate" → intent: "add", new_assertions: ["nuclear power debate"]
- "that's good" → intent: "accept"

IMPORTANT: Return ONLY a valid JSON object. Do not include any other text.

Return your analysis as JSON:
{{
    "intent": "accept|remove|add|modify|continue",
    "action": "description of what to do",
    "new_assertions": ["list of new assertions if adding"],
    "remove_indices": [list of 1-based indices to remove if removing by index],
    "remove_content": ["list of keywords/phrases to match for content-based removal"],
    "modifications": {{"index": "new_content"}} if modifying
}}"""),
            ("human", "Analyze this user feedback: {user_feedback}")
        ])
        
        current_assertions_text = "\n".join([
            f"{i+1}. {assertion.content}"
            for i, assertion in enumerate(state.assertions)
        ])
        
        chain = intent_prompt | self.llm
        response = chain.invoke({
            "current_assertions": current_assertions_text,
            "user_feedback": user_input
        })
        
        try:
            # Clean the response content
            content = response.content.strip()
            
            # Try to extract JSON from the response if it's wrapped in markdown
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            
            # Parse the intent analysis
            intent_data = json.loads(content)
            intent = intent_data.get("intent", "continue")
            
            if intent == "accept":
                return {
                    "messages": state.messages + [
                        AIMessage(content="Perfect! I'll keep all the assertions as they are. The session is now complete with your final set of assertions.")
                    ]
                }
            
            elif intent == "remove":
                # Handle both index-based and content-based removal
                remove_indices = intent_data.get("remove_indices", [])
                remove_content = intent_data.get("remove_content", [])
                
                valid_remove_indices = []
                
                # Process index-based removal
                if remove_indices:
                    for idx in remove_indices:
                        try:
                            # Convert to 0-based index
                            zero_based_idx = int(idx) - 1
                            if 0 <= zero_based_idx < len(state.assertions):
                                valid_remove_indices.append(zero_based_idx)
                        except (ValueError, TypeError):
                            continue
                
                # Process content-based removal
                if remove_content:
                    for i, assertion in enumerate(state.assertions):
                        assertion_lower = assertion.content.lower()
                        for keyword in remove_content:
                            if keyword.lower() in assertion_lower:
                                valid_remove_indices.append(i)
                                break  # Don't add the same assertion multiple times
                
                # Remove duplicates and sort in reverse order to avoid index shifting issues
                valid_remove_indices = sorted(list(set(valid_remove_indices)), reverse=True)
                
                # Create updated assertions list
                updated_assertions = state.assertions.copy()
                for idx in valid_remove_indices:
                    updated_assertions.pop(idx)
                
                removed_count = len(valid_remove_indices)
                if removed_count > 0:
                    # Safety check: Don't remove all assertions unless explicitly requested
                    if len(updated_assertions) == 0 and len(state.assertions) > 1:
                        return {
                            "messages": state.messages + [
                                AIMessage(content="I'm not sure you want to remove all assertions. Please be more specific about which assertions to remove, or say 'remove all' if you really want to clear everything.")
                            ]
                        }
                    
                    # Get the content of removed assertions for the message
                    removed_assertions = []
                    for idx in sorted(valid_remove_indices):
                        removed_assertions.append(state.assertions[idx].content)
                    
                    # Create simplified removal message
                    removal_message = f"Removed {removed_count} assertion(s):\n"
                    for i, content in enumerate(removed_assertions, 1):
                        removal_message += f"{i}. {content}\n"
                    removal_message += f"Remaining: {len(updated_assertions)} assertions"
                    
                    return {
                        "assertions": updated_assertions,
                        "messages": state.messages + [
                            AIMessage(content=removal_message)
                        ]
                    }
                else:
                    return {
                        "messages": state.messages + [
                            AIMessage(content="I couldn't identify which specific assertions you wanted to remove. Could you please specify the numbers of the assertions you'd like to remove or describe the content you want to remove?")
                        ]
                    }
            
            elif intent == "add":
                # Add new assertions
                new_assertions_text = intent_data.get("new_assertions", [])
                if new_assertions_text:
                    # Create new assertions
                    new_assertions = []
                    for i, content in enumerate(new_assertions_text):
                        new_assertions.append(Assertion(
                            id=f"assertion_{len(state.assertions) + i + 1}",
                            content=content,
                            confidence=0.8,
                            source=user_input[:100] + "..."
                        ))
                    
                    updated_assertions = state.assertions + new_assertions
                    
                    return {
                        "assertions": updated_assertions,
                        "messages": state.messages + [
                            AIMessage(content=f"I've added {len(new_assertions)} new assertions. You now have {len(updated_assertions)} total assertions.")
                        ]
                    }
            
            elif intent == "modify":
                # Modify existing assertions
                modifications = intent_data.get("modifications", {})
                updated_assertions = state.assertions.copy()
                
                for index_str, new_content in modifications.items():
                    try:
                        index = int(index_str) - 1  # Convert to 0-based index
                        if 0 <= index < len(updated_assertions):
                            updated_assertions[index].content = new_content
                    except (ValueError, IndexError):
                        continue
                
                return {
                    "assertions": updated_assertions,
                    "messages": state.messages + [
                        AIMessage(content=f"I've updated the assertions as requested. You now have {len(updated_assertions)} assertions.")
                    ]
                }
            
            else:  # continue or unknown intent
                return {
                    "messages": state.messages + [
                        AIMessage(content="I understand. Let me know what you'd like to do with the assertions, or if you have any other thoughts to add.")
                    ]
                }
                
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # If we can't parse the intent, just continue the conversation
            return {
                "messages": state.messages + [
                    AIMessage(content=f"I'm not sure I understood exactly what you'd like to do. Could you clarify what you want to change about the assertions? (Error: {str(e)})")
                ]
            }
    
    def _summarize_chat_node(self, state: IdeaCaptureState) -> Dict[str, Any]:
        """Summarize the chat conversation."""
        if not state.messages:
            return {"chat_summary": "No conversation to summarize."}
        
        # Create summary of the conversation
        conversation_text = "\n".join([
            f"{msg.__class__.__name__}: {msg.content}"
            for msg in state.messages
        ])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Summarize this conversation about idea capture and assertion extraction. Focus on the key points and outcomes."),
            ("human", "Conversation:\n{conversation}")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({"conversation": conversation_text})
        
        # Format final assertions list
        final_assertions_text = ""
        if state.assertions:
            final_assertions_text = "\n\nFinal Assertions:\n" + "\n".join([
                f"{i+1}. {assertion.content} (confidence: {assertion.confidence:.2f})"
                for i, assertion in enumerate(state.assertions)
            ])
        
        # Create final message with summary and assertions
        final_message = f"Session complete!\n\nSummary: {response.content}{final_assertions_text}"
        
        return {
            "chat_summary": response.content,
            "messages": state.messages + [
                AIMessage(content=final_message)
            ]
        }
    
    def _route_after_confirmation(self, state: IdeaCaptureState) -> Literal["update", "end"]:
        """Route after user confirmation - either update assertions or end session."""
        # Check if user wants to end the session
        if state.messages and isinstance(state.messages[-1], HumanMessage):
            last_message = state.messages[-1].content.lower()
            # Expanded list of completion indicators
            completion_words = [
                "done", "finish", "complete", "end", "that's all", "good", "perfect", "thanks", 
                "looks good", "that's perfect", "i'm done", "all set", "sounds good", "great",
                "yes", "yep", "ok", "okay", "sure", "fine", "acceptable", "good to go",
                "no more", "nothing else", "that's it", "i'm satisfied", "i'm happy",
                "accept", "keep them", "leave them", "don't change", "no changes", 
                "they're fine", "that works", "i'll keep them"
            ]
            if any(word in last_message for word in completion_words):
                return "end"
        
        # Check if the last AI message suggests ending
        if state.messages and isinstance(state.messages[-1], AIMessage):
            last_ai_message = state.messages[-1].content.lower()
            if any(phrase in last_ai_message for phrase in [
                "ready to finish", "satisfied with", "looks good", "perfect", "all set"
            ]) and len(state.messages) > 2:
                return "end"
        
        # Default to update if user wants changes
        return "update"
    
    def run(self, initial_input: str, thread_id: str = "default") -> Dict[str, Any]:
        """Run the idea capture workflow."""
        config = {"configurable": {"thread_id": thread_id}}
        
        initial_state = IdeaCaptureState(
            current_input=initial_input,
            messages=[HumanMessage(content=initial_input)]
        )
        
        result = self.graph.invoke(initial_state, config)
        return result


# Factory function for LangGraph Studio
def create_idea_capture_workflow(config: dict = None):
    """Factory function to create and return the compiled graph for LangGraph Studio."""
    workflow = IdeaCaptureWorkflow(config=config)
    return workflow.graph


# Alternative: Direct graph creation function
def create_idea_capture_graph(config: dict = None):
    """Create the idea capture graph directly for LangGraph Studio."""
    # Extract model name from config if available
    model_name = "gpt-4o-mini"
    if config and isinstance(config, dict):
        model_name = config.get("model_name", "gpt-4o-mini")
    
    # Create LLM
    llm = ChatOpenAI(model=model_name, temperature=0.3)
    
    # Create memory
    memory = MemorySaver()
    
    # Build graph
    builder = StateGraph(IdeaCaptureState)
    
    # Create workflow instance for node methods
    workflow = IdeaCaptureWorkflow(model_name=model_name)
    
    # Add nodes
    builder.add_node("extract_assertions", workflow._extract_assertions_node)
    builder.add_node("present_assertions", workflow._present_assertions_node)
    builder.add_node("user_confirmation", workflow._user_confirmation_node)
    builder.add_node("route_decision", workflow._route_decision_node)
    builder.add_node("update_assertions", workflow._update_assertions_node)
    builder.add_node("summarize_chat", workflow._summarize_chat_node)
    
    # Add edges
    builder.add_edge(START, "extract_assertions")
    builder.add_edge("extract_assertions", "present_assertions")
    builder.add_edge("present_assertions", "user_confirmation")
    builder.add_edge("user_confirmation", "route_decision")
    builder.add_conditional_edges(
        "route_decision",
        workflow._route_after_confirmation,
        {
            "update": "update_assertions",
            "end": "summarize_chat"
        }
    )
    builder.add_edge("update_assertions", "extract_assertions")
    builder.add_edge("summarize_chat", END)
    
    return builder.compile(checkpointer=memory)


# Example usage
if __name__ == "__main__":
    # Initialize the workflow
    workflow = IdeaCaptureWorkflow()
    
    # Example input
    sample_input = """
    I think we should focus on building a better user interface for our application. 
    The current design is confusing and users are having trouble finding the main features. 
    We need to prioritize mobile responsiveness and make sure the navigation is intuitive. 
    Also, I've been thinking about adding dark mode support since many users have requested it.
    """
    
    # Run the workflow
    result = workflow.run(sample_input)
    print("Final result:", result)
