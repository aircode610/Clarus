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
from pydantic import BaseModel, Field
import json


class Assertion(BaseModel):
    """Represents a discrete, atomic assertion extracted from user input."""
    id: str = Field(description="Unique identifier for the assertion")
    content: str = Field(description="The actual assertion text")
    confidence: float = Field(description="Confidence score (0-1) for this assertion")
    source: str = Field(description="Source text that led to this assertion")


class IdeaCaptureState(BaseModel):
    """State for the Idea Capture workflow."""
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
    assertions: List[Assertion] = Field(default_factory=list)
    current_input: str = Field(default="")
    iteration_count: int = Field(default=0)
    chat_summary: str = Field(default="")


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
        builder.add_node("update_assertions", self._update_assertions_node)
        builder.add_node("summarize_chat", self._summarize_chat_node)
        
        # Add edges
        builder.add_edge(START, "extract_assertions")
        builder.add_edge("extract_assertions", "present_assertions")
        builder.add_edge("present_assertions", "user_confirmation")
        builder.add_conditional_edges(
            "user_confirmation",
            self._should_continue,
            {
                "continue": "update_assertions",
                "end": "summarize_chat"
            }
        )
        builder.add_edge("update_assertions", "extract_assertions")
        builder.add_edge("summarize_chat", END)
        
        return builder.compile(checkpointer=self.memory)
    
    def _extract_assertions_node(self, state: IdeaCaptureState) -> Dict[str, Any]:
        """Extract assertions from user input using LLM."""
        if not state.current_input:
            return {}
        
        # Create prompt for assertion extraction
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at extracting discrete, atomic assertions from raw, unstructured text.

Your task is to:
1. Analyze the user's input text
2. Extract clear, specific assertions
3. Each assertion should be atomic (one clear idea)
4. Provide confidence scores (0-1) for each assertion
5. Include the source text that led to each assertion

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

Focus on extracting meaningful, actionable assertions that could be building blocks for a document."""),
            ("human", "Extract assertions from this text: {input_text}")
        ])
        
        # Run LLM
        chain = prompt | self.llm
        response = chain.invoke({
            "input_text": state.current_input
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
                assertion_data.setdefault("source", state.current_input[:100] + "...")
                
                new_assertions.append(Assertion(**assertion_data))
            
            # Add to existing assertions
            all_assertions = state.assertions + new_assertions
            
            return {
                "assertions": all_assertions,
                "messages": state.messages + [
                    AIMessage(content=f"I've extracted {len(new_assertions)} new assertions from your input.")
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
        """Get user confirmation and modifications."""
        user_response = interrupt({
            "type": "user_confirmation",
            "message": "Please review the assertions above. You can:",
            "options": [
                "Accept all assertions as-is",
                "Remove specific assertions (provide numbers)",
                "Add new assertions (provide text)",
                "Modify existing assertions (provide number and new text)"
            ],
            "current_assertions": [a.dict() for a in state.assertions]
        })
        
        return Command(
            update={
                "messages": state.messages + [HumanMessage(content=user_response.get("response", ""))],
                "current_input": user_response.get("response", "")
            },
            goto="update_assertions" if user_response.get("action") != "accept" else "summarize_chat"
        )
    
    def _update_assertions_node(self, state: IdeaCaptureState) -> Dict[str, Any]:
        """Update assertions based on user feedback."""
        user_input = state.current_input
        
        if not user_input:
            return {}
        
        # Parse user modifications
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are helping to update assertions based on user feedback.

The user may want to:
1. Remove assertions (they'll mention numbers)
2. Add new assertions (they'll provide new text)
3. Modify existing assertions (they'll provide number and new text)

Current assertions:
{current_assertions}

User feedback: {user_feedback}

Return updated assertions as JSON list with the same structure as before.
Only include assertions that should remain after the user's modifications."""),
            ("human", "Update the assertions based on this feedback: {user_feedback}")
        ])
        
        current_assertions_text = "\n".join([
            f"{i+1}. {assertion.content}"
            for i, assertion in enumerate(state.assertions)
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({
            "current_assertions": current_assertions_text,
            "user_feedback": user_input
        })
        
        try:
            assertions_data = json.loads(response.content)
            updated_assertions = [Assertion(**assertion) for assertion in assertions_data]
            
            return {
                "assertions": updated_assertions,
                "messages": state.messages + [
                    AIMessage(content=f"Assertions updated. You now have {len(updated_assertions)} assertions.")
                ]
            }
        except json.JSONDecodeError:
            return {
                "messages": state.messages + [
                    AIMessage(content="I had trouble updating the assertions. Please try again.")
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
        
        return {
            "chat_summary": response.content,
            "messages": state.messages + [
                AIMessage(content=f"Session complete! Summary: {response.content}")
            ]
        }
    
    def _should_continue(self, state: IdeaCaptureState) -> Literal["continue", "end"]:
        """Determine if the workflow should continue or end."""
        # Check if user wants to continue
        if state.messages and isinstance(state.messages[-1], HumanMessage):
            last_message = state.messages[-1].content.lower()
            if any(word in last_message for word in ["done", "finish", "complete", "end"]):
                return "end"
        
        return "continue"
    
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
    builder.add_node("update_assertions", workflow._update_assertions_node)
    builder.add_node("summarize_chat", workflow._summarize_chat_node)
    
    # Add edges
    builder.add_edge(START, "extract_assertions")
    builder.add_edge("extract_assertions", "present_assertions")
    builder.add_edge("present_assertions", "user_confirmation")
    builder.add_conditional_edges(
        "user_confirmation",
        workflow._should_continue,
        {
            "continue": "update_assertions",
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
