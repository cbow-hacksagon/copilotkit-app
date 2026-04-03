"""
Main entry point for the agent.
Defines the workflow graph, state, tools, and nodes.
"""
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from src.calculator import calculate


#state list

from langgraph.prebuilt import InjectedState
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.graph import MessagesState
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from typing import TypedDict, Literal, Annotated
import uuid

from langchain.tools import ToolRuntime, tool

from langchain_core.tools import tool
from langgraph.types import Command
from typing import Annotated

class MyAgentState(MessagesState):
    counter: int
    remaining_steps: int




@tool
def check_counter(expression: str, runtime: ToolRuntime) -> str:
    """
    Checks the user's current counter value.
    """
    # Access the state directly from the injected Annotated type
    val = runtime.state.get("counter", 0) 
    return f"The current counter value is {val}."

@tool
def increment(number: int, runtime:ToolRuntime) -> Command:
    """
    Increments the counter by a specific number.
    """
    current_counter = runtime.state.get("counter", 0)
    new_val = current_counter + number
    
    # We return Command to update the state AND content to inform the LLM
    return Command(update={
        "counter": new_val,
        "messages": [
            ToolMessage(
                content="Successfully updated todos",
                tool_call_id=runtime.tool_call_id
            )
        ]
    })

llm = ChatOpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
    model="qwen",
    max_tokens=81920,
    temperature=1.0,
    top_p=0.95,
    presence_penalty=1.5,
    extra_body={
        "top_k": 20,
    },
)

agent = create_react_agent(
    model=llm,
    tools=[calculate,increment, check_counter],
    state_schema=MyAgentState,
    prompt="""
        You are a polished, professional assistant powered by Llama.
        Keep responses brief and polished — 1 to 2 sentences max.
        Be helpful, accurate, and conversational.
    """,
)

graph = agent
