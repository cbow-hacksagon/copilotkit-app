from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI


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



@tool
def calculate(expression: str) -> str:
    """
    Evaluates a mathematical expression and returns the result.
    Examples: '2 + 2', '10 * 5', '100 / 4', '2 ** 8'
    """
    try:
        # Only allow safe math operations
        allowed = set("0123456789+-*/().**% ")
        if not all(c in allowed for c in expression):
            return "Error: invalid characters in expression"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


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
                content="Successfully updated the counter",
                tool_call_id=runtime.tool_call_id
            )
        ]
    })


calculator_tools = [calculate, increment, check_counter]
