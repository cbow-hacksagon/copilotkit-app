from langgraph.prebuilt import InjectedState
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.graph import MessagesState
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from typing import TypedDict, Literal, Annotated
import uuid

class Todo(TypedDict):
    id: str
    title: str
    description: str
    emoji: str
    status: Literal["pending", "completed"]

class AgentState(MessagesState):
    todos: list[Todo]
    remaining_steps: int
@tool
def manage_todos(
    todos: list[Todo],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Manage the current todos."""
    for todo in todos:
        if "id" not in todo or not todo["id"]:
            todo["id"] = str(uuid.uuid4())

    return Command(update={
        "todos": todos,
        "messages": [
            ToolMessage(
                content="Successfully updated todos",
                tool_call_id=tool_call_id,
            )
        ]
    })

@tool
def get_todos(state: Annotated[AgentState, InjectedState]) -> list[Todo]:
    """Get the current todos."""
    return state.get("todos", [])

todo_tools = [manage_todos, get_todos]
