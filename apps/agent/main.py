"""
Main entry point for the agent.
Defines the workflow graph, state, tools, and nodes.
"""
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from src.calculator import calculate


#state list

from langgraph.prebuilt import AgentState, InjectedState
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.graph import MessagesState
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from typing import TypedDict, Literal, Annotated
import uuid


class MyAgentState(AgentState):
    counter: int
    remaining_steps: int


        



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
    tools=[calculate,],
    state_schema=AgentState,
    prompt="""
        You are a polished, professional assistant powered by Llama.
        Keep responses brief and polished — 1 to 2 sentences max.
        Be helpful, accurate, and conversational.
    """,
)

graph = agent
