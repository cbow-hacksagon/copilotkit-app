"""
Main entry point for the agent.
Defines the workflow graph, state, tools, and nodes.
"""
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from src.todos import AgentState, todo_tools

llm = ChatOpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
    model="qwen",
    temperature=0.6,
    max_tokens=1024,
)

graph = create_react_agent(
    model=llm,
    tools=todo_tools,
    state_schema=AgentState,
    prompt="""
        You are a polished, professional assistant powered by Llama.
        Keep responses brief and polished — 1 to 2 sentences max.
        Be helpful, accurate, and conversational.
    """,
)
