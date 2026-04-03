"""
This is the main entry point for the agent.
It defines the workflow graph, state, tools, nodes and edges.
"""

from copilotkit import CopilotKitMiddleware
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_community.llms import LlamaCpp

from src.query import query_data
from src.todos import AgentState, todo_tools
from src.form import generate_form


llm = LlamaCpp(
    model_path="/home/joetheguide/Documents/dev/llama.cpp/models/Qwen3.5-9B-UD-Q4_K_XL.gguf",  # Path to your GGUF model file
    n_gpu_layers=10,  # Use -1 to offload all layers to GPU, or specify a number
    n_batch=512,
    n_ctx=32768,
    temperature=0.6,
    top_p=0.95,
)

agent = create_agent(
    model=llm,  # Use the Llama.cpp instance directly
    tools=[],  # No tools
    middleware=[CopilotKitMiddleware()],
    state_schema=AgentState,
    system_prompt="""
        You are a polished, professional assistant powered by Llama.
        Keep responses brief and polished — 1 to 2 sentences max.
        Be helpful, accurate, and conversational.
    """,
)
 
graph = agent

