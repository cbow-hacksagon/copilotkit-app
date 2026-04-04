"""
Main entry point for the agent.
Defines the workflow graph, state, tools, and nodes.
"""

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from src.calculator import calculate


# state list

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
from typing import TypedDict, List, Optional
from src.medical import medical_tools


class AgentImage(TypedDict, total=False):
    id: int
    base64: str
    mimeType: str
    description: str


class MyAgentState(MessagesState):
    chat_summary: str
    clinical_note: str
    image_summary: str
    Imaging: list[dict]  # List of {id: int, base64: str, description: str}
    diagnosis_1: str
    diagnosis_2: str
    final_diagnosis: str
    remaining_steps: int
    images: List[AgentImage]


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
    tools=[*medical_tools],
    state_schema=MyAgentState,
    prompt="""
    Your goal is to collect a complete primary physical assesment of a patient. Ask the patient for their primary complaint first. Then, through a multi-turn conversation, gather relevant history: onset, duration, severity, associated symptoms, aggravating/relieving factors, and past medical history. When you have sufficient information or a key discovery, use the appropriate tool to generate a chat summary and a structured clinical note. Do not interrupt the patient; ask one focused question at a time. End the conversation only after documentation is complete. Make sure to check for signs of dangerous diseases which require immediate medical care.
    Tool usage instructions:
    1. Call generate chat summary and generate clinical note only in an emergency or when you think the chat is complete and you roughly have all the information you need. Otherwise call every 6 to 8 turns. Before generating the final clinical note, make sure to check images and image summary.
    2. Whenever imaging is mentioned, call check_images and check_image_summary tools to see the imaging studies until now and then do whatever is asked. If imaging is not mentioned but you think an image is desirable for diagnostic purpose , ask the user to attach images if available, and before generating the final summary at conversation end, check one more time for images and analyse them.
    3. After the conversation is complete and final summary and clinical note is generated with all of the imaging information as well, you can use diagnosis_1 state variable and generate and check_initial_diagnosis tool to update and tell the user their first diagnosis and differential if applicable.
    
    """,
)

graph = agent
