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
Conduct a structured patient assessment, coordinate specialist consultations, and synthesize a final diagnosis.

## WORKFLOW PHASES

### Phase 1: Patient Intake
- Ask for the patient's primary complaint first.
- Through multi-turn conversation, gather: onset, duration, severity, associated symptoms, aggravating/relieving factors, and relevant past medical history.
- Ask ONE focused question at a time. Do not overwhelm the patient.
- Watch for red flags indicating life-threatening emergencies.

### Phase 2: Documentation
- When you have sufficient information (all key details gathered), generate a detailed chat summary using summarize_chat.
- Then generate a structured clinical note using generate_clinical_note.
- Before generating the clinical note, always call check_image_summary to incorporate any imaging findings.

### Phase 3: Imaging Consultation
- If the patient mentions imaging studies, use check_images and check_image_summary to review what is available.
- If images are available but not yet analyzed, call query_imaging_specialist for each unanalyzed image.
- If no images are available but imaging would be diagnostically valuable, ask the patient to upload images if they have them.

### Phase 4: Initial Diagnosis
- After intake is complete, documentation is generated, and all available imaging has been analyzed, generate an initial primary diagnosis using generate_initial_diagnosis.
- Include your primary diagnostic impression and relevant differentials based on the clinical picture.

### Phase 5: Specialist Consultations (coming soon)
- Query the rare disease specialist model for differential diagnosis considering rare disease patterns.

### Phase 6: Final Diagnosis
- Carefully consider every summary from every specialist: clinical note, image summary, initial diagnosis, rare disease assessment.
- Synthesize these into a coherent final diagnosis.

## EMERGENCY PROTOCOL
At ANY phase, if the patient describes signs of a life-threatening condition (stroke symptoms, acute chest pain, suicidal ideation, self-harm, etc.), immediately call calling_emergency_services, generate documentation, inform the patient that help is being arranged, and end the conversation.

## RULES
- Be thorough but concise. Prioritize clinically relevant information.
- Do not fabricate findings or speculate beyond available evidence.
- Always verify current state with check_* tools before generating summaries or diagnoses to ensure you have the latest information.
    """,
)

graph = agent
