from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI


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


@tool
def summarize_chat(expression: str, runtime: ToolRuntime) -> Command:
    """
    Use this tool to store the summary of the users chat inside the agent state variable, invoke this tool with the chat summary as the expression. Generate a detailed summary with user complaints and your questions and interpretations.
    """

    summary = runtime.state.get("chat_summary", "")
    new_summary = expression

    return Command(
        update={
            "chat_summary": new_summary,
            "messages": [
                ToolMessage(
                    content="Successfully updated the contents",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
def generate_clinical_note(expression: str, runtime: ToolRuntime) -> Command:
    """
    Generate a clinical note based on the user's chat summary and chat history, the note should be well structured with primary complaints, differential, assessment and concerns or tests if any required. Put the clinical note as the expression.
    """

    summary = runtime.state.get("clinical_note", "")
    new_summary = expression

    return Command(
        update={
            "clinical_note": new_summary,
            "messages": [
                ToolMessage(
                    content="Successfully updated the contents",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
def check_summaries(expression: str, runtime: ToolRuntime) -> str:
    """
    Checks the user's current chat summary and clinical note. You can input any expression, it does not matter, the tool will return the summaries anyway, if you check with this tool prioritize giving the exact string as returned from the tool.
    """
    # Access the state directly from the injected Annotated type
    cl = runtime.state.get("clinical_note", "")
    ch = runtime.state.get("chat_summary", "")
    return f"The current counter chat summary is {ch} and clinical note is {cl}."


@tool
def calling_emergency_services(emergency: str) -> str:
    """
    Use this tool if the user shows signs of any immediate life threatening conditions such as: a stroke, heart attack, suicidal tendencies, self harm, etc. Pass the most immediate threat as the emergency parameter. End any conversation after this, just generate a clinical note and summary using the tools, do not talk any further just inform the user that help is on the way and display the exact summaries to help the medical team.
    """

    return "Successfully called emergency services."

@tool
def check_images(expression: str, runtime: ToolRuntime) -> str:
    """
    Checks the images currently stored in agent state. Input any expression, it does not matter.
    Returns the ID, description, and a truncated base64 preview for each image.
    """
    images = runtime.state.get("images", [])
    
    if not images:
        return "No images currently in agent state."
    
    result = f"There are {len(images)} image(s) in agent state:\n\n"
    
    for img in images:
        img_id = img.get("id", "?")
        description = img.get("description", "")
        base64 = img.get("base64", "")
        preview = base64[:60] + "..." if len(base64) > 60 else base64
        result += (
            f"Image ID: {img_id}\n"
            f"Description: {description}\n"
            f"Base64 preview (first 60 chars): {preview}\n"
            f"Total base64 length: {len(base64)} chars\n\n"
        )
    
    return result.strip()

@tool
def store_medical_image(
    image_id: int, base64: str, description: str, runtime: ToolRuntime
) -> Command:
    """
    Store a medical image in the agent Imaging state for later analysis.
    Accepts a numeric ID, raw base64 string (no data URL prefix), and a description.
    The image will be saved and can be retrieved later via check_images tool.
    """
    try:
        existing_images = runtime.state.get("Imaging", [])

        image_record = {
            "id": image_id,
            "base64": base64,
            "description": description,
        }

        updated_images = existing_images + [image_record]

        return Command(
            update={
                "Imaging": updated_images,
                "messages": [
                    ToolMessage(
                        content=f"Successfully stored medical image (ID: {image_id}). Total images stored: {len(updated_images)}",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )
    except Exception as e:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Error storing image: {str(e)}",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )


@tool
def check_images(expression: str, runtime: ToolRuntime) -> str:
    """
    Checks the images currently stored in agent Imaging state. Input any expression, it does not matter.
    Returns the ID, description, and a truncated base64 preview for each image.
    """
    images = runtime.state.get("Imaging", [])

    if not images:
        return "No images currently in agent state."

    result = f"There are {len(images)} image(s) in agent state:\n\n"

    for img in images:
        img_id = img.get("id", "?")
        description = img.get("description", "")
        base64 = img.get("base64", "")
        preview = base64[:60] + "..." if len(base64) > 60 else base64
        result += (
            f"Image ID: {img_id}\n"
            f"Description: {description}\n"
            f"Base64 preview (first 60 chars): {preview}\n"
            f"Total base64 length: {len(base64)} chars\n\n"
        )

    return result.strip()


medical_tools = [
    check_summaries,
    generate_clinical_note,
    summarize_chat,
    calling_emergency_services,
    check_images,
    store_medical_image,
]
medical_tools = [check_summaries, generate_clinical_note, summarize_chat, calling_emergency_services, check_images]
