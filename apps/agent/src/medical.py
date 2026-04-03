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
def store_medical_image(
    filename: str, image_data_url: str, description: str, runtime: ToolRuntime
) -> Command:
    """
    Store a medical image in the agent state for later analysis.
    This tool accepts the filename, base64 encoded image data URL, and a brief description of what the image shows.
    The image will be saved and can be retrieved later for analysis without immediate LLM processing.
    """
    try:
        existing_images = runtime.state.get("Imaging", [])

        image_record = {
            "id": str(uuid.uuid4()),
            "filename": filename,
            "data_url": image_data_url,
            "description": description,
            "timestamp": str(uuid.uuid4())[:8],
        }

        updated_images = existing_images + [image_record]

        return Command(
            update={
                "Imaging": updated_images,
                "messages": [
                    ToolMessage(
                        content=f"Successfully stored medical image '{filename}'. Total images stored: {len(updated_images)}",
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
def get_imaging(runtime: ToolRuntime) -> str:
    """
    Retrieve the list of stored medical images from the Imaging state.
    Returns a summary of all stored images with their filenames, descriptions, and IDs.
    Use this tool when you need to reference or analyze previously uploaded images.
    """
    images = runtime.state.get("Imaging", [])
    if not images:
        return "No images have been uploaded yet."

    result = "Stored images:\n"
    for img in images:
        desc = img.get("description", "No description")
        result += f"- ID: {img['id']}, File: {img['filename']}, Description: {desc}\n"
    return result


medical_tools = [
    check_summaries,
    generate_clinical_note,
    summarize_chat,
    calling_emergency_services,
    store_medical_image,
    get_imaging,
]
