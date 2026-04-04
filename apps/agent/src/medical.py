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


from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
import base64

# Initialize the remote LLaMA client
# Replace with the actual local IP of the device running the server
medmo_agent = ChatOpenAI(
    base_url="http://10.221.180.237:8080/v1",  # ← replace with actual IP
    api_key="not-needed",
    model="llama",  # ← replace with actual model name
    max_tokens=4096,
    temperature=0.7,
)


@tool
def query_diagnostic_specialist(prompt: str, runtime: ToolRuntime) -> str:
    """
    Send a prompt or query to a diagnostic specialist, put your question in the prompt.
    """
    case_summary = runtime.state.get(
        "clinical_note", runtime.state.get("chat_summary", "")
    )
    response = medmo_agent.invoke([HumanMessage(content=prompt + case_summary)])
    return response.content


@tool
def query_imaging_specialist(prompt: str, img_id: int, runtime: ToolRuntime) -> Command:
    """
    Whenever you are asked to analyse an image call this tool , your prompt does not matter but the img_id should have the id of the image in integer form. And report the exact response first then your assesments. This specialist can also be referred to as MedMo model.

    """
    images = runtime.state.get("Imaging", [])
    current_summary = runtime.state.get("image_summary", "")
    image_base64 = ""
    mime_type = "image/png"
    for i in images:
        if i["id"] == img_id:
            image_base64 = i["base64"]
            mime_type = i.get("mimeType", "image/png")
            print(image_base64)
            break

    if not image_base64:
        return f"Error: No image found with ID {img_id}. Available images: {[img.get('id') for img in images]}"

    message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_base64}"},
            },
            {"type": "text", "text": prompt},
        ]
    )
    response = medmo_agent.invoke([message])
    return Command(
        update={
            "image_summary": current_summary+response.content,
            "messages": [
                ToolMessage(
                    content=f"Successfully appended to the image summary state, result of the image analysis {response.content}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


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
def check_image_summary(expression: str, runtime: ToolRuntime) -> str:
    """
    Check the user's current image summary state which includes all of the images consulted with the medmo agent before this. Put anything as the expression the result will be a complete summary.
    """

    images = runtime.state.get("image_summary", "")

    return images



@tool
def check_initial_diagnosis(expression: str, runtime: ToolRuntime) -> str:
    """
    Check the user's initial diagnosis given after a primary assessment.
    """

    images = runtime.state.get("diagnosis_1", "")

    return images


@tool
def generate_initial_diagnosis(expression: str, runtime: ToolRuntime) -> Command:
    """
    After enough information has beem retrieved you can generate an initial diagnosis of the patient, this tool should only be called once and not changed later.
    """

    new_summary = expression

    return Command(
        update={
            "diagnosis_1": new_summary,
            "messages": [
                ToolMessage(
                    content=f"Successfully updated the contents to {new_summary}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
def calling_emergency_services(emergency: str) -> str:
    """
    Use this tool if the user shows signs of any immediate life threatening conditions such as: a stroke, heart attack, suicidal tendencies, self harm, etc. Pass the most immediate threat as the emergency parameter. End any conversation after this, just generate a clinical note and summary using the tools, do not talk any further just inform the user that help is on the way and display the exact summaries to help the medical team.
    """

    return "Successfully called emergency services."


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


medical_tools = [
    query_diagnostic_specialist,
    query_imaging_specialist,
    check_summaries,
    generate_clinical_note,
    summarize_chat,
    calling_emergency_services,
    check_images,
    store_medical_image,
    check_image_summary,
    check_initial_diagnosis,
    generate_initial_diagnosis,
]
