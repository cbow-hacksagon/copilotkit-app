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
import os
import base64

# Initialize the remote LLaMA client
medmo_agent = ChatOpenAI(
    base_url=os.environ.get("MEDMO_BASE_URL", "http://localhost:8080/v1"),
    api_key="not-needed",
    model="llama",
    max_tokens=4096,
    temperature=0.7,
)


@tool
def query_diagnostic_specialist(prompt: str, runtime: ToolRuntime) -> str:
    """
    Consult the MedMo diagnostic specialist model for a second-opinion clinical
    assessment. Pass a focused clinical question or case summary in the prompt.
    The specialist will append the current clinical note and chat summary as
    context before responding. Use this when you need differential diagnosis
    input or want to validate your clinical reasoning.
    """
    case_summary = runtime.state.get(
        "clinical_note", runtime.state.get("chat_summary", "")
    )
    response = medmo_agent.invoke([HumanMessage(content=prompt + case_summary)])
    return response.content


@tool
def query_imaging_specialist(prompt: str, img_id: int, runtime: ToolRuntime) -> Command:
    """
    Analyze a medical image using the MedMo vision specialist model. Provide the
    exact integer img_id of the image to analyze. The specialist returns a
    structured radiological description including: modality, anatomical category,
    findings with confidence tags (H/M/L), urgency classification, and a visual
    impression. Results are automatically appended to the image_summary state.
    Always call check_images first to see available image IDs. Report the
    specialist's exact response to the patient, then provide your clinical
    interpretation.
    """
    images = runtime.state.get("Imaging", [])
    prompt1 = """## ROLE
Medical imaging visual description sub-agent. Describe only what is visible.
No diagnosis. No speculation. No text outside the template below.
If input is not a medical image: "INPUT ERROR." Stop.

## OPTIONAL CONTEXT
[SYMPTOMS] and [IMAGING TYPE/REGION] may or may not be provided.
If present: use only to orient description. If absent: infer from image.
Never fabricate findings to fill missing context.

## RULES
- Circular cross-section = CT/MRI slice. Never use PA/AP for CT/MRI.
- No lead marker → assume radiological convention (Patient-R = Image-L). State this.
- Never write "Normal" or "Unremarkable." Unseen structure = "NV" (Not Visualized).
- Tag every finding: (H) (M) or (L) confidence.
- Paired structures (lungs, joints, hemispheres): note L/R asymmetry if present.

## MODALITY TERMS (use only these)
X-ray: Opacity, Lucency, Air bronchogram, Silhouette sign
CT: Hyperdense, Hypodense, Attenuation, Calcification
MRI: T1/T2 signal, Diffusion restriction
US: Anechoic, Hyperechoic, Shadowing

## URGENCY (one line max)
CRITICAL: Tension PTX, large ICH, epiglottitis, aortic dissection, free perforation
URGENT: Significant but not immediately fatal
ROUTINE / NONE: all else

─────────────────────────────────────────
OUTPUT TEMPLATE — fill only, no extra text
─────────────────────────────────────────
MOD: [modality] | VIEW: [view] | QUAL: [Diagnostic/Sub-optimal/Non-diagnostic]
CAT: [A-Chest / B-Abdomen / C-Head / D-MSK / E-Neck / F-Other]
CTX: [symptoms/type if given, else "None"]

FINDINGS:
[structure]: [description] ([H/M/L])
[structure]: NV
[paired]: L=[desc] R=[desc]

PERIMETER: [finding | Clear]
URGENCY: [CRITICAL/URGENT/ROUTINE/NONE] — [finding if C/U]
IMPRESSION: [2 sentences. Visual pattern only. No diagnosis.]
─────────────────────────────────────────
AI visual description only. Not a report. Clinical review required."""

    current_summary = runtime.state.get("image_summary", "")
    image_base64 = ""

    mime_type = "image/png"
    for i in images:
        if i["id"] == img_id:
            image_base64 = i["base64"]
            prompt = (
                prompt1
                + f"Clinical context: {prompt}"
                + f"Image description {i['description']}"
            )
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
            "image_summary": current_summary + response.content,
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
    Store a comprehensive chat summary in the agent state. The summary should
    include: the patient's primary complaint, key history points (onset, duration,
    severity, associated symptoms, aggravating/relieving factors), relevant past
    medical history, your clinical observations and interpretations, and any
    notable findings or red flags. This summary serves as context for downstream
    specialist consultations and documentation. Pass the summary text as the expression.
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
    Store a structured clinical note in the agent state. The note should follow
    standard clinical format: Chief Complaint, History of Present Illness,
    relevant Past Medical History, Assessment/Impression, key Concerns, and
    Recommended investigations or follow-up. Incorporate any available imaging
    findings from image_summary. This note serves as the formal clinical record
    and context for specialist consultations. Pass the clinical note text as the expression.
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
    Retrieve the current chat_summary and clinical_note from agent state. Use
    this to review what has been documented so far before generating updated
    summaries, clinical notes, or diagnoses.
    """
    # Access the state directly from the injected Annotated type
    cl = runtime.state.get("clinical_note", "")
    ch = runtime.state.get("chat_summary", "")
    return f"The current counter chat summary is {ch} and clinical note is {cl}."


@tool
def check_image_summary(expression: str, runtime: ToolRuntime) -> str:
    """
    Retrieve the accumulated image analysis results from all previously consulted
    imaging studies. Use this to review imaging findings before generating
    clinical notes, summaries, or diagnoses.
    """

    images = runtime.state.get("image_summary", "")

    return images


@tool
def check_initial_diagnosis(expression: str, runtime: ToolRuntime) -> str:
    """
    Retrieve the current initial diagnosis (diagnosis_1) from agent state. Use
    this to review the primary diagnostic impression before proceeding to
    specialist consultations or final diagnosis synthesis.
    """

    images = runtime.state.get("diagnosis_1", "")

    return images


@tool
def generate_initial_diagnosis(expression: str, runtime: ToolRuntime) -> Command:
    """
    Record the initial primary diagnosis after completing patient intake,
    documentation, and imaging review. This should be called ONCE when you have
    sufficient clinical information. The diagnosis should include: your primary
    diagnostic impression, relevant differential diagnoses with reasoning, and
    key supporting clinical findings. This forms the foundation for subsequent
    specialist consultations and final diagnosis synthesis. Pass the diagnosis
    text as the expression.
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
    Activate emergency protocol when the patient describes signs of an immediate
    life-threatening condition: stroke symptoms (FAST), acute coronary syndrome,
    suicidal ideation, self-harm, anaphylaxis, severe trauma, or other critical
    emergencies. Pass the specific threat description as the emergency parameter.
    After calling this tool, generate final documentation, inform the patient
    that emergency help is being arranged, and do not continue the conversation.
    """

    return "Successfully called emergency services."


@tool
def check_images(expression: str, runtime: ToolRuntime) -> str:
    """
    List all medical images currently stored in agent state, including their ID,
    description, and a base64 preview. Use this to see what images are available
    before calling query_imaging_specialist. The image ID (integer) is required
    for image analysis.
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
    Requires: image_id (integer), base64-encoded image data (raw, without data
    URL prefix), and a brief description. The image will be available for
    subsequent analysis via query_imaging_specialist. Note: Images are typically
    uploaded by the patient through the frontend UI -- this tool is called by
    the frontend, not directly by you.
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
