"""Minimal agent_factory — keeps only logic needed for AI Brain POC.

Cloned from ai-skill-platform/src/agent_factory.py, removed:
- NutriGO / diet related tools
- RemoteA2aAgent (A2A sub-agent)
- preload_memory_tool
Kept:
- GoogleSearchToolCompat
- VertexRagTool
- full_context knowledge base injection
- model alias resolution
"""

import json
import logging
import mimetypes
from typing import TYPE_CHECKING

from google.adk.agents import Agent
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from vertexai.preview import rag

if TYPE_CHECKING:
    from google.adk.models import LlmRequest

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"

# BE model alias → actual Gemini model ID
MODEL_ALIAS = {
    "gemini-flash": "gemini-3-flash-preview",
    "gemini-flash-lite": "gemini-3.1-flash-lite-preview",
    "gemini-pro": "gemini-2.5-pro-preview-05-06",
}


def resolve_model(alias: str) -> str:
    """Resolve BE model alias to actual model ID."""
    return MODEL_ALIAS.get(alias, alias)


class GoogleSearchToolCompat(BaseTool):
    """Google Search tool, supports all Gemini models (ADK built-in only supports 1.x/2.x)."""

    def __init__(self):
        super().__init__(name="google_search", description="google_search")

    async def process_llm_request(
        self, *, tool_context: ToolContext, llm_request: "LlmRequest"
    ) -> None:
        llm_request.config = llm_request.config or types.GenerateContentConfig()
        llm_request.config.tools = llm_request.config.tools or []
        llm_request.config.tools.append(
            types.Tool(google_search=types.GoogleSearch())
        )


class VertexRagTool(BaseTool):
    """ADK tool: Vertex AI RAG Engine knowledge base retrieval using built-in retrieval."""

    def __init__(
        self,
        rag_resource_name: str,
        similarity_top_k: int = 5,
        vector_distance_threshold: float = 0.6,
    ):
        super().__init__(
            name="rag_retrieval",
            description="Retrieve relevant nutrition and diet data from knowledge base",
        )
        self.rag_resources = [rag.RagResource(rag_corpus=rag_resource_name)]
        self.vertex_rag_store = types.VertexRagStore(
            rag_resources=[
                types.VertexRagStoreRagResource(rag_corpus=rag_resource_name)
            ],
            similarity_top_k=similarity_top_k,
            vector_distance_threshold=vector_distance_threshold,
        )
        self.similarity_top_k = similarity_top_k
        self.vector_distance_threshold = vector_distance_threshold

    async def process_llm_request(
        self, *, tool_context: ToolContext, llm_request: "LlmRequest"
    ) -> None:
        llm_request.config = llm_request.config or types.GenerateContentConfig()
        llm_request.config.tools = llm_request.config.tools or []
        llm_request.config.tools.append(
            types.Tool(
                retrieval=types.Retrieval(
                    vertex_rag_store=self.vertex_rag_store
                )
            )
        )


def _guess_mime_type(file_name: str) -> str:
    """Infer MIME type from file extension, default application/pdf."""
    mime, _ = mimetypes.guess_type(file_name)
    if not mime:
        return "application/pdf"
    if mime == "application/pdf":
        return mime
    return "text/plain"


def _build_knowledge_parts(rag_files: list[dict]) -> list[types.Part]:
    """full_context mode: convert rag_files file URLs into file_data Parts."""
    parts = []
    for f in rag_files:
        url = f.get("url") or f.get("gcs_uri", "")
        name = f.get("file_name", "")
        if not url:
            continue
        if url.startswith("gs://") and f.get("url"):
            url = f["url"]
        mime = _guess_mime_type(name)
        parts.append(
            types.Part(
                file_data=types.FileData(
                    mime_type=mime,
                    file_uri=url,
                )
            )
        )
        logger.info(f"Knowledge file: {name} (mime={mime}, {url[:60]}...)")
    return parts


def create_agent(
    config: dict,
    context_data: dict,
    skill_key: str,
) -> tuple[Agent, list[types.Part]]:
    """Dynamically create an ADK Agent from BE config and context_data.

    Returns:
        (agent, knowledge_parts): agent and knowledge base Parts for full_context mode.
    """
    system_prompt = config.get("system_prompt", "")
    context_json = json.dumps(context_data, ensure_ascii=False, indent=2)
    instruction_text = f"{system_prompt}\n\n## Reference Data\n{context_json}"
    # Use callable to bypass ADK template variable injection
    instruction = lambda _ctx, _t=instruction_text: _t

    # Model
    model_config = config.get("model_config") or {}
    raw_model = model_config.get("model") or config.get("model") or DEFAULT_MODEL
    model = resolve_model(raw_model)

    # Tools
    BUILTIN_TOOL_NAMES = {"google_search"}
    all_tool_names = config.get("tools") or []
    tools = []

    # Knowledge base
    rag_resource_name = config.get("rag_resource_name")
    rag_files = config.get("rag_files") or []
    knowledge_parts: list[types.Part] = []

    full_context_files = [f for f in rag_files if f.get("rag_mode") == "full_context"]
    if full_context_files:
        knowledge_parts = _build_knowledge_parts(full_context_files)

    if rag_resource_name and rag_files:
        tools.append(VertexRagTool(rag_resource_name))

    generate_content_config = types.GenerateContentConfig()

    # Gemini native tools
    if "google_search" in all_tool_names:
        tools.append(GoogleSearchToolCompat())

    agent = Agent(
        name=f"brain_{skill_key.replace('-', '_')}",
        model=model,
        instruction=instruction,
        tools=tools,
        generate_content_config=generate_content_config,
    )

    logger.info(
        f"Created agent: skill_key={skill_key}, model={model}, "
        f"full_context_files={len(knowledge_parts)}, tools={len(tools)}"
    )
    return agent, knowledge_parts


def create_orchestrator(
    system_prompt: str,
    model_config: dict,
    sub_agents: list[Agent],
    skill_key: str,
) -> Agent:
    """Create an orchestrator Agent with sub_agents for routing.

    Args:
        system_prompt: orchestrator system prompt (routing only, not analysis)
        model_config: orchestrator model config (typically gemini-pro)
        sub_agents: list of already-created skill Agents
        skill_key: top-level skill key (used for naming)

    Returns:
        orchestrator Agent
    """
    if not sub_agents:
        raise ValueError("create_orchestrator requires at least one sub_agent")

    raw_model = model_config.get("model", DEFAULT_MODEL)
    model = resolve_model(raw_model)

    instruction = lambda _ctx, _t=(system_prompt or ""): _t

    agent = Agent(
        name=f"orchestrator_{skill_key.replace('-', '_')}",
        model=model,
        instruction=instruction,
        sub_agents=sub_agents,
        generate_content_config=types.GenerateContentConfig(),
    )

    logger.info(
        f"Created orchestrator: skill_key={skill_key}, model={model}, "
        f"sub_agents={[a.name for a in sub_agents]}"
    )
    return agent
