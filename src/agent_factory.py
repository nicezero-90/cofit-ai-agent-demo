"""精簡版 agent_factory — 僅保留 AI Brain POC 所需邏輯。

從 ai-skill-platform/src/agent_factory.py clone，移除：
- NutriGO / diet 相關 tool
- RemoteA2aAgent (A2A sub-agent)
- preload_memory_tool
保留：
- GoogleSearchToolCompat
- VertexRagTool
- full_context 知識庫注入
- model alias 解析
"""

import json
import logging
import mimetypes
from typing import Any, TYPE_CHECKING

from google.adk.agents import Agent
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from vertexai.preview import rag

if TYPE_CHECKING:
    from google.adk.models import LlmRequest

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"

# BE model alias → 實際 Gemini model ID
MODEL_ALIAS = {
    "gemini-flash": "gemini-3-flash-preview",
    "gemini-flash-lite": "gemini-3.1-flash-lite-preview",
    "gemini-pro": "gemini-2.5-pro-preview-05-06",
}


def resolve_model(alias: str) -> str:
    """將 BE 回傳的 model alias 解析為實際 model ID。"""
    return MODEL_ALIAS.get(alias, alias)


class GoogleSearchToolCompat(BaseTool):
    """Google Search tool，支援所有 Gemini model（ADK 內建版只支援 1.x/2.x）。"""

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
    """ADK tool：Vertex AI RAG Engine 知識庫檢索。

    - gemini-2 model → built-in retrieval tool
    - 其他 model → rag.retrieval_query 獨立查詢
    """

    def __init__(
        self,
        rag_resource_name: str,
        similarity_top_k: int = 5,
        vector_distance_threshold: float = 0.6,
    ):
        super().__init__(
            name="rag_retrieval",
            description="從知識庫檢索相關營養與飲食資料",
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
        if llm_request.model and llm_request.model.startswith("gemini-2"):
            llm_request.config = llm_request.config or types.GenerateContentConfig()
            llm_request.config.tools = llm_request.config.tools or []
            llm_request.config.tools.append(
                types.Tool(
                    retrieval=types.Retrieval(
                        vertex_rag_store=self.vertex_rag_store
                    )
                )
            )
        else:
            await super().process_llm_request(
                tool_context=tool_context, llm_request=llm_request
            )

    async def run_async(
        self, *, args: dict[str, Any], tool_context: ToolContext
    ) -> Any:
        """非 gemini-2 model 時，用 rag.retrieval_query 獨立查詢。"""
        response = rag.retrieval_query(
            text=args["query"],
            rag_resources=self.rag_resources,
            similarity_top_k=self.similarity_top_k,
            vector_distance_threshold=self.vector_distance_threshold,
        )
        if not response.contexts.contexts:
            return "No matching result found in knowledge base."
        return [ctx.text for ctx in response.contexts.contexts]


def _guess_mime_type(file_name: str) -> str:
    """根據副檔名推斷 MIME type，預設 application/pdf。"""
    mime, _ = mimetypes.guess_type(file_name)
    if not mime:
        return "application/pdf"
    if mime == "application/pdf":
        return mime
    return "text/plain"


def _build_knowledge_parts(rag_files: list[dict]) -> list[types.Part]:
    """full_context 模式：將 rag_files 的檔案 URL 轉成 file_data Parts。"""
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
    """根據 BE 設定和 context_data 動態建立 ADK Agent。

    Returns:
        (agent, knowledge_parts): agent 和 full_context 模式的知識庫 Parts。
    """
    system_prompt = config.get("system_prompt", "")
    context_json = json.dumps(context_data, ensure_ascii=False, indent=2)
    instruction_text = f"{system_prompt}\n\n## 參考資料\n{context_json}"
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

    # 知識庫
    rag_resource_name = config.get("rag_resource_name")
    rag_files = config.get("rag_files") or []
    knowledge_parts: list[types.Part] = []

    full_context_files = [f for f in rag_files if f.get("rag_mode") == "full_context"]
    if full_context_files:
        knowledge_parts = _build_knowledge_parts(full_context_files)

    if rag_resource_name and rag_files:
        tools.append(VertexRagTool(rag_resource_name))

    generate_content_config = types.GenerateContentConfig()

    # Gemini 原生 tools
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
    """建立 orchestrator Agent，掛載 sub_agents 做編排。

    Args:
        system_prompt: orchestrator 的 system prompt（只管路由，不管分析）
        model_config: orchestrator 的 model 設定（通常用 gemini-pro）
        sub_agents: 已建好的 skill Agent 清單
        skill_key: 頂層 skill key（用於命名）

    Returns:
        orchestrator Agent
    """
    if not sub_agents:
        raise ValueError("create_orchestrator requires at least one sub_agent")

    raw_model = model_config.get("model", DEFAULT_MODEL)
    model = resolve_model(raw_model)

    instruction = lambda _ctx, _t=system_prompt: _t

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
