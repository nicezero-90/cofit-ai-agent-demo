"""Orchestrator — AI Brain A2A 編排層。

負責：
1. resolve_skill_configs: 解析 skills[]，inline 或打 BE
2. run_auto / run_parallel / run_sequential: 三種編排模式（後續 Task 實作）
"""

import asyncio
import json
import logging
from typing import Any

from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

from src.agent_factory import create_agent, create_orchestrator
from src.cofit_api_client import CofitApiClient
from src.constants import COFIT_API_URL, COFIT_TOKEN

logger = logging.getLogger(__name__)

_api_client: CofitApiClient | None = None


def _get_api_client() -> CofitApiClient:
    global _api_client
    if _api_client is None:
        _api_client = CofitApiClient(base_url=COFIT_API_URL, token=COFIT_TOKEN)
    return _api_client


async def resolve_skill_configs(
    skills: list[dict],
    client_id: int | None,
) -> list[dict]:
    """解析 skills[]，inline 有 system_prompt 就直接用，否則打 BE。

    回傳格式:
    [
        {
            "skill_key": "lab_report",
            "description": "分析檢驗報告",
            "config": {"system_prompt": "...", "model_config": {...}, "tools": [], "rag_files": []},
            "context_data": {...},
        },
        ...
    ]
    """
    inline_results = []
    remote_tasks = []

    for skill in skills:
        skill_key = skill["skill_key"]
        description = skill.get("description", "")

        if skill.get("system_prompt"):
            inline_results.append({
                "skill_key": skill_key,
                "description": description,
                "config": {
                    "system_prompt": skill["system_prompt"],
                    "model_config": skill.get("model_config") or {},
                    "tools": skill.get("tools") or [],
                    "rag_files": skill.get("rag_files") or [],
                    "rag_resource_name": skill.get("rag_resource_name"),
                },
                "context_data": skill.get("context_data") or {},
            })
        else:
            remote_tasks.append((skill_key, description))

    remote_results = []
    if remote_tasks and client_id is not None:
        api = _get_api_client()
        loop = asyncio.get_event_loop()

        async def fetch_one(sk: str, desc: str) -> dict | None:
            try:
                config, context_data = await loop.run_in_executor(
                    None, lambda: api.get_context_data(sk, client_id=client_id)
                )
                return {
                    "skill_key": sk,
                    "description": desc,
                    "config": config,
                    "context_data": context_data,
                }
            except Exception:
                logger.warning(f"Failed to fetch skill config for '{sk}', skipping")
                return None

        fetched = await asyncio.gather(*[fetch_one(sk, desc) for sk, desc in remote_tasks])
        remote_results = [r for r in fetched if r is not None]

    return inline_results + remote_results


def _extract_text_from_event(event) -> str:
    """Extract text content from an ADK event."""
    if not hasattr(event, "content") or event.content is None:
        return ""
    parts = getattr(event.content, "parts", None)
    if not parts:
        return ""
    return "".join(getattr(p, "text", "") or "" for p in parts)


def _build_sub_agents(resolved_skills: list[dict]) -> tuple[list, list[types.Part]]:
    """從 resolved_skills 建立 sub-agent 清單。

    Returns:
        (sub_agents, all_knowledge_parts)
    """
    sub_agents = []
    all_knowledge_parts = []
    for skill in resolved_skills:
        agent, knowledge_parts = create_agent(
            config=skill["config"],
            context_data=skill["context_data"],
            skill_key=skill["skill_key"],
        )
        sub_agents.append(agent)
        all_knowledge_parts.extend(knowledge_parts)
    return sub_agents, all_knowledge_parts


async def run_auto(
    resolved_skills: list[dict],
    system_prompt: str,
    model_config: dict,
    skill_key: str,
    message: str,
    stream: bool = False,
):
    """auto 模式：建立 orchestrator + sub_agents，讓 ADK 自動路由。

    Returns:
        str（非 streaming）或 async generator（streaming）
    """
    sub_agents, knowledge_parts = _build_sub_agents(resolved_skills)

    orchestrator = create_orchestrator(
        system_prompt=system_prompt,
        model_config=model_config,
        sub_agents=sub_agents,
        skill_key=skill_key,
    )

    session_service = InMemorySessionService()
    app_name = f"brain_{skill_key}"
    user_id = "be_caller"
    session = await session_service.create_session(app_name=app_name, user_id=user_id)
    runner = Runner(app_name=app_name, agent=orchestrator, session_service=session_service)

    parts = knowledge_parts + [types.Part(text=message)]
    user_content = types.Content(role="user", parts=parts)

    if stream:
        return _stream_runner(runner, user_id, session.id, user_content)
    else:
        return await _collect_runner(runner, user_id, session.id, user_content)


async def _collect_runner(runner, user_id: str, session_id: str, user_content) -> str:
    """跑 runner 並收集完整文字結果。"""
    full_text = []
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=user_content,
        run_config=RunConfig(streaming_mode=StreamingMode.NONE),
    ):
        text = _extract_text_from_event(event)
        if text:
            full_text.append(text)
    return "".join(full_text) or "抱歉，我無法產生回覆，請再試一次。"


async def _stream_runner(runner, user_id: str, session_id: str, user_content):
    """跑 runner 並 yield SSE chunks。"""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=user_content,
        run_config=RunConfig(streaming_mode=StreamingMode.SSE),
    ):
        if not getattr(event, "partial", False):
            continue
        text = _extract_text_from_event(event)
        if text:
            yield text


async def _run_single_agent(
    config: dict,
    context_data: dict,
    skill_key: str,
    message: str,
) -> tuple[str, str]:
    """跑單一 skill agent，回傳 (skill_key, result)。"""
    try:
        agent, knowledge_parts = create_agent(
            config=config, context_data=context_data, skill_key=skill_key,
        )
        session_service = InMemorySessionService()
        app_name = f"skill_{skill_key}"
        user_id = "be_caller"
        session = await session_service.create_session(app_name=app_name, user_id=user_id)
        runner = Runner(app_name=app_name, agent=agent, session_service=session_service)

        parts = knowledge_parts + [types.Part(text=message)]
        user_content = types.Content(role="user", parts=parts)

        result = await _collect_runner(runner, user_id, session.id, user_content)
        return skill_key, result
    except Exception as e:
        logger.warning(f"Sub-agent '{skill_key}' failed: {e}")
        return skill_key, f"[錯誤] {skill_key} 執行失敗: {e}"


async def run_parallel(
    resolved_skills: list[dict],
    system_prompt: str,
    model_config: dict,
    skill_key: str,
    message: str,
    stream: bool = False,
):
    """parallel 模式：同時跑所有 sub-agent，orchestrator 彙整。"""
    # 1. 並行跑所有 sub-agent
    tasks = [
        _run_single_agent(
            config=s["config"],
            context_data=s["context_data"],
            skill_key=s["skill_key"],
            message=message,
        )
        for s in resolved_skills
    ]
    results = await asyncio.gather(*tasks)

    # 2. 組 summary prompt
    summary_parts = []
    for sk, result_text in results:
        desc = next((s["description"] for s in resolved_skills if s["skill_key"] == sk), sk)
        summary_parts.append(f"### {desc}（{sk}）\n{result_text}")
    summary = "\n\n".join(summary_parts)

    summarize_prompt = (
        f"以下是各專家的分析結果，請彙整成一份完整報告：\n\n{summary}"
    )

    # 3. Orchestrator 彙整（不需要 sub_agents，直接用單一 agent）
    summarize_config = {
        "system_prompt": system_prompt,
        "model_config": model_config,
        "tools": [],
        "rag_files": [],
    }
    summarizer, _ = create_agent(
        config=summarize_config, context_data={}, skill_key=f"{skill_key}_summarizer",
    )
    session_service = InMemorySessionService()
    app_name = f"brain_{skill_key}_summarize"
    user_id = "be_caller"
    session = await session_service.create_session(app_name=app_name, user_id=user_id)
    runner = Runner(app_name=app_name, agent=summarizer, session_service=session_service)

    user_content = types.Content(role="user", parts=[types.Part(text=summarize_prompt)])

    if stream:
        return _stream_runner(runner, user_id, session.id, user_content)
    else:
        return await _collect_runner(runner, user_id, session.id, user_content)


async def run_sequential(
    resolved_skills: list[dict],
    system_prompt: str,
    model_config: dict,
    skill_key: str,
    message: str,
    stream: bool = False,
):
    """sequential 模式：依序執行，前一步結果串接到下一步。"""
    accumulated_context = ""

    for i, skill in enumerate(resolved_skills):
        # 組合 message：原始問題 + 前面步驟的結果
        if accumulated_context:
            step_message = (
                f"原始問題：{message}\n\n"
                f"前面步驟的分析結果：\n{accumulated_context}\n\n"
                f"請根據以上資訊繼續分析。"
            )
        else:
            step_message = message

        sk, result_text = await _run_single_agent(
            config=skill["config"],
            context_data=skill["context_data"],
            skill_key=skill["skill_key"],
            message=step_message,
        )

        desc = skill.get("description", sk)
        accumulated_context += f"\n### {desc}（{sk}）\n{result_text}\n"

    # 最後 orchestrator 彙整
    summarize_prompt = (
        f"原始問題：{message}\n\n"
        f"以下是各專家依序分析的結果，請彙整成一份完整報告：\n{accumulated_context}"
    )

    summarize_config = {
        "system_prompt": system_prompt,
        "model_config": model_config,
        "tools": [],
        "rag_files": [],
    }
    summarizer, _ = create_agent(
        config=summarize_config, context_data={}, skill_key=f"{skill_key}_summarizer",
    )
    session_service = InMemorySessionService()
    app_name = f"brain_{skill_key}_summarize"
    user_id = "be_caller"
    session = await session_service.create_session(app_name=app_name, user_id=user_id)
    runner = Runner(app_name=app_name, agent=summarizer, session_service=session_service)

    user_content = types.Content(role="user", parts=[types.Part(text=summarize_prompt)])

    if stream:
        return _stream_runner(runner, user_id, session.id, user_content)
    else:
        return await _collect_runner(runner, user_id, session.id, user_content)
