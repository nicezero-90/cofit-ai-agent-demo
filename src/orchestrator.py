"""Orchestrator — AI Brain A2A orchestration layer.

Responsibilities:
1. resolve_skill_configs: resolve skills[], use inline or fetch from BE
2. run_auto / run_parallel / run_sequential: three orchestration modes
3. run_graph: graph mode (topological sort execution via DAG edges)
"""

import asyncio
import json
import logging
from collections import defaultdict, deque
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
    """Resolve skills[]: use inline system_prompt directly, otherwise fetch from BE.

    Returns:
    [
        {
            "skill_key": "lab_report",
            "description": "Analyze lab report",
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
    """Build sub-agent list from resolved_skills.

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
    """auto mode: create orchestrator + sub_agents, let ADK route automatically.

    Returns:
        str (non-streaming) or async generator (streaming)
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
    """Run runner and collect full text result."""
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
    return "".join(full_text) or "Sorry, I was unable to generate a response. Please try again."


async def _stream_runner(runner, user_id: str, session_id: str, user_content):
    """Run runner and yield SSE chunks."""
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
    """Run a single skill agent, returns (skill_key, result)."""
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
        return skill_key, f"[Error] {skill_key} execution failed: {e}"


async def run_parallel(
    resolved_skills: list[dict],
    system_prompt: str,
    model_config: dict,
    skill_key: str,
    message: str,
    stream: bool = False,
):
    """parallel mode: run all sub-agents concurrently, orchestrator aggregates results."""
    # 1. Run all sub-agents in parallel
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

    # 2. Build summary prompt
    summary_parts = []
    for sk, result_text in results:
        desc = next((s["description"] for s in resolved_skills if s["skill_key"] == sk), sk)
        summary_parts.append(f"### {desc} ({sk})\n{result_text}")
    summary = "\n\n".join(summary_parts)

    summarize_prompt = (
        f"The following are analysis results from each expert. Please compile them into a comprehensive report:\n\n{summary}"
    )

    # 3. Orchestrator aggregates (no sub_agents needed, use single agent directly)
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


def _topo_layers(node_ids: set[str], edges: list[dict]) -> list[list[str]]:
    """Kahn's BFS — returns parallel execution layers (same layer has no deps; lower layers run after upper layers complete)."""
    in_deg = {nid: 0 for nid in node_ids}
    succ: dict[str, list[str]] = defaultdict(list)
    for e in edges:
        f, t = e["from"], e["to"]
        if f in node_ids and t in node_ids:
            succ[f].append(t)
            in_deg[t] += 1
    q = deque(nid for nid in node_ids if in_deg[nid] == 0)
    layers: list[list[str]] = []
    while q:
        layer = list(q)
        q.clear()
        layers.append(layer)
        for nid in layer:
            for s in succ[nid]:
                in_deg[s] -= 1
                if in_deg[s] == 0:
                    q.append(s)
    return layers


async def run_graph(
    manifest: dict,
    skills_data: dict,
    message: str,
    stream: bool = False,
):
    """graph mode: topological sort by manifest edges, same layer runs in parallel, downstream receives upstream output.

    Args:
        manifest: response from GET /v5/ai_agents/:key (contains nodes / edges)
        skills_data: skills dict from GET /v5/ai_agents/:key/context_data
                     {skill_key: {system_prompt, model_config, tools, context_data, ...}}
        message: user input
        stream: whether to stream (graph mode streaming = stream final output only)

    Returns:
        str (non-streaming) or async generator (streaming, only streams final output)
    """
    nodes: list[dict] = [n for n in manifest.get("nodes", []) if n.get("connected", True)]
    edges: list[dict] = manifest.get("edges", [])
    node_skill: dict[str, str] = {n["node_id"]: n["skill_key"] for n in nodes}
    node_ids = set(node_skill.keys())
    layers = _topo_layers(node_ids, edges)

    # predecessor map: nid → [upstream nid] (exclude root, root is only a topological entry point with no output)
    pred: dict[str, list[str]] = defaultdict(list)
    for e in edges:
        if e["from"] != "root":
            pred[e["to"]].append(e["from"])

    # successor map (used to find leaf nodes)
    has_successor = {e["from"] for e in edges if e["from"] in node_ids}
    leaf_nodes = [nid for nid in node_ids if nid not in has_successor]

    outputs: dict[str, str] = {}

    for layer in layers:
        tasks = []
        for nid in layer:
            sk = node_skill[nid]
            skill_data = skills_data.get(sk, {})
            config = {
                "system_prompt": skill_data.get("system_prompt", ""),
                "model_config": skill_data.get("model_config", {}),
                "tools": skill_data.get("tools", []),
                "rag_files": skill_data.get("rag_files", []),
                "rag_resource_name": skill_data.get("rag_resource_name"),
            }
            context_data = skill_data.get("context_data", {})

            # Build input: original message + upstream outputs
            upstream_texts = [outputs[p] for p in pred[nid] if p in outputs]
            if upstream_texts:
                node_input = (
                    f"## Upstream Analysis Results\n{chr(10).join(upstream_texts)}\n\n"
                    f"## User Input\n{message}"
                )
            else:
                node_input = message

            tasks.append(
                _run_single_agent(
                    config=config,
                    context_data=context_data,
                    skill_key=sk,
                    message=node_input,
                )
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            nid = layer[i]
            if isinstance(result, Exception):
                logger.error("Graph: node %s failed: %s", nid, result)
                outputs[nid] = f"[{nid} execution failed: {result}]"
            else:
                _, text = result
                outputs[nid] = text

    # Final output: merge results from leaf nodes
    final_parts = [outputs.get(nid, "") for nid in leaf_nodes if outputs.get(nid)]
    final_text = "\n\n".join(final_parts) or "Sorry, I was unable to generate a response. Please try again."

    if stream:
        async def _gen():
            yield final_text
        return _gen()
    else:
        return final_text


async def run_sequential(
    resolved_skills: list[dict],
    system_prompt: str,
    model_config: dict,
    skill_key: str,
    message: str,
    stream: bool = False,
):
    """sequential mode: execute in order, chain previous result into next step."""
    accumulated_context = ""

    for i, skill in enumerate(resolved_skills):
        # Build message: original question + results from previous steps
        if accumulated_context:
            step_message = (
                f"Original question: {message}\n\n"
                f"Analysis results from previous steps:\n{accumulated_context}\n\n"
                f"Please continue the analysis based on the above information."
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
        accumulated_context += f"\n### {desc} ({sk})\n{result_text}\n"

    # Final orchestrator aggregation
    summarize_prompt = (
        f"Original question: {message}\n\n"
        f"The following are sequential analysis results from each expert. Please compile them into a comprehensive report:\n{accumulated_context}"
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
