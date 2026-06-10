# main.py — Clinic AI Brain POC
import asyncio
import json
import logging
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pythonjsonlogger import jsonlogger

from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

import requests as _requests

from src.agent_factory import create_agent
from src.cofit_api_client import CofitApiClient
from src.constants import COFIT_API_URL, COFIT_TOKEN
from src.orchestrator import resolve_skill_configs, run_auto, run_parallel, run_sequential, run_graph

# ── Logging ──────────────────────────────────────────────
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(message)s"))
logging.root.handlers = [handler]
logging.root.setLevel(logging.INFO)

# ── FastAPI App ──────────────────────────────────────────
app = FastAPI(
    title="Clinic AI Brain POC",
    description="BE sends full config, AI platform creates and runs Gemini Agent, returns result.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ──────────────────────────────────────────────
def _extract_text_from_event(event) -> str:
    """Extract text content from an ADK event."""
    if not hasattr(event, "content") or event.content is None:
        return ""
    parts = getattr(event.content, "parts", None)
    if not parts:
        return ""
    texts = []
    for part in parts:
        text = getattr(part, "text", None)
        if text:
            texts.append(text)
    return "".join(texts)


# ── Health Check ─────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


# ── AI Brain Endpoint ────────────────────────────────────
async def run_brain_v2(
    body: dict,
    skill_key: str,
    system_prompt: str,
    model_config: dict,
    context_data: dict,
    message: str,
    skills: list[dict],
    stream: bool,
):
    """v2 path: orchestrator + sub-agents."""
    client_id = None
    if isinstance(context_data, dict):
        client_id = context_data.get("client_id")

    resolved = await resolve_skill_configs(skills, client_id=client_id)
    if not resolved:
        raise ValueError("Failed to fetch skill configs")

    mode = body.get("orchestration_mode", "auto")

    if mode == "parallel":
        run_fn = run_parallel
    elif mode == "sequential":
        run_fn = run_sequential
    else:
        run_fn = run_auto

    return await run_fn(
        resolved_skills=resolved,
        system_prompt=system_prompt,
        model_config=model_config,
        skill_key=skill_key,
        message=message,
        stream=stream,
    )


@app.post(
    "/ai-brain",
    tags=["AI Brain"],
    summary="AI Brain (BE direct call, no auth required)",
    description=(
        "BE sends full config, AI platform creates and runs Gemini Agent, returns result.\n\n"
        "**No auth required** — internal BE call, no token needed.\n\n"
        "**BE call example:**\n"
        "```\n"
        "POST /ai-brain\n"
        "Content-Type: application/json\n\n"
        '{"key":"doctor_visit_initial","system_prompt":"...","model_config":{"model":"gemini-flash"},'
        '"tools":["google_search"],"rag_files":[...],"context_data":{...}}\n'
        "```\n\n"
        "**Response:**\n"
        "```json\n"
        '{"result": "Based on patient report...", "skill_key": "doctor_visit_initial"}\n'
        "```\n\n"
        "**Streaming (`stream: true`):** Returns SSE, each chunk format `data: {\"text\": \"...\"}`, ends with `data: [DONE]`"
    ),
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "required": ["key", "system_prompt", "model_config", "context_data"],
                        "properties": {
                            "key": {
                                "type": "string",
                                "description": "Skill key (maps to BE ai_skills config)",
                                "example": "doctor_visit_initial",
                            },
                            "system_prompt": {
                                "type": "string",
                                "description": "Agent system prompt (full content)",
                                "example": "You are an experienced physician with expertise in functional medicine...",
                            },
                            "model_config": {
                                "type": "object",
                                "description": "Model configuration",
                                "required": ["model"],
                                "properties": {
                                    "model": {
                                        "type": "string",
                                        "description": "Model alias: gemini-flash / gemini-flash-lite / gemini-pro",
                                        "example": "gemini-flash",
                                    },
                                },
                            },
                            "tools": {
                                "type": "array",
                                "description": "Enabled tools (optional)",
                                "items": {"type": "string", "enum": ["google_search"]},
                                "example": ["google_search"],
                            },
                            "rag_files": {
                                "type": "array",
                                "description": "Knowledge base files (optional). rag_mode=full_context injects full file content into context",
                                "items": {
                                    "type": "object",
                                    "required": ["file_name", "url", "rag_mode"],
                                    "properties": {
                                        "file_name": {
                                            "type": "string",
                                            "example": "clinic_supplement_guide.json",
                                        },
                                        "url": {
                                            "type": "string",
                                            "example": "https://storage.googleapis.com/cofit-pro-staging/rag_files/260421_d120ab3f.json",
                                        },
                                        "gcs_uri": {
                                            "type": "string",
                                            "example": "gs://cofit-pro-staging/rag_files/260421_d120ab3f.json",
                                        },
                                        "rag_mode": {
                                            "type": "string",
                                            "enum": ["full_context", "retrieval"],
                                            "example": "full_context",
                                        },
                                    },
                                },
                            },
                            "rag_resource_name": {
                                "type": "string",
                                "description": "Vertex AI RAG corpus (optional)",
                                "example": "projects/78906692519/locations/asia-east1/ragCorpora/7665689515738005504",
                            },
                            "context_data": {
                                "type": "object",
                                "description": "Client data (JSON, injected into the Reference Data section of system prompt)",
                                "example": {
                                    "client": {
                                        "real_name": "Samuel QA_Test",
                                        "gender": "male",
                                        "age": 35,
                                        "current_date": "2026-04-22",
                                    },
                                },
                            },
                            "message": {
                                "type": "string",
                                "description": "User message (optional, default: 'Please generate analysis report based on the data')",
                                "example": "Please generate an initial consultation report based on this patient's data",
                            },
                            "stream": {
                                "type": "boolean",
                                "description": "SSE streaming mode (default false)",
                                "default": False,
                            },
                            "skills": {
                                "type": "array",
                                "description": "Sub-agent list (optional, v2). If omitted, uses v1 single agent",
                                "items": {
                                    "type": "object",
                                    "required": ["skill_key", "description"],
                                    "properties": {
                                        "skill_key": {
                                            "type": "string",
                                            "description": "Maps to BE /v5/ai_skills/:key",
                                            "example": "lab_report",
                                        },
                                        "description": {
                                            "type": "string",
                                            "description": "Used by orchestrator for routing decisions",
                                            "example": "Analyze lab report data, identify abnormal values and provide recommendations",
                                        },
                                        "system_prompt": {
                                            "type": "string",
                                            "description": "Inline mode: skill system prompt (optional, fetched from BE if not provided)",
                                        },
                                        "model_config": {
                                            "type": "object",
                                            "description": "Inline mode: model config (optional)",
                                        },
                                        "context_data": {
                                            "type": "object",
                                            "description": "Inline mode: context data (optional)",
                                        },
                                        "tools": {
                                            "type": "array",
                                            "description": "Inline mode: tools (optional)",
                                            "items": {"type": "string"},
                                        },
                                        "rag_files": {
                                            "type": "array",
                                            "description": "Inline mode: knowledge base files (optional)",
                                        },
                                    },
                                },
                                "example": [
                                    {"skill_key": "lab_report", "description": "Analyze lab report data"},
                                    {"skill_key": "body_measurement", "description": "Analyze body composition trends"},
                                ],
                            },
                            "orchestration_mode": {
                                "type": "string",
                                "description": "Orchestration mode (optional, default: auto)",
                                "enum": ["auto", "parallel", "sequential"],
                                "default": "auto",
                            },
                        },
                    },
                }
            },
        },
        "responses": {
            "200": {
                "description": "Execution successful",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "result": {"type": "string"},
                                "skill_key": {"type": "string"},
                            },
                        },
                        "example": {
                            "result": "Hi Samuel! Based on your test report and questionnaire, here are my recommendations...",
                            "skill_key": "doctor_visit_initial",
                        },
                    },
                    "text/event-stream": {
                        "schema": {"type": "string"},
                        "example": 'data: {"text": "Hi Samuel!"}\ndata: {"text": "Based on your..."}\ndata: [DONE]\n',
                    },
                },
            },
            "400": {
                "description": "Missing required fields",
                "content": {
                    "application/json": {
                        "example": {"error": "key, system_prompt, model_config, and context_data are required"},
                    }
                },
            },
            "500": {
                "description": "Agent execution failed",
                "content": {
                    "application/json": {
                        "example": {"error": "Agent execution failed"},
                    }
                },
            },
        },
    },
)
async def ai_brain(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    skill_key = body.get("key", "")
    system_prompt = body.get("system_prompt", "")
    model_config = body.get("model_config", {})
    context_data = body.get("context_data", {})
    stream = body.get("stream", False)
    message = body.get("message") or body.get("user_prompt") or "Please generate analysis report based on the data"

    if not skill_key or not system_prompt or not model_config:
        return JSONResponse(
            status_code=400,
            content={"error": "key, system_prompt, model_config, and context_data are required"},
        )

    # ── v2: has skills → use orchestrator ──
    skills = body.get("skills") or []
    if skills:
        if stream:
            async def v2_stream():
                try:
                    result = await run_brain_v2(
                        body=body, skill_key=skill_key, system_prompt=system_prompt,
                        model_config=model_config, context_data=context_data,
                        message=message, skills=skills, stream=True,
                    )
                    async for text in result:
                        yield f"data: {json.dumps({'text': text}, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    logger.exception(f"Error in /ai-brain v2 streaming: {e}")
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                    yield "data: [DONE]\n\n"
            return StreamingResponse(v2_stream(), media_type="text/event-stream")
        else:
            try:
                result = await run_brain_v2(
                    body=body, skill_key=skill_key, system_prompt=system_prompt,
                    model_config=model_config, context_data=context_data,
                    message=message, skills=skills, stream=False,
                )
                return JSONResponse(status_code=200, content={"result": result, "skill_key": skill_key})
            except Exception as e:
                logger.exception(f"Error in /ai-brain v2: {e}")
                return JSONResponse(status_code=500, content={"error": "Agent execution failed"})

    # ── v1: original logic (no skills) ──
    # Build agent directly from BE-provided config
    config = {
        "system_prompt": system_prompt,
        "model_config": model_config,
        "tools": body.get("tools") or [],
        "rag_resource_name": body.get("rag_resource_name"),
        "rag_files": body.get("rag_files") or [],
    }

    agent, knowledge_parts = create_agent(
        config=config,
        context_data=context_data,
        skill_key=skill_key,
    )

    # Session + Runner (stateless, new session each time)
    session_service = InMemorySessionService()
    app_name = f"brain_{skill_key}"
    user_id = "be_caller"
    session = await session_service.create_session(
        app_name=app_name, user_id=user_id
    )
    runner = Runner(
        app_name=app_name,
        agent=agent,
        session_service=session_service,
    )

    # Build user message with knowledge parts
    parts = knowledge_parts + [types.Part(text=message)]
    user_content = types.Content(role="user", parts=parts)

    if stream:
        async def event_generator():
            try:
                async for event in runner.run_async(
                    user_id=user_id,
                    session_id=session.id,
                    new_message=user_content,
                    run_config=RunConfig(streaming_mode=StreamingMode.SSE),
                ):
                    if not getattr(event, "partial", False):
                        continue
                    text = _extract_text_from_event(event)
                    if text:
                        yield f"data: {json.dumps({'text': text}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.exception(f"Error in /ai-brain streaming: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    else:
        try:
            full_text = []
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session.id,
                new_message=user_content,
                run_config=RunConfig(streaming_mode=StreamingMode.NONE),
            ):
                text = _extract_text_from_event(event)
                if text:
                    full_text.append(text)

            result = "".join(full_text)
            if not result:
                result = "Sorry, I was unable to generate a response. Please try again."
        except Exception as e:
            logger.exception(f"Error in /ai-brain: {e}")
            return JSONResponse(status_code=500, content={"error": "Agent execution failed"})

        return JSONResponse(
            status_code=200,
            content={"result": result, "skill_key": skill_key},
        )


@app.post(
    "/v1/agents/{agent_key}/run",
    tags=["AI Agent"],
    summary="AI Agent execution (multi-skill orchestration)",
    description=(
        "Execute multi-skill orchestration flow based on agent manifest from BE.\n\n"
        "**Two modes:**\n"
        "- `auto`: pass agent + all skills to coordinator LLM, which decides which skills to call\n"
        "- `graph`: execute directed graph defined by edges, same-layer nodes run in parallel\n\n"
        "**Returns 422 when `usable == false`** (agent config has issues, notify backend)"
    ),
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "required": ["client_id"],
                        "properties": {
                            "client_id": {
                                "type": "integer",
                                "description": "User ID",
                                "example": 351,
                            },
                            "message": {
                                "type": "string",
                                "description": "User input (optional, default: 'Please generate analysis report based on the data')",
                                "example": "Please generate a comprehensive analysis report for this patient",
                            },
                            "stream": {
                                "type": "boolean",
                                "description": "SSE streaming mode (default false)",
                                "default": False,
                            },
                        },
                    },
                    "example": {
                        "client_id": 351,
                        "message": "Please generate a comprehensive analysis report",
                        "stream": False,
                    },
                }
            },
        },
        "responses": {
            "200": {
                "description": "Execution successful",
                "content": {
                    "application/json": {
                        "example": {
                            "result": "Based on patient data...",
                            "agent_key": "nutrition-agent",
                            "mode": "graph",
                        },
                    },
                    "text/event-stream": {
                        "schema": {"type": "string"},
                        "example": 'data: {"text": "Based on patient data..."}\ndata: [DONE]\n',
                    },
                },
            },
            "400": {"description": "Missing client_id"},
            "404": {"description": "agent_key not found"},
            "422": {
                "description": "Agent usable == false (skill config has issues)",
                "content": {
                    "application/json": {
                        "example": {
                            "error": "Agent has blocked nodes",
                            "blocked_nodes": [{"node_id": "n1", "reason": "inactive"}],
                        }
                    }
                },
            },
            "502": {"description": "Upstream BE API error"},
            "500": {"description": "Agent execution failed"},
        },
    },
)
async def run_agent(agent_key: str, request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    client_id = body.get("client_id")
    if client_id is None:
        return JSONResponse(status_code=400, content={"error": "client_id is required"})
    client_id = int(client_id)

    message = body.get("message") or "Please generate analysis report based on the data"
    stream = body.get("stream", False)

    # 1. Fetch manifest (orchestration structure)
    api = CofitApiClient(base_url=COFIT_API_URL, token=COFIT_TOKEN)
    loop = asyncio.get_event_loop()
    try:
        manifest = await loop.run_in_executor(None, api.get_ai_agent_manifest, agent_key)
    except _requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else 500
        if status == 404:
            return JSONResponse(status_code=404, content={"error": f"Agent '{agent_key}' not found"})
        logger.error("get_ai_agent_manifest error: %s", e)
        return JSONResponse(status_code=502, content={"error": "Upstream API error"})
    except Exception as e:
        logger.error("get_ai_agent_manifest error: %s", e)
        return JSONResponse(status_code=502, content={"error": "Upstream API error"})

    # 2. usable == False → stop
    if not manifest.get("usable", True):
        blocked = manifest.get("blocked_nodes") or []
        logger.warning("Agent '%s' is not usable, blocked_nodes: %s", agent_key, blocked)
        return JSONResponse(
            status_code=422,
            content={"error": "Agent has blocked nodes", "blocked_nodes": blocked},
        )

    # 3. Batch fetch skill data (connected: false = orphan node, default True for backward compatibility)
    mode = manifest.get("orchestration_mode", "auto")
    connected_nodes = [n for n in manifest.get("nodes", []) if n.get("connected", True)]

    # auto mode only needs first layer (coordinator selects from them), graph mode needs all connected nodes
    if mode == "auto":
        first_layer_node_ids = {e["to"] for e in manifest.get("edges", []) if e["from"] == "root"}
        skill_keys = list({n["skill_key"] for n in connected_nodes if n["node_id"] in first_layer_node_ids})
    else:
        first_layer_node_ids = set()
        skill_keys = list({n["skill_key"] for n in connected_nodes})

    try:
        context_resp = await loop.run_in_executor(
            None, api.get_ai_agent_context_data, agent_key, client_id, skill_keys
        )
    except Exception as e:
        logger.error("get_ai_agent_context_data error: %s", e)
        return JSONResponse(status_code=502, content={"error": "Failed to fetch context data"})

    skills_data: dict = context_resp.get("skills", {})
    errors: dict = context_resp.get("errors", {})
    if errors:
        logger.warning("Agent '%s' context_data errors: %s", agent_key, errors)

    # 4. Execute
    try:
        if mode == "graph":
            result_or_gen = await run_graph(
                manifest=manifest,
                skills_data=skills_data,
                message=message,
                stream=stream,
            )
        else:  # auto
            seen_skills: set[str] = set()
            resolved_skills = []
            for node in connected_nodes:
                if node["node_id"] not in first_layer_node_ids:
                    continue
                sk = node["skill_key"]
                if sk in seen_skills:
                    continue
                seen_skills.add(sk)
                sd = skills_data.get(sk, {})
                resolved_skills.append({
                    "skill_key": sk,
                    "description": node.get("name") or sk,
                    "config": {
                        "system_prompt": sd.get("system_prompt", ""),
                        "model_config": sd.get("model_config") or {},
                        "tools": sd.get("tools", []),
                        "rag_files": sd.get("rag_files", []),
                        "rag_resource_name": sd.get("rag_resource_name"),
                    },
                    "context_data": sd.get("context_data", {}),
                })
            result_or_gen = await run_auto(
                resolved_skills=resolved_skills,
                system_prompt=manifest.get("system_prompt") or "",
                model_config=manifest.get("model_config") or {},
                skill_key=agent_key,
                message=message,
                stream=stream,
            )
    except Exception as e:
        logger.exception("Agent '%s' execution failed: %s", agent_key, e)
        return JSONResponse(status_code=500, content={"error": "Agent execution failed"})

    if stream:
        async def event_generator():
            try:
                async for text in result_or_gen:
                    yield f"data: {json.dumps({'text': text}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.exception("Agent '%s' streaming error: %s", agent_key, e)
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                yield "data: [DONE]\n\n"
        return StreamingResponse(event_generator(), media_type="text/event-stream")
    else:
        return JSONResponse(
            status_code=200,
            content={"result": result_or_gen, "agent_key": agent_key, "mode": mode},
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
