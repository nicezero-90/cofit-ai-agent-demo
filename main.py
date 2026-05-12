# main.py — 診所 AI 大腦 POC
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

from src.agent_factory import create_agent
from src.orchestrator import resolve_skill_configs, run_auto, run_parallel, run_sequential

# ── Logging ──────────────────────────────────────────────
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(message)s"))
logging.root.handlers = [handler]
logging.root.setLevel(logging.INFO)

# ── FastAPI App ──────────────────────────────────────────
app = FastAPI(
    title="診所 AI 大腦 POC",
    description="BE 送完整 config，AI platform 建立 Gemini Agent 執行後回傳結果。",
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
    """v2 路徑：orchestrator + sub-agents。"""
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
    summary="AI 大腦（BE 直接呼叫，免認證）",
    description=(
        "BE 送整包 config，AI platform 建立 Gemini Agent 執行後回傳結果。\n\n"
        "**免認證** — BE 內部呼叫，不需帶任何 token。\n\n"
        "**BE 呼叫範例：**\n"
        "```\n"
        "POST /ai-brain\n"
        "Content-Type: application/json\n\n"
        '{"key":"doctor_visit_initial","system_prompt":"...","model_config":{"model":"gemini-flash"},'
        '"tools":["google_search"],"rag_files":[...],"context_data":{...}}\n'
        "```\n\n"
        "**回傳：**\n"
        "```json\n"
        '{"result": "根據病人的報告...", "skill_key": "doctor_visit_initial"}\n'
        "```\n\n"
        "**Streaming（`stream: true`）：** 回傳 SSE，每個 chunk 格式 `data: {\"text\": \"...\"}`，結束 `data: [DONE]`"
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
                                "description": "Skill key（對應 BE ai_skills 設定）",
                                "example": "doctor_visit_initial",
                            },
                            "system_prompt": {
                                "type": "string",
                                "description": "Agent 的系統 prompt（完整內容）",
                                "example": "你是一名經驗豐富的醫師，具有專業的功能醫學知識...",
                            },
                            "model_config": {
                                "type": "object",
                                "description": "模型設定",
                                "required": ["model"],
                                "properties": {
                                    "model": {
                                        "type": "string",
                                        "description": "模型別名：gemini-flash / gemini-flash-lite / gemini-pro",
                                        "example": "gemini-flash",
                                    },
                                },
                            },
                            "tools": {
                                "type": "array",
                                "description": "啟用的工具（選填）",
                                "items": {"type": "string", "enum": ["google_search"]},
                                "example": ["google_search"],
                            },
                            "rag_files": {
                                "type": "array",
                                "description": "知識庫檔案（選填）。rag_mode=full_context 會將檔案全文注入 context",
                                "items": {
                                    "type": "object",
                                    "required": ["file_name", "url", "rag_mode"],
                                    "properties": {
                                        "file_name": {
                                            "type": "string",
                                            "example": "初日診所營養品指南.json",
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
                                "description": "Vertex AI RAG corpus（選填）",
                                "example": "projects/78906692519/locations/asia-east1/ragCorpora/7665689515738005504",
                            },
                            "context_data": {
                                "type": "object",
                                "description": "客戶資料（JSON，注入 system prompt 的「參考資料」區段）",
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
                                "description": "使用者訊息（選填，預設「請根據資料產出分析報告」）",
                                "example": "請根據這位病人的資料，產出初診諮詢稿",
                            },
                            "stream": {
                                "type": "boolean",
                                "description": "SSE streaming 模式（預設 false）",
                                "default": False,
                            },
                        },
                    },
                }
            },
        },
        "responses": {
            "200": {
                "description": "執行成功",
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
                            "result": "Samuel，你好！根據您的檢測報告與問卷，我為您整理了以下建議...",
                            "skill_key": "doctor_visit_initial",
                        },
                    },
                    "text/event-stream": {
                        "schema": {"type": "string"},
                        "example": 'data: {"text": "Samuel，你好！"}\ndata: {"text": "根據您的..."}\ndata: [DONE]\n',
                    },
                },
            },
            "400": {
                "description": "缺少必要欄位",
                "content": {
                    "application/json": {
                        "example": {"error": "key, system_prompt, model_config, and context_data are required"},
                    }
                },
            },
            "500": {
                "description": "Agent 執行失敗",
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
    message = body.get("message") or body.get("user_prompt") or "請根據資料產出分析報告"

    if not skill_key or not system_prompt or not model_config:
        return JSONResponse(
            status_code=400,
            content={"error": "key, system_prompt, model_config, and context_data are required"},
        )

    # ── v2：有 skills → 走 orchestrator ──
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

    # ── v1：原有邏輯（無 skills）──
    # 直接用 BE 送來的 config 建立 agent
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

    # Session + Runner（stateless，每次建新 session）
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
                result = "抱歉，我無法產生回覆，請再試一次。"
        except Exception as e:
            logger.exception(f"Error in /ai-brain: {e}")
            return JSONResponse(status_code=500, content={"error": "Agent execution failed"})

        return JSONResponse(
            status_code=200,
            content={"result": result, "skill_key": skill_key},
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
