# Architecture — Cofit AI Agent Demo

**Version**: 0.1.0 | **Last Updated**: 2026-06-10

Cofit AI Agent Demo is a POC service extracted from `ai-skill-platform`'s `/ai-brain` endpoint. It supports single-agent (v1) and orchestrator + sub-agents (v2) execution paths, with four orchestration modes.

---

## Table of Contents

- [System Overview](#system-overview)
- [API Endpoints](#api-endpoints)
- [Orchestration Modes](#orchestration-modes)
- [Module Breakdown](#module-breakdown)
- [Data Flow](#data-flow)
- [Knowledge Base (RAG)](#knowledge-base-rag)
- [Model Aliases](#model-aliases)
- [Deployment](#deployment)
- [External Dependencies](#external-dependencies)
- [Environment Variables](#environment-variables)

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                       Cofit Backend (BE)                        │
│   - /v5/ai_skills/:key/context_data   (skill config + data)    │
│   - /v5/ai_agents/:key                (agent manifest)          │
│   - /v5/ai_agents/:key/context_data   (batch skill data)        │
└──────────────────────────────┬──────────────────────────────────┘
                               │  HTTP (Bearer token)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              cofit-ai-agent-demo  (Cloud Run)                   │
│                                                                  │
│  FastAPI (main.py)                                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  POST /ai-brain         POST /v1/agents/{key}/run        │   │
│  │      │                          │                        │   │
│  │      ▼                          ▼                        │   │
│  │  v1: Single Agent        Fetch manifest from BE          │   │
│  │  v2: Orchestrator  ──►   run_auto / run_graph            │   │
│  │      + Sub-agents        (orchestrator.py)               │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  agent_factory.py                                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  create_agent()        → ADK Agent (skill execution)     │   │
│  │  create_orchestrator() → ADK Agent (routing only)        │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
         ┌─────────────────────────────────────┐
         │       Vertex AI / Gemini             │
         │  - Gemini models (flash / pro)       │
         │  - Vertex AI RAG Engine              │
         │  - Google Search (native tool)       │
         └─────────────────────────────────────┘
```

---

## API Endpoints

### `GET /health`

Returns `{"status": "ok"}`.

---

### `POST /ai-brain`

Core endpoint, no auth required (internal BE call). Routes to v1 or v2 based on whether the `skills` field is present.

**Required fields**: `key`, `system_prompt`, `model_config`

| Field | Description |
|-------|-------------|
| `key` | Skill key (maps to BE `ai_skills` config) |
| `system_prompt` | Agent system prompt |
| `model_config.model` | Model alias (see [Model Aliases](#model-aliases)) |
| `context_data` | Client data (JSON, injected into the Reference Data section of system prompt) |
| `tools` | Enabled tools — currently supports `["google_search"]` |
| `rag_files` | Knowledge base file list (see [Knowledge Base](#knowledge-base-rag)) |
| `rag_resource_name` | Vertex AI RAG corpus resource name |
| `message` | User input (default: `"Please generate analysis report based on the data"`) |
| `stream` | SSE streaming mode (default `false`) |
| `skills` | Sub-agent list (v2 only, see below) |
| `orchestration_mode` | `auto` (default) / `parallel` / `sequential` |

**v2 `skills` array — per skill fields:**

| Field | Description |
|-------|-------------|
| `skill_key` | Maps to BE `/v5/ai_skills/:key` |
| `description` | Used by orchestrator for routing decisions |
| `system_prompt` | Inline mode: provide directly (otherwise fetched from BE) |
| `model_config` | Inline mode: model config |
| `context_data` | Inline mode: client data |
| `tools` / `rag_files` | Inline mode: tools and knowledge base |

**Non-streaming response:**
```json
{ "result": "...", "skill_key": "doctor_visit_initial" }
```

**Streaming response (`stream: true`):**
```
data: {"text": "Hi Samuel!"}
data: {"text": "Based on your..."}
data: [DONE]
```

---

### `POST /v1/agents/{agent_key}/run`

Executes multi-skill orchestration by agent key. Automatically fetches the agent manifest (nodes/edges/mode) and context data from BE, then runs `auto` or `graph` mode based on the manifest's `orchestration_mode`.

| Field | Description |
|-------|-------------|
| `client_id` | User ID (required) |
| `message` | User input |
| `stream` | SSE streaming mode |

**HTTP status codes:**

| Status | Description |
|--------|-------------|
| 200 | Execution successful |
| 400 | Missing `client_id` |
| 404 | `agent_key` not found |
| 422 | Agent `usable == false` (blocked nodes — notify backend) |
| 502 | Upstream BE API error |
| 500 | Agent execution failed |

---

## Orchestration Modes

### v1 — Single Agent

Used when no `skills` field is present. Builds a single ADK Agent directly from BE-provided config.

```
Request ──> create_agent() ──> ADK Runner ──> result
```

---

### v2 / auto — Orchestrator + Sub-agents (ADK native routing)

Creates an Orchestrator Agent with all sub-agents attached. The LLM decides which sub-agent to call.

```
Orchestrator (model from request model_config)
  ├── sub_agent: lab_report (model from skill config)
  ├── sub_agent: body_measurement (model from skill config)
  └── sub_agent: orders (model from skill config)
      └── ADK auto-routes ──> aggregate result
```

Implemented in `run_auto()` in `orchestrator.py`.

---

### v2 / parallel — Concurrent execution + aggregation

All sub-agents run concurrently via `asyncio.gather()`. A summarizer agent aggregates the results.

```
                  ┌── skill_a ──┐
message ──> fork ─┼── skill_b ──┼──> summarizer ──> result
                  └── skill_c ──┘
```

Implemented in `run_parallel()`.

---

### v2 / sequential — Serial execution with context chaining

Runs skills in order. Each step receives the original message plus all prior steps' outputs. A final summarizer aggregates everything.

```
message ──> skill_a ──> skill_b (with skill_a result) ──> skill_c ──> summarizer ──> result
```

Implemented in `run_sequential()`.

---

### graph — DAG topological execution (`/v1/agents/{key}/run` only)

Builds a directed acyclic graph from manifest `nodes` / `edges`. Uses Kahn's BFS topological sort: nodes in the same layer run in parallel; downstream nodes receive their upstream outputs as input.

```
root ──> [n1, n2]  (layer 1, parallel)
              └──> [n3]  (layer 2, receives n1 + n2 outputs)
                       └──> result  (leaf node outputs merged)
```

- Nodes with `connected: false` are excluded as orphans
- Outputs from leaf nodes (no successors) are merged as the final result
- In streaming mode, only the final leaf output is streamed

Implemented in `run_graph()` using `_topo_layers()`.

---

## Module Breakdown

```
cofit-ai-agent-demo/
├── main.py                    # FastAPI app, routing, v1/v2 dispatch, SSE wrapping
├── src/
│   ├── orchestrator.py        # Core orchestration logic
│   │   ├── resolve_skill_configs()     # Parse skills[] (inline or fetch from BE)
│   │   ├── run_auto()                  # auto mode
│   │   ├── run_parallel()              # parallel mode
│   │   ├── run_sequential()            # sequential mode
│   │   ├── run_graph()                 # graph mode (DAG)
│   │   ├── _topo_layers()              # Kahn's BFS topological sort
│   │   ├── _run_single_agent()         # Execute a single skill agent
│   │   ├── _collect_runner()           # Collect full text result
│   │   └── _stream_runner()            # SSE streaming generator
│   ├── agent_factory.py       # ADK Agent creation
│   │   ├── create_agent()              # Build skill Agent (with RAG / tools)
│   │   ├── create_orchestrator()       # Build Orchestrator Agent (with sub_agents)
│   │   ├── GoogleSearchToolCompat      # Google Search tool (all Gemini versions)
│   │   ├── VertexRagTool               # Vertex AI RAG built-in retrieval
│   │   └── resolve_model()             # Model alias resolution
│   ├── cofit_api_client.py    # Cofit BE HTTP client (with retry strategy)
│   │   ├── get_context_data()          # GET /v5/ai_skills/:key/context_data
│   │   ├── get_ai_agent_manifest()     # GET /v5/ai_agents/:key
│   │   └── get_ai_agent_context_data() # GET /v5/ai_agents/:key/context_data (batch)
│   └── constants.py           # GCP_PROJECT_ID / COFIT_API_URL / COFIT_TOKEN
├── Dockerfile
├── requirements.txt
└── .github/workflows/deploy.yml
```

---

## Data Flow

### `/ai-brain` v2 (auto mode) — full request lifecycle

```
1. POST /ai-brain  (body includes skills[])
   │
2. resolve_skill_configs(skills, client_id)
   ├── inline skill: use system_prompt from request directly
   └── remote skill: GET /v5/ai_skills/:key/context_data  ←── Cofit BE
   │
3. run_auto(resolved_skills, system_prompt, model_config, ...)
   │
4. _build_sub_agents(resolved_skills)
   └── create_agent() × N  — one ADK Agent per skill
   │
5. create_orchestrator(sub_agents=[...])
   │
6. ADK Runner.run_async()
   ├── Orchestrator LLM decides → delegates to sub-agent
   └── Sub-agent calls Gemini API (with RAG / Google Search)
   │
7. Return result (JSON or SSE)
```

### `/v1/agents/{key}/run` (graph mode) — full request lifecycle

```
1. POST /v1/agents/{agent_key}/run
   │
2. GET /v5/ai_agents/{key}  ←── fetch manifest (nodes/edges/mode)
   │
3. Filter nodes where connected=true
   │
4. GET /v5/ai_agents/{key}/context_data?client_id=X&skill_keys[]=...
   ←── batch fetch skill config + context data
   │
5. _topo_layers(node_ids, edges)  →  DAG layer segmentation
   │
6. Execute layer by layer (asyncio.gather within each layer)
   ├── Each node: _run_single_agent(config, context_data, upstream_outputs)
   └── Store output in outputs[nid]
   │
7. Merge leaf node outputs → return
```

---

## Knowledge Base (RAG)

Two RAG modes, controlled by `rag_files[].rag_mode`:

| Mode | Description | Implementation |
|------|-------------|----------------|
| `full_context` | Entire file injected as a `file_data` Part in the user message | `_build_knowledge_parts()` |
| `retrieval` | Vector search via Vertex AI RAG Engine (built-in retrieval) | `VertexRagTool` |

`retrieval` mode also requires `rag_resource_name` (Vertex AI RAG corpus resource name).

`VertexRagTool` always uses built-in retrieval — no model version branching.

---

## Model Aliases

BE uses aliases; `resolve_model()` translates them to actual Gemini model IDs before execution:

| Alias | Actual Model ID |
|-------|-----------------|
| `gemini-flash` | `gemini-3-flash-preview` |
| `gemini-flash-lite` | `gemini-3.1-flash-lite-preview` |
| `gemini-pro` | `gemini-2.5-pro-preview-05-06` |

Unknown aliases are passed through as-is. Default model: `gemini-3.1-flash-lite-preview`.

---

## Deployment

| Setting | Value |
|---------|-------|
| Platform | Google Cloud Run |
| GCP Project | `cofit-ai-agent-demo-496007` |
| Service name | `cofit-ai-agent-demo` |
| Region | `asia-east1` |
| Image registry | Artifact Registry (`asia-east1-docker.pkg.dev`) |
| Memory | 1 GiB |
| CPU | 1 vCPU |
| Request timeout | 300 seconds |
| Concurrency | 80 requests/instance |
| Min instances | 0 (cold start) |
| Max instances | 10 |
| Execution environment | Gen2 |

**Auto-deploy trigger**: push to `master` or `main` branch.

**Deploy pipeline:**
```
push to master
  → Build Docker image (tagged with commit SHA)
  → Push to Artifact Registry
  → gcloud run deploy (new revision)
  → update-traffic --to-latest (100% traffic cut)
  → Health check (up to 5 retries)
```

---

## External Dependencies

| Service | Purpose | How called |
|---------|---------|------------|
| Cofit BE API | Fetch skill config, context data, agent manifest | HTTP REST (Bearer token) |
| Vertex AI / Gemini | LLM inference | Google ADK (`google-adk`) |
| Vertex AI RAG Engine | Knowledge base vector retrieval | `VertexRagTool` (built-in retrieval) |
| Google Search | Web search | `GoogleSearchToolCompat` (native tool) |
| Google Cloud Storage | RAG file storage (`full_context` mode references GCS URIs directly) | — |

**Key Python packages:**

| Package | Version | Purpose |
|---------|---------|---------|
| `google-adk` | ≥1.31.0 | ADK Agent creation and execution |
| `google-cloud-aiplatform` | ≥1.60.0 | Vertex AI RAG |
| `fastapi` | — | HTTP API framework |
| `uvicorn` | ≥0.30.0 | ASGI server |
| `requests` | ≥2.32.4,<3.0.0 | BE API HTTP client (with retry) |

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GCP_PROJECT_ID` | GCP project ID | `""` |
| `COFIT_API_URL` | Cofit BE API base URL | `""` (required) |
| `COFIT_TOKEN` | BE API Bearer token | `""` |
| `GOOGLE_CLOUD_PROJECT` | Injected by Cloud Run (same as GCP_PROJECT_ID) | — |
| `GOOGLE_GENAI_USE_VERTEXAI` | Enable Vertex AI backend | `true` (Cloud Run injected) |
| `GOOGLE_CLOUD_LOCATION` | Vertex AI region | `global` (Cloud Run injected) |
