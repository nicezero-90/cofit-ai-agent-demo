# AI Agent API 整合設計

**日期：** 2026-06-05  
**範圍：** cofit-ai-agent-demo — `/v1/agents/{agent_key}/run` 端點 audit & fix  
**目標：** 讓現有 draft 實作與 Cofit AI Agents API spec（2026-06-05）完全對齊

---

## 背景

`cofit-ai-agent-demo` 已有 `/v1/agents/{agent_key}/run` 端點的 draft 實作，包含：

- `CofitApiClient.get_ai_agent_manifest()` / `get_ai_agent_context_data()`
- `run_auto()` / `run_graph()` in orchestrator.py
- FastAPI endpoint in main.py

Draft 有數個 bug 和與 spec 不符之處，需要 audit & fix。

---

## API Spec 摘要（執行時只需兩個端點）

```
GET /v5/ai_agents/{key}
  → manifest: nodes, edges, orchestration_mode, usable, system_prompt, model_config

GET /v5/ai_agents/{key}/context_data?client_id=X&skill_keys[]=...
  → {skills: {skill_key: {system_prompt, model_config, tools, rag_files, context_data}}, errors: {}}
```

`usable` 是唯一的 go/no-go 判斷：`false` → 停止並回傳訊息，`true` → 繼續。

---

## 問題清單（現有 draft vs spec）

| # | 檔案 | 問題 | 修法 |
|---|------|------|------|
| 1 | `main.py` | 缺少 `import asyncio`，執行會爆 NameError | 補上 import |
| 2 | `main.py:554` | 用 `blocked_nodes` 非空作為停止條件 | 改用 `manifest["usable"] == False` |
| 3 | `main.py:604` | `sd.get("description", sk)` — context_data 無 description 欄 | 改用 manifest `node["name"]` |
| 4 | `main.py:615` | `manifest.get("model_config", {})` 未處理 null | 改成 `manifest.get("model_config") or {}` |

---

## 執行流程（修正後）

```
POST /v1/agents/{agent_key}/run  {client_id, message, stream}
  ↓
GET /v5/ai_agents/{key}
  ↓ manifest["usable"] == False
      → 422 + blocked_nodes (optional detail)
  ↓ manifest["usable"] == True
  ↓
GET /v5/ai_agents/{key}/context_data?client_id=X
  （fetch all skills，不過濾）
  ↓
orchestration_mode == "auto"
  → run_auto(): ADK Orchestrator + sub_agents，LLM 決定路由
     ∟ sub-agent description 來自 manifest node["name"]

orchestration_mode == "graph"
  → run_graph(): 拓撲排序，同層平行，下游接上游輸出
     ∟ leaf nodes 結果合併為最終輸出
  ↓
stream == True  → StreamingResponse (SSE)
stream == False → JSONResponse {result, agent_key, mode}
```

---

## 架構決策

### auto 模式：Fetch All First
coordinator LLM 選哪些 skill 是執行期決策，無法在 fetch context 前知道。
→ 一開始就 fetch 所有 skill context，交給 ADK sub_agents 機制路由。
→ 不做兩段式（coordinator 先決定 → 再 fetch），POC 不需要這層複雜度。

### graph streaming：假 streaming
graph 模式 streaming = leaf nodes 全部執行完後，一次 yield 最終輸出。
→ 不做最後 summarizer agent 的真正 SSE。POC 夠用。

### context_data errors 欄
若有 `errors` 非空，只 log warning，不阻斷執行（per spec：per-skill 失敗隔離）。

---

## 不在此次範圍

- `multilingual` locale 注入（`model_config.multilingual` → `context_data.client.locale`）
- graph 模式最後一段真正 SSE streaming
- auto 模式兩段式（coordinator 先選 skill_keys → 再 fetch context）

---

## 需修改的檔案

- `cofit-ai-agent-demo/main.py` — 4 個 fix（見問題清單）
- `cofit-ai-agent-demo/src/orchestrator.py` — 無需改動
- `cofit-ai-agent-demo/src/cofit_api_client.py` — 無需改動

---

## 驗收條件

1. `import asyncio` 在 main.py 頂部存在
2. `manifest["usable"] == False` 時回傳 422，不繼續呼叫 context_data
3. auto 模式 sub-agent description 來自 `node["name"]`
4. `model_config` 為 null 時不崩潰
5. `pytest tests/` 全部通過
