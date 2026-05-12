# AI Brain A2A 架構設計

**Date:** 2026-05-12
**Ticket:** [AI] AI skill A2A 架構設計 (5 SP) — 86exfun5u
**專案:** ai-brain-poc

## 概述

將診所 AI 大腦從單一 Agent 升級為 Orchestrator + Sub-agents 的 A2A 架構。Orchestrator 用好模型做路由判斷，各 skill sub-agent 用便宜模型跑分析。三種編排模式（auto / parallel / sequential）滿足不同場景。

**核心原則：向下相容，沒傳 `skills` 就跟 v1 行為完全一樣。**

## API 格式

### Request（POST /ai-brain）

```json
{
  "key": "clinic_brain_v2",
  "system_prompt": "你是診所AI大腦，根據問題委派給合適的專家分析...",
  "model_config": { "model": "gemini-pro" },
  "context_data": { "client_id": 351 },
  "message": "這個客戶最近檢驗報告和體重變化怎麼樣？",

  "skills": [
    { "skill_key": "lab_report", "description": "分析檢驗報告數據，判斷異常值並給建議" },
    { "skill_key": "body_measurement", "description": "分析體組成與身體數據變化趨勢" },
    { "skill_key": "orders", "description": "查詢與分析客戶訂單紀錄" }
  ],
  "orchestration_mode": "auto",
  "stream": false
}
```

**新增欄位：**

| 欄位 | 型別 | 必填 | 說明 |
|------|------|------|------|
| skills | array | 否 | Sub-agent 清單，不傳則走 v1 流程 |
| skills[].skill_key | string | 是 | 對應 BE `/v5/ai_skills/:key` |
| skills[].description | string | 是 | 給 orchestrator 判斷路由用 |
| skills[].system_prompt | string | 否 | 有值就 inline 使用，沒值就打 BE 拿 |
| skills[].model_config | object | 否 | inline 模式的 model 設定 |
| skills[].context_data | object | 否 | inline 模式的 context 資料 |
| skills[].tools | array | 否 | inline 模式的 tools |
| skills[].rag_files | array | 否 | inline 模式的知識庫檔案 |
| orchestration_mode | string | 否 | `auto`（預設）/ `parallel` / `sequential` |

**v1 原有欄位全部保留不變。**

### Response

與 v1 相同：

```json
{ "result": "...", "skill_key": "clinic_brain_v2" }
```

Streaming 模式同 v1：`data: {"text": "..."}\n\n` + `data: [DONE]\n\n`

## Skill Config 解析（Fallback 機制）

每個 skill item 支援兩種模式：

### 精簡模式（去 BE 拿 config）

```json
{ "skill_key": "lab_report", "description": "分析檢驗報告" }
```

AI platform 呼叫 `GET /v5/ai_skills/lab_report/context_data?client_id=351`，拿回完整 config + context_data。

### 完整模式（inline，不打 BE）

```json
{
  "skill_key": "lab_report",
  "description": "分析檢驗報告",
  "system_prompt": "你是檢驗報告分析專家...",
  "model_config": { "model": "gemini-flash-lite" },
  "context_data": { "lab_results": [...] }
}
```

**判斷邏輯：** skill item 有 `system_prompt` → inline 模式，沒有 → 去 BE 拿。

多個 skill 需要打 BE 時，用 `asyncio.gather` 並行 fetch。

`client_id` 從 request 頂層 `context_data.client_id` 取得。

## Orchestration Modes

### auto — ADK 自動路由

```
醫師問題 → orchestrator（gemini-pro, sub_agents=[...])
              ↓ ADK 根據 description 自動判斷委派哪個 skill
           → 回覆結果
```

直接用 ADK 原生 sub_agents 機制，orchestrator 的 instruction 帶上每個 skill 的 description。

### parallel — 全部同時跑，orchestrator 彙整

```
醫師問題 → asyncio.gather 同時跑所有 sub-agent
              lab_report_agent → 「報告顯示維生素D偏低...」
              body_measurement_agent → 「體脂率下降2%...」
              orders_agent → 「上次購買了維生素D補充品...」
           → orchestrator 收到三份結果，寫成一份完整報告
           → 回覆結果
```

N+1 次 LLM 呼叫（N 個 skill + 1 次 orchestrator 彙整）。

### sequential — 依序執行，前後串接

```
醫師問題 → lab_report_agent 先分析
              「報告顯示維生素D偏低...」
           → body_measurement_agent 收到前一步結果 + 自己的資料
              「結合檢驗報告，體脂下降但維生素D不足...」
           → orders_agent 收到前面所有結果
              「根據以上分析，建議調整補充品用量...」
           → orchestrator 做最終彙整
           → 回覆結果
```

N+1 次 LLM 呼叫，每步有上下文傳遞。

### Streaming 支援

| Mode | Streaming 行為 |
|------|----------------|
| auto | 直接 stream orchestrator 的 SSE 輸出 |
| parallel | sub-agent 不 stream，只 stream 最後 orchestrator 彙整 |
| sequential | sub-agent 不 stream，只 stream 最後 orchestrator 彙整 |

## 向下相容

| 情境 | 行為 |
|------|------|
| 沒傳 `skills` | 跟 v1 完全一樣，orchestrator 自己回答 |
| 傳 `skills` + `auto` | ADK sub_agent 自動路由 |
| 傳 `skills` + 只傳一個 skill | 等同 BE 指定用哪個 |

## Model Config 層級

| 角色 | Model | 說明 |
|------|-------|------|
| Orchestrator | gemini-pro | 需要理解力做路由判斷 |
| Sub-agent | 各 skill config 自帶 | 便宜快速，由 BE 控制 |

各層各自設，互不干擾。

## Error Handling

| 情境 | 處理 |
|------|------|
| 某個 skill 的 BE API 打不到（timeout/500） | 跳過該 skill，log warning，其他繼續 |
| 所有 skill 都打 BE 失敗 | 回 500 `{"error": "Failed to fetch skill configs"}` |
| parallel/sequential 某個 sub-agent 執行失敗 | 該 skill 標記 error，其他結果照常彙整 |
| orchestration_mode 不認識的值 | fallback 到 `auto` |

原則：盡量給結果，不因一個 skill 失敗就整個掛掉。

## 檔案結構

```
ai-brain-poc/
├── main.py                    # HTTP 層
│                                - v1（無 skills）→ 直接建 agent 跑
│                                - v2（有 skills）→ 呼叫 orchestrator
├── src/
│   ├── orchestrator.py        # 編排層（新增）
│   │    - resolve_skill_configs()  → 解析 skills[]，inline 或打 BE
│   │    - run_auto()               → ADK sub_agents 自動路由
│   │    - run_parallel()           → gather 全部跑，orchestrator 彙整
│   │    - run_sequential()         → 依序跑，串接 context
│   │
│   ├── agent_factory.py       # Agent 建立（小改）
│   │    - create_agent()           → 建單一 agent（v1 + sub-agent 共用）
│   │    - create_orchestrator()    → 建 orchestrator agent（新增）
│   │
│   ├── cofit_api_client.py    # BE API client（從 ai-skill-platform 搬）
│   │    - get_context_data()       → GET /v5/ai_skills/:key/context_data
│   │
│   └── constants.py           # 環境變數（加 COFIT_API_URL、COFIT_TOKEN）
├── requirements.txt
├── Dockerfile
└── README.md
```

## 不在 Scope 內

- save_result tool（存結果回 BE）
- 認證 middleware
- PostgreSQL session 持久化
- LINE / Vitera 整合
- 前端 UI
