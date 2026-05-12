# 診所 AI 大腦 POC

從 `ai-skill-platform` 的 `/ai-brain` endpoint 獨立出來的 POC 專案，支援 A2A（Agent-to-Agent）多專家編排架構。

## 架構

### v1（單一 Agent）
```
BE ──POST /ai-brain──> FastAPI ──> ADK Agent (Gemini) ──> 回傳結果
```

### v2（Orchestrator + Sub-agents）
```
BE ──POST /ai-brain──> FastAPI ──> Orchestrator（gemini-pro）
                                        │
                                        ├── Sub-agent: lab_report（gemini-flash-lite）
                                        ├── Sub-agent: body_measurement（gemini-flash-lite）
                                        └── Sub-agent: orders（gemini-flash-lite）
                                        │
                                        └── 彙整結果 ──> 回傳
```

**三種編排模式：**
- `auto`：ADK 自動判斷委派哪個 sub-agent（預設）
- `parallel`：全部同時跑，orchestrator 彙整
- `sequential`：依序執行，前一步結果傳給下一步

## 專案結構

```
ai-brain-poc/
├── main.py                    # FastAPI app + /ai-brain endpoint（v1/v2 路由）
├── src/
│   ├── __init__.py
│   ├── orchestrator.py        # A2A 編排邏輯（resolve + auto/parallel/sequential）
│   ├── agent_factory.py       # ADK Agent 建立（create_agent + create_orchestrator）
│   ├── cofit_api_client.py    # BE API client（GET skill config + context_data）
│   └── constants.py           # 環境變數
├── tests/                     # 測試
├── docs/superpowers/          # 設計文件 + 實作計畫
├── requirements.txt
├── Dockerfile
├── .env.example
└── README.md
```

## 快速啟動

```bash
# 安裝依賴
pip install -r requirements.txt

# 設定環境變數
export GCP_PROJECT_ID=cofit-stg
export COFIT_API_URL=https://staging.cofit.me  # v2 remote skill 需要
export COFIT_TOKEN=your-token                   # v2 remote skill 需要

# 本地啟動
python main.py
# or
uvicorn main:app --reload --port 8080
```

## API

### POST /ai-brain

**免認證** — BE 內部呼叫。

#### v1 Request（單一 Agent）：
```json
{
  "key": "doctor_visit_initial",
  "system_prompt": "你是一名經驗豐富的醫師...",
  "model_config": { "model": "gemini-flash" },
  "context_data": {
    "client": { "real_name": "Samuel", "gender": "male", "age": 35 }
  },
  "message": "請根據這位病人的資料，產出初診諮詢稿",
  "stream": false
}
```

#### v2 Request（Orchestrator + Sub-agents）：
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

每個 skill 支援兩種模式：
- **精簡模式**：只帶 `skill_key` + `description`，AI platform 自動打 BE 拿 config
- **完整模式**：帶齊 `system_prompt`、`model_config`、`context_data`，不打 BE

#### Response（非 streaming）：
```json
{
  "result": "Samuel，你好！根據您的檢測報告...",
  "skill_key": "clinic_brain_v2"
}
```

#### Response（streaming, `stream: true`）：
```
data: {"text": "Samuel，你好！"}
data: {"text": "根據您的..."}
data: [DONE]
```

### GET /health

健康檢查。

### GET /docs

Swagger API 文件（FastAPI 自動產生）。

## 向下相容

| 情境 | 行為 |
|------|------|
| 沒傳 `skills` | 跟 v1 完全一樣 |
| 傳 `skills` + `auto` | ADK sub_agent 自動路由 |
| 傳 `skills` + `parallel` | 全部同時跑，orchestrator 彙整 |
| 傳 `skills` + `sequential` | 依序跑，前後串接 context |

## 來源

Clone from `ai-skill-platform` commit at 2026-05-12，加入 A2A 編排架構。
設計文件：`docs/superpowers/specs/2026-05-12-ai-brain-a2a-design.md`
