# 診所 AI 大腦 POC

從 `ai-skill-platform` 的 `/ai-brain` endpoint 獨立出來的 POC 專案。

## 架構

BE 送完整 config（system_prompt、model_config、tools、rag_files、context_data），AI platform 建立 Gemini Agent 執行後回傳結果。

```
BE ──POST /ai-brain──> FastAPI ──> ADK Agent (Gemini) ──> 回傳結果
                                      │
                                      ├── Google Search（原生 tool）
                                      ├── RAG 知識庫檢索（Vertex AI）
                                      └── full_context 知識庫注入
```

## 專案結構

```
ai-brain-poc/
├── main.py                 # FastAPI app + /ai-brain endpoint
├── src/
│   ├── __init__.py
│   ├── agent_factory.py    # ADK Agent 建立邏輯
│   └── constants.py        # 環境變數
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

# 本地啟動
python main.py
# or
uvicorn main:app --reload --port 8080
```

## API

### POST /ai-brain

**免認證** — BE 內部呼叫。

**Request：**
```json
{
  "key": "doctor_visit_initial",
  "system_prompt": "你是一名經驗豐富的醫師...",
  "model_config": { "model": "gemini-flash" },
  "tools": ["google_search"],
  "rag_files": [
    {
      "file_name": "初日診所營養品指南.json",
      "url": "https://storage.googleapis.com/...",
      "rag_mode": "full_context"
    }
  ],
  "context_data": {
    "client": { "real_name": "Samuel", "gender": "male", "age": 35 }
  },
  "message": "請根據這位病人的資料，產出初診諮詢稿",
  "stream": false
}
```

**Response（非 streaming）：**
```json
{
  "result": "Samuel，你好！根據您的檢測報告...",
  "skill_key": "doctor_visit_initial"
}
```

**Response（streaming, `stream: true`）：**
```
data: {"text": "Samuel，你好！"}
data: {"text": "根據您的..."}
data: [DONE]
```

### GET /health

健康檢查。

### GET /docs

Swagger API 文件（FastAPI 自動產生）。

## 與 ai-skill-platform 的差異

| 項目 | ai-skill-platform | ai-brain-poc |
|------|-------------------|--------------|
| Endpoints | /ai-brain, /ai-chat, /webhook/line, /vitera/run | 僅 /ai-brain |
| Auth middleware | JWT 認證 + bypass | 無（POC 免認證） |
| Skills | nutrigo, ai_expert, doctor_visit 等 | 通用（由 BE config 決定） |
| Session | PostgreSQL + InMemory | InMemory only |
| A2A sub-agent | diet_photo_analyzer | 無 |
| LINE integration | 完整 webhook | 無 |

## 來源

Clone from `ai-skill-platform` commit at 2026-05-12，僅保留 `/ai-brain` 相關程式碼。
