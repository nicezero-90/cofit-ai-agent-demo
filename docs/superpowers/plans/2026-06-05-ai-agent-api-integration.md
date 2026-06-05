# AI Agent API 整合 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修正 `cofit-ai-agent-demo` 中 `/v1/agents/{agent_key}/run` 端點的 4 個 bug，使其與 Cofit AI Agents API spec（2026-06-05）完全對齊。

**Architecture:** Audit & Fix — 保留現有架構（FastAPI + ADK orchestrator），只修正 `main.py` 中的錯誤。`orchestrator.py` 和 `cofit_api_client.py` 無需改動。

**Tech Stack:** Python 3.12+、FastAPI、Google ADK、pytest、pytest-asyncio

---

## File Map

| 動作 | 路徑 | 說明 |
|------|------|------|
| Modify | `main.py` | 補 import asyncio + 3 個 fix |
| Modify | `tests/test_agent_endpoint.py` | 新增 `/v1/agents/{key}/run` 的測試（現有檔案無覆蓋） |

> `src/orchestrator.py` 和 `src/cofit_api_client.py` **不動**。

---

## Task 1: 補 `import asyncio` + 驗證

### 問題
`main.py` 在 `/v1/agents/{agent_key}/run` 的 handler 中使用 `asyncio.get_event_loop()`（line 541、566），但頂部沒有 `import asyncio`。執行時會 `NameError: name 'asyncio' is not defined`。

**Files:**
- Modify: `main.py`（line 1–5 附近）
- Test: `tests/test_agent_endpoint.py`

- [ ] **Step 1: 寫失敗的測試**

在 `tests/test_agent_endpoint.py` 建立新檔（如果不存在）：

```python
# tests/test_agent_endpoint.py
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def _manifest(usable=True, mode="auto", blocked=None):
    """Helper: 建立 manifest fixture。"""
    return {
        "key": "nutrition-agent",
        "orchestration_mode": mode,
        "usable": usable,
        "blocked_nodes": blocked or [],
        "nodes": [
            {"node_id": "n1", "skill_key": "daily_diet_summary", "name": "每日飲食摘要",
             "status": "active", "position_x": 0, "position_y": 0},
        ],
        "edges": [],
        "system_prompt": "你是營養師",
        "model_config": {"model": "gemini-flash"},
        "tools": [],
    }


def _context_data():
    return {
        "skills": {
            "daily_diet_summary": {
                "key": "daily_diet_summary",
                "system_prompt": "分析每日飲食",
                "model_config": {"model": "gemini-flash-lite"},
                "tools": [],
                "rag_files": [],
                "context_data": {"client": {"real_name": "Test User"}},
            }
        },
        "errors": {},
    }


def test_asyncio_import_exists():
    """main.py 必須有 import asyncio，否則 run_agent endpoint 執行時會 NameError。"""
    import main
    import inspect
    source = inspect.getsource(main)
    assert "import asyncio" in source, "main.py 缺少 import asyncio"
```

- [ ] **Step 2: 執行測試，確認失敗**

```bash
cd /Users/luanjiulin/Documents/cofit/cofit-ai-agent-demo
pytest tests/test_agent_endpoint.py::test_asyncio_import_exists -v
```

預期：**FAIL** — `AssertionError: main.py 缺少 import asyncio`

- [ ] **Step 3: 在 main.py 補 import asyncio**

開啟 `main.py`，在現有 `import json` / `import logging` / `import os` 之後加一行：

```python
import asyncio
import json
import logging
import os
```

（維持 stdlib import 的字母順序）

- [ ] **Step 4: 執行測試，確認通過**

```bash
pytest tests/test_agent_endpoint.py::test_asyncio_import_exists -v
```

預期：**PASS**

- [ ] **Step 5: Commit**

```bash
cd /Users/luanjiulin/Documents/cofit/cofit-ai-agent-demo
git add main.py tests/test_agent_endpoint.py
git commit -m "fix: add missing import asyncio in main.py; add agent endpoint tests"
```

---

## Task 2: 改用 `usable` 作為 go/no-go 判斷

### 問題
`main.py` line 554–561 檢查 `blocked_nodes` 非空才停止。  
Spec 規定用 `manifest["usable"] == False` 作為唯一停止條件，且 `usable == False` 時**不應呼叫** `context_data` API。

**Files:**
- Modify: `main.py`（`run_agent` function，`blocked_nodes` 判斷區段）
- Test: `tests/test_agent_endpoint.py`

- [ ] **Step 1: 寫失敗的測試**

在 `tests/test_agent_endpoint.py` 新增：

```python
def test_run_agent_usable_false_returns_422_without_context_call():
    """usable == False 時回 422，且不呼叫 context_data API。"""
    manifest = _manifest(usable=False, blocked=[{"node_id": "n1", "reason": "inactive"}])

    mock_api = MagicMock()
    mock_api.get_ai_agent_manifest.return_value = manifest

    with patch("main.CofitApiClient", return_value=mock_api):
        response = client.post("/v1/agents/nutrition-agent/run", json={"client_id": 351})

    assert response.status_code == 422
    assert response.json()["error"] == "Agent has blocked nodes"
    # context_data 不應被呼叫
    mock_api.get_ai_agent_context_data.assert_not_called()


def test_run_agent_usable_true_proceeds():
    """usable == True 且 auto 模式，正常走完並回 200。"""
    manifest = _manifest(usable=True, mode="auto")
    ctx = _context_data()

    mock_api = MagicMock()
    mock_api.get_ai_agent_manifest.return_value = manifest
    mock_api.get_ai_agent_context_data.return_value = ctx

    with patch("main.CofitApiClient", return_value=mock_api):
        with patch("main.run_auto", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "auto 結果"
            response = client.post(
                "/v1/agents/nutrition-agent/run",
                json={"client_id": 351, "message": "分析", "stream": False},
            )

    assert response.status_code == 200
    assert response.json()["result"] == "auto 結果"
    assert response.json()["mode"] == "auto"
```

- [ ] **Step 2: 執行測試，確認失敗**

```bash
pytest tests/test_agent_endpoint.py::test_run_agent_usable_false_returns_422_without_context_call tests/test_agent_endpoint.py::test_run_agent_usable_true_proceeds -v
```

預期：兩個測試都 **FAIL**（`usable` 欄位沒被讀取，`context_data` 可能被呼叫）

- [ ] **Step 3: 修改 main.py — 改用 usable 判斷**

找到 `run_agent` function 中的這段（約 line 554–561）：

```python
# 舊：
blocked = manifest.get("blocked_nodes") or []
if blocked:
    logger.warning("Agent '%s' has blocked_nodes: %s", agent_key, blocked)
    return JSONResponse(
        status_code=422,
        content={"error": "Agent has blocked nodes", "blocked_nodes": blocked},
    )
```

改為：

```python
# 新：
if not manifest.get("usable", True):
    blocked = manifest.get("blocked_nodes") or []
    logger.warning("Agent '%s' is not usable, blocked_nodes: %s", agent_key, blocked)
    return JSONResponse(
        status_code=422,
        content={"error": "Agent has blocked nodes", "blocked_nodes": blocked},
    )
```

- [ ] **Step 4: 執行測試，確認通過**

```bash
pytest tests/test_agent_endpoint.py::test_run_agent_usable_false_returns_422_without_context_call tests/test_agent_endpoint.py::test_run_agent_usable_true_proceeds -v
```

預期：兩個測試都 **PASS**

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_agent_endpoint.py
git commit -m "fix: use manifest usable flag as go/no-go gate per API spec"
```

---

## Task 3: auto 模式 description 改用 node name

### 問題
`main.py` line 604：`sd.get("description", sk)` — `context_data` 回傳的 skill 物件沒有 `description` 欄位，所以 sub-agent 的 description 永遠退化為 `skill_key`（無意義的 key 字串），ADK orchestrator 無法正確路由。  
應改用 manifest `node["name"]`（e.g. `"每日飲食摘要"`）。

**Files:**
- Modify: `main.py`（`run_agent` function，auto 模式 resolved_skills 組建區段）
- Test: `tests/test_agent_endpoint.py`

- [ ] **Step 1: 寫失敗的測試**

在 `tests/test_agent_endpoint.py` 新增：

```python
def test_run_agent_auto_uses_node_name_as_description():
    """auto 模式組 resolved_skills 時，description 應來自 manifest node['name']，不是 skill_key。"""
    manifest = _manifest(usable=True, mode="auto")
    ctx = _context_data()

    mock_api = MagicMock()
    mock_api.get_ai_agent_manifest.return_value = manifest
    mock_api.get_ai_agent_context_data.return_value = ctx

    captured = {}

    async def capture_run_auto(resolved_skills, **kwargs):
        captured["descriptions"] = [s["description"] for s in resolved_skills]
        return "ok"

    with patch("main.CofitApiClient", return_value=mock_api):
        with patch("main.run_auto", side_effect=capture_run_auto):
            client.post(
                "/v1/agents/nutrition-agent/run",
                json={"client_id": 351, "stream": False},
            )

    # node["name"] == "每日飲食摘要"，不是 skill_key "daily_diet_summary"
    assert captured.get("descriptions") == ["每日飲食摘要"]
```

- [ ] **Step 2: 執行測試，確認失敗**

```bash
pytest tests/test_agent_endpoint.py::test_run_agent_auto_uses_node_name_as_description -v
```

預期：**FAIL** — `assert ['daily_diet_summary'] == ['每日飲食摘要']`

- [ ] **Step 3: 修改 main.py — 改用 node name**

找到 auto 模式組 `resolved_skills` 的 for loop（約 line 592–610）：

```python
# 舊：
for node in manifest.get("nodes", []):
    sk = node["skill_key"]
    if sk in seen_skills:
        continue
    seen_skills.add(sk)
    sd = skills_data.get(sk, {})
    resolved_skills.append({
        "skill_key": sk,
        "description": sd.get("description", sk),   # <-- 錯誤
        ...
    })
```

改為：

```python
# 新：
for node in manifest.get("nodes", []):
    sk = node["skill_key"]
    if sk in seen_skills:
        continue
    seen_skills.add(sk)
    sd = skills_data.get(sk, {})
    resolved_skills.append({
        "skill_key": sk,
        "description": node.get("name") or sk,       # <-- 用 node name
        "config": {
            "system_prompt": sd.get("system_prompt", ""),
            "model_config": sd.get("model_config") or {},
            "tools": sd.get("tools", []),
            "rag_files": sd.get("rag_files", []),
            "rag_resource_name": sd.get("rag_resource_name"),
        },
        "context_data": sd.get("context_data", {}),
    })
```

- [ ] **Step 4: 執行測試，確認通過**

```bash
pytest tests/test_agent_endpoint.py::test_run_agent_auto_uses_node_name_as_description -v
```

預期：**PASS**

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_agent_endpoint.py
git commit -m "fix: use node name as sub-agent description in auto mode"
```

---

## Task 4: model_config null guard

### 問題
Spec 說 `model_config` 可為 `null`。`main.py` line 615：`manifest.get("model_config", {})` 當 BE 回傳 `"model_config": null` 時，`manifest.get("model_config", {})` 會回傳 `None`（default 只在 key 不存在時觸發，key 存在但值為 null 時不生效），導致後續 `model_config.get("model", ...)` 崩潰。

**Files:**
- Modify: `main.py`（auto 模式呼叫 `run_auto` 的那行，以及 `run_agent` 中的 `sd.get("model_config", {})` 改為 `sd.get("model_config") or {}`）
- Test: `tests/test_agent_endpoint.py`

- [ ] **Step 1: 寫失敗的測試**

在 `tests/test_agent_endpoint.py` 新增：

```python
def test_run_agent_null_model_config_does_not_crash():
    """manifest 的 model_config 為 null 時，端點不應崩潰。"""
    manifest = _manifest(usable=True, mode="auto")
    manifest["model_config"] = None   # <-- null

    ctx = _context_data()
    ctx["skills"]["daily_diet_summary"]["model_config"] = None  # skill level 也 null

    mock_api = MagicMock()
    mock_api.get_ai_agent_manifest.return_value = manifest
    mock_api.get_ai_agent_context_data.return_value = ctx

    with patch("main.CofitApiClient", return_value=mock_api):
        with patch("main.run_auto", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "ok"
            response = client.post(
                "/v1/agents/nutrition-agent/run",
                json={"client_id": 351, "stream": False},
            )

    assert response.status_code == 200
```

- [ ] **Step 2: 執行測試，確認失敗**

```bash
pytest tests/test_agent_endpoint.py::test_run_agent_null_model_config_does_not_crash -v
```

預期：**FAIL** — 500 error 或 `AttributeError: 'NoneType' object has no attribute 'get'`

- [ ] **Step 3: 修改 main.py — 兩處加 null guard**

**第一處**（auto 模式呼叫 `run_auto` 前，manifest level model_config）：

```python
# 舊：
result_or_gen = await run_auto(
    resolved_skills=resolved_skills,
    system_prompt=manifest.get("system_prompt", ""),
    model_config=manifest.get("model_config", {}),
    ...
)

# 新：
result_or_gen = await run_auto(
    resolved_skills=resolved_skills,
    system_prompt=manifest.get("system_prompt", ""),
    model_config=manifest.get("model_config") or {},
    ...
)
```

**第二處**（auto 模式組 resolved_skills，skill level model_config，Task 3 中已改成 `sd.get("model_config") or {}`，確認是否已正確）：

確認 Task 3 修改後的程式碼裡 `model_config` 那行是：

```python
"model_config": sd.get("model_config") or {},
```

如果 Task 3 已正確寫，這裡不需要額外修改。

- [ ] **Step 4: 執行測試，確認通過**

```bash
pytest tests/test_agent_endpoint.py::test_run_agent_null_model_config_does_not_crash -v
```

預期：**PASS**

- [ ] **Step 5: 跑全部測試，確認沒有回歸**

```bash
pytest tests/ -v
```

預期：全部 **PASS**（含之前的 `test_main_v2.py`、`test_orchestrator.py`）

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_agent_endpoint.py
git commit -m "fix: null guard for model_config in run_agent endpoint"
```

---

## Task 5: 最終驗收

- [ ] **Step 1: 跑完整測試套件**

```bash
cd /Users/luanjiulin/Documents/cofit/cofit-ai-agent-demo
pytest tests/ -v --tb=short
```

預期輸出（所有測試 PASS）：
```
tests/test_agent_endpoint.py::test_asyncio_import_exists PASSED
tests/test_agent_endpoint.py::test_run_agent_usable_false_returns_422_without_context_call PASSED
tests/test_agent_endpoint.py::test_run_agent_usable_true_proceeds PASSED
tests/test_agent_endpoint.py::test_run_agent_auto_uses_node_name_as_description PASSED
tests/test_agent_endpoint.py::test_run_agent_null_model_config_does_not_crash PASSED
tests/test_main_v2.py::test_v1_no_skills_unchanged PASSED
tests/test_main_v2.py::test_v2_with_skills_routes_to_orchestrator PASSED
tests/test_main_v2.py::test_v2_error_returns_500 PASSED
tests/test_orchestrator.py::... PASSED (all)
```

- [ ] **Step 2: 確認 spec 覆蓋**

| Spec 要求 | 測試 | 狀態 |
|----------|------|------|
| `usable == false` → 422，不呼叫 context_data | `test_run_agent_usable_false_returns_422_without_context_call` | ✓ |
| `usable == true` → 正常執行 | `test_run_agent_usable_true_proceeds` | ✓ |
| auto 模式 description 來自 node name | `test_run_agent_auto_uses_node_name_as_description` | ✓ |
| model_config null 不崩潰 | `test_run_agent_null_model_config_does_not_crash` | ✓ |
| import asyncio 存在 | `test_asyncio_import_exists` | ✓ |

- [ ] **Step 3: Final commit（如有殘留變更）**

```bash
git status
# 確認沒有 untracked 變更，或 commit 殘留
```
