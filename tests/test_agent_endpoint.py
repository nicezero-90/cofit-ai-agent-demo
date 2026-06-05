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


def test_run_agent_usable_false_returns_422_without_context_call():
    """usable == False 時回 422，且不呼叫 context_data API。"""
    manifest = _manifest(usable=False, blocked=[])

    mock_api = MagicMock()
    mock_api.get_ai_agent_manifest.return_value = manifest

    with patch("src.cofit_api_client.CofitApiClient", return_value=mock_api):
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

    with patch("src.cofit_api_client.CofitApiClient", return_value=mock_api):
        with patch("main.run_auto", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "auto 結果"
            response = client.post(
                "/v1/agents/nutrition-agent/run",
                json={"client_id": 351, "message": "分析", "stream": False},
            )

    assert response.status_code == 200
    assert response.json()["result"] == "auto 結果"
    assert response.json()["mode"] == "auto"
