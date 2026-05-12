# tests/test_main_v2.py
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_v1_no_skills_unchanged():
    """沒傳 skills → 走 v1 路徑。"""
    with patch("main.create_agent") as mock_create:
        mock_agent = MagicMock()
        mock_create.return_value = (mock_agent, [])

        async def mock_run(**kwargs):
            event = MagicMock()
            event.content = MagicMock()
            event.content.parts = [MagicMock(text="v1 結果")]
            yield event

        with patch("main.Runner") as MockRunner:
            MockRunner.return_value.run_async = mock_run
            with patch("main.InMemorySessionService") as MockSession:
                mock_session = MagicMock()
                mock_session.id = "test-session"
                MockSession.return_value.create_session = AsyncMock(return_value=mock_session)

                response = client.post("/ai-brain", json={
                    "key": "doctor_visit_initial",
                    "system_prompt": "你是醫師",
                    "model_config": {"model": "gemini-flash"},
                    "context_data": {"client": {"name": "test"}},
                })

    assert response.status_code == 200
    assert response.json()["skill_key"] == "doctor_visit_initial"


def test_v2_with_skills_routes_to_orchestrator():
    """傳 skills → 走 v2 orchestrator 路徑。"""
    with patch("main.run_brain_v2", new_callable=AsyncMock) as mock_v2:
        mock_v2.return_value = "orchestrator 結果"

        response = client.post("/ai-brain", json={
            "key": "clinic_brain_v2",
            "system_prompt": "你是診所AI大腦",
            "model_config": {"model": "gemini-pro"},
            "context_data": {"client_id": 351},
            "message": "分析報告",
            "skills": [
                {"skill_key": "lab_report", "description": "檢驗報告"}
            ],
            "orchestration_mode": "auto",
        })

    assert response.status_code == 200
    assert response.json()["result"] == "orchestrator 結果"
    mock_v2.assert_called_once()


def test_v2_error_returns_500():
    """v2 執行失敗應回 500。"""
    with patch("main.run_brain_v2", new_callable=AsyncMock) as mock_v2:
        mock_v2.side_effect = Exception("Agent crashed")

        response = client.post("/ai-brain", json={
            "key": "clinic_brain_v2",
            "system_prompt": "你是診所AI大腦",
            "model_config": {"model": "gemini-pro"},
            "context_data": {"client_id": 351},
            "skills": [
                {"skill_key": "lab_report", "description": "test"}
            ],
        })

    assert response.status_code == 500
    assert "Agent execution failed" in response.json()["error"]
