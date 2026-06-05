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
