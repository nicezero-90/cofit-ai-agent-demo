import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from src.orchestrator import resolve_skill_configs


@pytest.mark.asyncio
async def test_resolve_inline_skills():
    """有 system_prompt 的 skill 走 inline，不打 BE。"""
    skills = [
        {
            "skill_key": "lab_report",
            "description": "分析檢驗報告",
            "system_prompt": "你是檢驗報告專家",
            "model_config": {"model": "gemini-flash-lite"},
            "context_data": {"lab": [1, 2, 3]},
        }
    ]
    resolved = await resolve_skill_configs(skills, client_id=351)

    assert len(resolved) == 1
    assert resolved[0]["skill_key"] == "lab_report"
    assert resolved[0]["config"]["system_prompt"] == "你是檢驗報告專家"
    assert resolved[0]["context_data"]["lab"] == [1, 2, 3]


@pytest.mark.asyncio
async def test_resolve_remote_skills():
    """沒有 system_prompt 的 skill 要打 BE API。"""
    mock_client = MagicMock()
    mock_client.get_context_data.return_value = (
        {"system_prompt": "from BE", "model_config": {"model": "gemini-flash-lite"}, "tools": []},
        {"lab_results": [{"item": "VitD"}]},
    )

    skills = [
        {"skill_key": "lab_report", "description": "分析檢驗報告"}
    ]

    with patch("src.orchestrator._get_api_client", return_value=mock_client):
        resolved = await resolve_skill_configs(skills, client_id=351)

    assert len(resolved) == 1
    assert resolved[0]["config"]["system_prompt"] == "from BE"
    assert resolved[0]["context_data"]["lab_results"][0]["item"] == "VitD"
    mock_client.get_context_data.assert_called_once_with("lab_report", client_id=351)


@pytest.mark.asyncio
async def test_resolve_mixed_skills():
    """混合 inline + remote，remote 失敗時跳過。"""
    mock_client = MagicMock()
    mock_client.get_context_data.side_effect = Exception("BE timeout")

    skills = [
        {
            "skill_key": "lab_report",
            "description": "inline",
            "system_prompt": "inline prompt",
            "model_config": {"model": "gemini-flash-lite"},
            "context_data": {},
        },
        {"skill_key": "body_measurement", "description": "remote, will fail"},
    ]

    with patch("src.orchestrator._get_api_client", return_value=mock_client):
        resolved = await resolve_skill_configs(skills, client_id=351)

    assert len(resolved) == 1
    assert resolved[0]["skill_key"] == "lab_report"


@pytest.mark.asyncio
async def test_resolve_all_remote_fail():
    """所有 remote skill 都失敗時，回傳空 list。"""
    mock_client = MagicMock()
    mock_client.get_context_data.side_effect = Exception("BE down")

    skills = [
        {"skill_key": "lab_report", "description": "fail"},
        {"skill_key": "orders", "description": "fail too"},
    ]

    with patch("src.orchestrator._get_api_client", return_value=mock_client):
        resolved = await resolve_skill_configs(skills, client_id=351)

    assert resolved == []
