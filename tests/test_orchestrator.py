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


from src.orchestrator import run_auto


@pytest.mark.asyncio
async def test_run_auto_builds_orchestrator_and_runs():
    """run_auto 應建立 orchestrator + sub_agents 並執行。"""
    resolved_skills = [
        {
            "skill_key": "lab_report",
            "description": "分析檢驗報告",
            "config": {
                "system_prompt": "你是檢驗報告專家",
                "model_config": {"model": "gemini-flash-lite"},
                "tools": [],
                "rag_files": [],
            },
            "context_data": {"lab": "data"},
        },
    ]

    mock_event = MagicMock()
    mock_event.content = MagicMock()
    mock_event.content.parts = [MagicMock(text="檢驗結果正常")]

    async def mock_run_async(**kwargs):
        yield mock_event

    with patch("src.orchestrator.Runner") as MockRunner:
        MockRunner.return_value.run_async = mock_run_async
        result = await run_auto(
            resolved_skills=resolved_skills,
            system_prompt="你是診所AI大腦",
            model_config={"model": "gemini-pro"},
            skill_key="clinic_brain_v2",
            message="分析檢驗報告",
        )

    assert "檢驗結果正常" in result


from src.orchestrator import run_parallel


@pytest.mark.asyncio
async def test_run_parallel_gathers_and_summarizes():
    """parallel 應同時跑所有 skill，最後 orchestrator 彙整。"""
    resolved_skills = [
        {
            "skill_key": "lab_report",
            "description": "檢驗",
            "config": {
                "system_prompt": "你是檢驗專家",
                "model_config": {"model": "gemini-flash-lite"},
                "tools": [], "rag_files": [],
            },
            "context_data": {"lab": "data"},
        },
        {
            "skill_key": "body_measurement",
            "description": "體組成",
            "config": {
                "system_prompt": "你是體組成專家",
                "model_config": {"model": "gemini-flash-lite"},
                "tools": [], "rag_files": [],
            },
            "context_data": {"body": "data"},
        },
    ]

    call_count = 0

    async def mock_run_async(**kwargs):
        nonlocal call_count
        call_count += 1
        mock_event = MagicMock()
        mock_event.content = MagicMock()
        mock_event.content.parts = [MagicMock(text=f"結果{call_count}")]
        yield mock_event

    with patch("src.orchestrator.Runner") as MockRunner:
        MockRunner.return_value.run_async = mock_run_async
        result = await run_parallel(
            resolved_skills=resolved_skills,
            system_prompt="你是診所AI大腦",
            model_config={"model": "gemini-pro"},
            skill_key="clinic_brain_v2",
            message="全面分析",
        )

    assert result  # 有回傳結果


from src.orchestrator import run_sequential


@pytest.mark.asyncio
async def test_run_sequential_chains_context():
    """sequential 應依序執行，前一步結果傳給下一步。"""
    resolved_skills = [
        {
            "skill_key": "lab_report",
            "description": "檢驗",
            "config": {
                "system_prompt": "你是檢驗專家",
                "model_config": {"model": "gemini-flash-lite"},
                "tools": [], "rag_files": [],
            },
            "context_data": {},
        },
        {
            "skill_key": "orders",
            "description": "訂單",
            "config": {
                "system_prompt": "你是訂單專家",
                "model_config": {"model": "gemini-flash-lite"},
                "tools": [], "rag_files": [],
            },
            "context_data": {},
        },
    ]

    step = 0

    async def mock_run_async(**kwargs):
        nonlocal step
        step += 1
        mock_event = MagicMock()
        mock_event.content = MagicMock()
        mock_event.content.parts = [MagicMock(text=f"步驟{step}結果")]
        yield mock_event

    with patch("src.orchestrator.Runner") as MockRunner:
        MockRunner.return_value.run_async = mock_run_async
        result = await run_sequential(
            resolved_skills=resolved_skills,
            system_prompt="你是診所AI大腦",
            model_config={"model": "gemini-pro"},
            skill_key="clinic_brain_v2",
            message="先看報告再建議產品",
        )

    assert result  # 有回傳結果
    # 3 次呼叫：2 sub-agent + 1 orchestrator 彙整
    assert step == 3
