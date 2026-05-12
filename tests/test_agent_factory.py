import pytest
from unittest.mock import patch, MagicMock
from google.adk.agents import Agent
from src.agent_factory import create_agent, create_orchestrator, resolve_model


def test_resolve_model_known_alias():
    assert resolve_model("gemini-pro") == "gemini-2.5-pro-preview-05-06"
    assert resolve_model("gemini-flash") == "gemini-3-flash-preview"


def test_resolve_model_unknown_passthrough():
    assert resolve_model("some-custom-model") == "some-custom-model"


def test_create_agent_returns_agent_and_parts():
    """create_agent 應回傳 (Agent, knowledge_parts) tuple。"""
    config = {
        "system_prompt": "你是測試專家",
        "model_config": {"model": "gemini-flash-lite"},
        "tools": [],
        "rag_files": [],
    }
    agent, parts = create_agent(config=config, context_data={"name": "test"}, skill_key="test_skill")
    assert agent.name == "brain_test_skill"
    assert parts == []


def test_create_orchestrator_with_sub_agents():
    """create_orchestrator 應建立含 sub_agents 的 orchestrator Agent。"""
    sub_agent_1 = Agent(name="skill_lab_report", model="gemini-2.0-flash", instruction="test")
    sub_agent_2 = Agent(name="skill_body_measurement", model="gemini-2.0-flash", instruction="test")

    orchestrator = create_orchestrator(
        system_prompt="你是診所AI大腦的總指揮",
        model_config={"model": "gemini-pro"},
        sub_agents=[sub_agent_1, sub_agent_2],
        skill_key="clinic_brain_v2",
    )

    assert orchestrator.name == "orchestrator_clinic_brain_v2"
    assert len(orchestrator.sub_agents) == 2


def test_create_orchestrator_no_sub_agents():
    """沒有 sub_agents 時應 raise ValueError。"""
    with pytest.raises(ValueError, match="at least one sub_agent"):
        create_orchestrator(
            system_prompt="test",
            model_config={"model": "gemini-pro"},
            sub_agents=[],
            skill_key="test",
        )
