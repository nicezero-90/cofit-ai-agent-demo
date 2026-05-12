# tests/test_cofit_api_client.py
import pytest
from unittest.mock import patch, MagicMock
from src.cofit_api_client import CofitApiClient


def test_get_context_data_returns_config_and_context():
    """get_context_data 應回傳 (config, context_data) tuple。"""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "system_prompt": "你是檢驗報告專家",
        "model_config": {"model": "gemini-flash-lite"},
        "tools": [],
        "context_data": {"lab_results": [{"item": "VitD", "value": 18}]},
    }
    mock_response.raise_for_status = MagicMock()

    with patch("src.cofit_api_client.requests.Session") as MockSession:
        MockSession.return_value.get.return_value = mock_response
        MockSession.return_value.mount = MagicMock()
        MockSession.return_value.headers = {}

        client = CofitApiClient(base_url="https://test.cofit.me", token="test-token")
        client.session = MockSession.return_value

        config, context_data = client.get_context_data("lab_report", client_id=351)

    assert config["system_prompt"] == "你是檢驗報告專家"
    assert "context_data" not in config
    assert context_data["lab_results"][0]["item"] == "VitD"


def test_get_context_data_missing_context_data():
    """BE 回傳沒有 context_data 時，應回傳空 dict。"""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "system_prompt": "test",
        "model_config": {"model": "gemini-flash-lite"},
    }
    mock_response.raise_for_status = MagicMock()

    with patch("src.cofit_api_client.requests.Session") as MockSession:
        MockSession.return_value.get.return_value = mock_response
        MockSession.return_value.mount = MagicMock()
        MockSession.return_value.headers = {}

        client = CofitApiClient(base_url="https://test.cofit.me", token="t")
        client.session = MockSession.return_value

        config, context_data = client.get_context_data("lab_report", client_id=351)

    assert context_data == {}
